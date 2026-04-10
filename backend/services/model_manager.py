"""
VeritasAI – Model Version Manager
Tracks versioned model checkpoints and supports safe rollback.

The base model (veritas_model.pth) is NEVER modified. Fine-tuned models
are saved as veritas_model_v2.pth, veritas_model_v3.pth, etc.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config import MODEL_DIR, MODEL_REGISTRY_PATH, MODEL_WEIGHTS_PATH

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages versioned model checkpoints with a JSON registry."""

    def __init__(self):
        self.registry_path = MODEL_REGISTRY_PATH
        self._ensure_registry()

    def _ensure_registry(self):
        """Create the registry file with the base model if it doesn't exist."""
        if self.registry_path.exists():
            return

        base_entry = {
            "version": 1,
            "filename": "veritas_model.pth",
            "path": str(MODEL_WEIGHTS_PATH),
            "created_at": datetime.now().isoformat(),
            "metrics": {"note": "Original Kaggle-trained model"},
            "active": True,
        }
        registry = {"models": [base_entry], "active_version": 1}
        self._save_registry(registry)
        logger.info("Initialised model registry with base model v1")

    def _load_registry(self) -> dict:
        with open(self.registry_path, "r") as f:
            return json.load(f)

    def _save_registry(self, registry: dict):
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def get_active_model_path(self) -> Path:
        """Return the file path of the currently active model."""
        registry = self._load_registry()
        active_v = registry.get("active_version", 1)

        for entry in registry["models"]:
            if entry["version"] == active_v:
                p = Path(entry["path"])
                if p.exists():
                    return p
                # Fallback: try relative to MODEL_DIR
                p2 = MODEL_DIR / entry["filename"]
                if p2.exists():
                    return p2

        # Ultimate fallback: base model
        logger.warning("Active model not found, falling back to base model")
        return MODEL_WEIGHTS_PATH

    def register_model(
        self,
        filename: str,
        metrics: dict | None = None,
        set_active: bool = True,
    ) -> int:
        """Register a new fine-tuned model version.

        Args:
            filename: Name of the .pth file inside MODEL_DIR.
            metrics: Validation metrics dict.
            set_active: Whether to make this the active model.

        Returns:
            The new version number.
        """
        registry = self._load_registry()
        new_version = max(m["version"] for m in registry["models"]) + 1

        entry = {
            "version": new_version,
            "filename": filename,
            "path": str(MODEL_DIR / filename),
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "active": set_active,
        }
        registry["models"].append(entry)

        if set_active:
            # Deactivate all others
            for m in registry["models"]:
                m["active"] = m["version"] == new_version
            registry["active_version"] = new_version

        self._save_registry(registry)
        logger.info("Registered model v%d: %s (active=%s)", new_version, filename, set_active)
        return new_version

    def set_active(self, version: int) -> bool:
        """Set a specific version as the active model."""
        registry = self._load_registry()
        found = False
        for m in registry["models"]:
            if m["version"] == version:
                found = True
            m["active"] = m["version"] == version

        if not found:
            logger.error("Version %d not found in registry", version)
            return False

        registry["active_version"] = version
        self._save_registry(registry)
        logger.info("Set active model to v%d", version)
        return True

    def rollback(self) -> int:
        """Roll back to the previous version."""
        registry = self._load_registry()
        current = registry.get("active_version", 1)

        if current <= 1:
            logger.warning("Already at base model v1, cannot rollback further")
            return 1

        prev = current - 1
        self.set_active(prev)
        logger.info("Rolled back from v%d to v%d", current, prev)
        return prev

    def list_versions(self) -> list[dict]:
        """Return all registered model versions."""
        registry = self._load_registry()
        return registry["models"]

    def get_active_version(self) -> int:
        """Return the currently active version number."""
        registry = self._load_registry()
        return registry.get("active_version", 1)
