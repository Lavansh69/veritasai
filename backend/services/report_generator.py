"""
VeritasAI – PDF Evidence Report Generator
Generates a structured forensic evidence report using ReportLab.
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import HEATMAP_DIR, REPORT_DIR


def generate_pdf_report(result: dict) -> str:
    """Generate a formatted PDF evidence report.
    
    Args:
        result: Full analysis result dict from the /analyze endpoint.
    
    Returns:
        Path to generated PDF file.
    """
    analysis_id = result["analysis_id"]
    pdf_path = REPORT_DIR / f"{analysis_id}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=16,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        leading=14,
    )
    verdict_style = ParagraphStyle(
        "Verdict",
        parent=styles["Heading1"],
        fontSize=18,
        alignment=1,  # center
        spaceBefore=12,
        spaceAfter=12,
    )

    elements: list = []

    # ── Header ─────────────────────────────────────────────────────
    elements.append(Paragraph("VeritasAI", title_style))
    elements.append(
        Paragraph("Digital Forensic Evidence Report", styles["Heading3"])
    )
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8 * mm))

    # ── Analysis Info ──────────────────────────────────────────────
    elements.append(Paragraph("Analysis Information", heading_style))
    info_data = [
        ["Analysis ID", analysis_id],
        ["Timestamp", datetime.now().isoformat()],
        ["File Hash (SHA-256)", result.get("file_hash", "N/A")],
    ]
    info_table = Table(info_data, colWidths=[5 * cm, 12 * cm])
    info_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    elements.append(info_table)
    elements.append(Spacer(1, 6 * mm))

    # ── Verdict ────────────────────────────────────────────────────
    scorecard = result.get("scorecard", {})
    verdict = scorecard.get("verdict", "Unknown")
    overall = scorecard.get("overall_score", "N/A")
    verdict_color_map = {
        "green": colors.HexColor("#2e7d32"),
        "orange": colors.HexColor("#f57c00"),
        "red": colors.HexColor("#c62828"),
    }
    vc = verdict_color_map.get(scorecard.get("verdict_color", ""), colors.black)
    verdict_style.textColor = vc
    elements.append(
        Paragraph(f"Verdict: {verdict} ({overall}/100)", verdict_style)
    )
    elements.append(Spacer(1, 4 * mm))

    # ── Scorecard Table ────────────────────────────────────────────
    elements.append(Paragraph("Authenticity Scorecard", heading_style))
    scores = scorecard.get("scores", {})
    score_data = [
        ["Factor", "Score"],
        ["Deepfake Probability", f"{scores.get('deepfake_probability', 'N/A')}%"],
        ["Face Consistency", f"{scores.get('face_consistency', 'N/A')}%"],
        ["Metadata Integrity", f"{scores.get('metadata_integrity', 'N/A')}%"],
        ["Artifact Detection", f"{scores.get('artifact_detection', 'N/A')}%"],
        ["Overall Score", f"{overall}/100"],
    ]
    score_table = Table(score_data, colWidths=[8 * cm, 5 * cm])
    score_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ])
    )
    elements.append(score_table)
    elements.append(Spacer(1, 6 * mm))

    # ── AI Explanation ─────────────────────────────────────────────
    elements.append(Paragraph("AI Analysis Explanation", heading_style))
    heatmap = result.get("heatmap", {})
    explanation = heatmap.get("explanation", "No explanation available.")
    for line in explanation.split("\n"):
        elements.append(Paragraph(line.strip(), body_style))
    elements.append(Spacer(1, 4 * mm))

    # ── Heatmap Image ──────────────────────────────────────────────
    heatmap_url = heatmap.get("heatmap_url", "")
    if heatmap_url:
        heatmap_file = HEATMAP_DIR / f"{analysis_id}_heatmap.jpg"
        if heatmap_file.exists():
            elements.append(Paragraph("Heatmap Visualization", heading_style))
            img = Image(str(heatmap_file), width=10 * cm, height=10 * cm)
            elements.append(img)
            elements.append(Spacer(1, 4 * mm))

    # ── Metadata Findings ──────────────────────────────────────────
    elements.append(Paragraph("Metadata Forensic Findings", heading_style))
    meta = result.get("metadata", {})
    warnings = meta.get("warnings", [])
    if warnings:
        for w in warnings:
            elements.append(
                Paragraph(f"⚠ {w}", body_style)
            )
    else:
        elements.append(Paragraph("No metadata anomalies detected.", body_style))
    elements.append(Spacer(1, 6 * mm))

    # ── Disclaimer ─────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 3 * mm))
    disclaimer = ParagraphStyle("Disclaimer", parent=body_style, fontSize=7, textColor=colors.grey)
    elements.append(
        Paragraph(
            "This report was generated by VeritasAI, an automated AI forensic analysis tool. "
            "Results are probabilistic assessments and should be reviewed by qualified experts "
            "before being used as evidence in legal proceedings.",
            disclaimer,
        )
    )

    doc.build(elements)
    return str(pdf_path)
