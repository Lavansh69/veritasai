'use client';
import { useRef, useCallback, useState, useEffect } from 'react';
import WebcamView, { WebcamViewHandle } from './WebcamView';
import ScorePanel from './ScorePanel';
import { Play, Square, RotateCcw } from 'lucide-react';
import { motion } from 'framer-motion';

const WS_URL = 'ws://localhost:8000/api/live-detection';
const CAPTURE_INTERVAL_MS = 100; // ~10 FPS
const MAX_HISTORY = 60;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

interface FaceResult {
  bbox: { top: number; right: number; bottom: number; left: number };
  confidence: number;
  label: string;
}

interface DetectionResult {
  label: string;
  confidence: number;
  raw_confidence: number;
  bbox: { top: number; right: number; bottom: number; left: number } | null;
  faces: FaceResult[];
  heatmap: string | null;
  latency_ms: number;
  faces_detected: number;
  frame_index: number;
}

export default function LiveDetectionPanel() {
  const webcamRef = useRef<WebcamViewHandle>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const intentionalStopRef = useRef(false);

  // State
  const [status, setStatus] = useState<'idle' | 'connecting' | 'active' | 'error'>('idle');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [history, setHistory] = useState<{ t: number; v: number }[]>([]);
  const [heatmapEnabled, setHeatmapEnabled] = useState(false);
  const [frameSkip, setFrameSkip] = useState(3);

  // ── WebSocket lifecycle ──────────────────────────────────────────
  const cleanupWs = useCallback(() => {
    if (captureTimerRef.current) {
      clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connectWs = useCallback(() => {
    intentionalStopRef.current = false;
    cleanupWs();
    setStatus('connecting');

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      setStatus('active');
      reconnectAttemptRef.current = 0;

      // Send initial config
      ws.send(JSON.stringify({ type: 'config', heatmap: heatmapEnabled, frame_skip: frameSkip }));

      // Start capture loop
      captureTimerRef.current = setInterval(async () => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const blob = await webcamRef.current?.captureFrame();
        if (!blob) return;
        const buffer = await blob.arrayBuffer();
        ws.send(buffer);
      }, CAPTURE_INTERVAL_MS);
    };

    ws.onmessage = (event) => {
      try {
        const data: DetectionResult = JSON.parse(event.data);
        setResult(data);
        setHistory((prev) => {
          const next = [...prev, { t: Date.now(), v: data.confidence }];
          return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
        });
      } catch {
        /* ignore malformed */
      }
    };

    ws.onclose = () => {
      if (captureTimerRef.current) {
        clearInterval(captureTimerRef.current);
        captureTimerRef.current = null;
      }
      // Attempt reconnect only if the close was NOT from a manual stop
      if (!intentionalStopRef.current) {
        setStatus('error');
        const delay = Math.min(
          RECONNECT_BASE_MS * 2 ** reconnectAttemptRef.current,
          RECONNECT_MAX_MS
        );
        reconnectAttemptRef.current += 1;
        reconnectTimerRef.current = setTimeout(connectWs, delay);
      }
    };

    ws.onerror = () => {
      // onclose will fire after this — reconnect logic lives there
    };
  }, [cleanupWs, heatmapEnabled, frameSkip]);

  // ── Push config changes to server ────────────────────────────────
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: 'config', heatmap: heatmapEnabled, frame_skip: frameSkip })
      );
    }
  }, [heatmapEnabled, frameSkip]);

  // ── Start detection ──────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    const ok = await webcamRef.current?.startCamera();
    if (!ok) return;
    // Small delay to ensure camera is ready
    setTimeout(() => connectWs(), 300);
  }, [connectWs]);

  // ── Stop detection ───────────────────────────────────────────────
  const handleStop = useCallback(() => {
    intentionalStopRef.current = true;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    cleanupWs();
    webcamRef.current?.stopCamera();
    setStatus('idle');
    setResult(null);
    setHistory([]);
  }, [cleanupWs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      handleStop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isRunning = status !== 'idle';

  return (
    <div className="flex flex-col lg:flex-row gap-6 w-full">
      {/* ── Left: Webcam ──────────────────────────────────────── */}
      <div className="flex-1 flex flex-col gap-4 min-w-0">
        <WebcamView
          ref={webcamRef}
          faces={result?.faces ?? []}
          heatmapSrc={result?.heatmap ?? null}
          showHeatmap={heatmapEnabled}
          facesDetected={result?.faces_detected ?? 0}
          isActive={isRunning}
          label={result?.label ?? ''}
        />

        {/* Start / Stop bar */}
        <div className="flex items-center gap-3">
          {status === 'idle' ? (
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={handleStart}
              className="btn-primary flex-1 flex items-center justify-center gap-2 text-sm"
            >
              <Play className="w-4 h-4" />
              Start Live Detection
            </motion.button>
          ) : (
            <>
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={handleStop}
                className="flex-1 flex items-center justify-center gap-2 text-sm px-6 py-3 rounded-xl font-semibold text-white transition-all duration-300"
                style={{
                  background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                  boxShadow: '0 4px 15px rgba(239,68,68,0.4)',
                }}
              >
                <Square className="w-4 h-4" />
                Stop
              </motion.button>
              {status === 'error' && (
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={connectWs}
                  className="btn-outline flex items-center gap-2 text-sm !px-4"
                >
                  <RotateCcw className="w-4 h-4" />
                  Reconnect
                </motion.button>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Right: Score panel ─────────────────────────────────── */}
      <div className="w-full lg:w-80 shrink-0">
        <ScorePanel
          confidence={result?.confidence ?? 0}
          rawConfidence={result?.raw_confidence ?? 0}
          label={result?.label ?? '—'}
          latencyMs={result?.latency_ms ?? 0}
          history={history}
          facesDetected={result?.faces_detected ?? 0}
          heatmapEnabled={heatmapEnabled}
          frameSkip={frameSkip}
          connectionStatus={status}
          onToggleHeatmap={() => setHeatmapEnabled((p) => !p)}
          onFrameSkipChange={setFrameSkip}
        />
      </div>
    </div>
  );
}
