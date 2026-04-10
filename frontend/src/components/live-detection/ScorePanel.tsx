'use client';
import { motion } from 'framer-motion';
import { LineChart, Line, ResponsiveContainer, YAxis, ReferenceLine } from 'recharts';
import {
  Activity,
  Gauge,
  Thermometer,
  Eye,
  EyeOff,
  SlidersHorizontal,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react';

interface ScorePanelProps {
  confidence: number;
  rawConfidence: number;
  label: string;
  latencyMs: number;
  history: { t: number; v: number }[];
  facesDetected: number;
  heatmapEnabled: boolean;
  frameSkip: number;
  connectionStatus: 'idle' | 'connecting' | 'active' | 'error';
  onToggleHeatmap: () => void;
  onFrameSkipChange: (n: number) => void;
}

export default function ScorePanel({
  confidence,
  rawConfidence,
  label,
  latencyMs,
  history,
  facesDetected,
  heatmapEnabled,
  frameSkip,
  connectionStatus,
  onToggleHeatmap,
  onFrameSkipChange,
}: ScorePanelProps) {
  const pct = Math.round(confidence * 100);
  const rawPct = Math.round(rawConfidence * 100);

  const labelColor =
    label === 'Likely Deepfake'
      ? '#f87171'
      : label === 'Suspicious'
      ? '#fbbf24'
      : '#34d399';

  const statusColor =
    connectionStatus === 'active'
      ? '#34d399'
      : connectionStatus === 'connecting'
      ? '#fbbf24'
      : connectionStatus === 'error'
      ? '#f87171'
      : '#94a3b8';

  const statusLabel =
    connectionStatus === 'active'
      ? 'Live'
      : connectionStatus === 'connecting'
      ? 'Connecting…'
      : connectionStatus === 'error'
      ? 'Disconnected'
      : 'Idle';

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* ── Status bar ────────────────────────────────────────── */}
      <div
        className="glass-card px-4 py-3 flex items-center justify-between text-xs font-medium"
      >
        <div className="flex items-center gap-2">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{
              backgroundColor: statusColor,
              boxShadow: `0 0 8px ${statusColor}`,
              animation: connectionStatus === 'active' ? 'pulse 2s infinite' : undefined,
            }}
          />
          <span style={{ color: statusColor }}>{statusLabel}</span>
        </div>
        <div className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          {connectionStatus === 'active' ? (
            <Wifi className="w-3.5 h-3.5" />
          ) : (
            <WifiOff className="w-3.5 h-3.5" />
          )}
          <span>{facesDetected} face{facesDetected !== 1 ? 's' : ''}</span>
        </div>
      </div>

      {/* ── Confidence gauge ────────────────────────────────── */}
      <div className="glass-card p-5 flex flex-col items-center gap-3">
        <div className="relative w-32 h-16">
          {/* Semi-circle gauge SVG */}
          <svg width="128" height="72" viewBox="0 0 128 72">
            <path
              d="M 8 64 A 56 56 0 0 1 120 64"
              fill="none"
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="10"
              strokeLinecap="round"
            />
            <motion.path
              d="M 8 64 A 56 56 0 0 1 120 64"
              fill="none"
              stroke={labelColor}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={`${Math.PI * 56}`}
              animate={{
                strokeDashoffset: Math.PI * 56 - (pct / 100) * Math.PI * 56,
              }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              style={{ filter: `drop-shadow(0 0 6px ${labelColor}40)` }}
            />
          </svg>
          <div className="absolute inset-0 flex items-end justify-center pb-0">
            <span className="text-2xl font-bold font-display" style={{ color: labelColor }}>
              {pct}%
            </span>
          </div>
        </div>

        {/* Label badge */}
        <motion.div
          animate={{ backgroundColor: `${labelColor}18`, borderColor: `${labelColor}50` }}
          className="px-3 py-1 rounded-full text-xs font-semibold border"
          style={{ color: labelColor }}
        >
          {label || '—'}
        </motion.div>

        <div className="flex items-center gap-4 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <span className="flex items-center gap-1">
            <Gauge className="w-3 h-3" /> Raw: {rawPct}%
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-3 h-3" /> {latencyMs.toFixed(0)}ms
          </span>
        </div>
      </div>

      {/* ── Score history sparkline ────────────────────────── */}
      <div className="glass-card p-4">
        <h4 className="text-xs font-semibold mb-3 flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <Activity className="w-3.5 h-3.5" />
          Confidence History
        </h4>
        <div style={{ width: '100%', height: 100 }}>
          <ResponsiveContainer>
            <LineChart data={history}>
              <YAxis domain={[0, 1]} hide />
              <ReferenceLine y={0.8} stroke="#f8717140" strokeDasharray="4 4" />
              <ReferenceLine y={0.65} stroke="#fbbf2440" strokeDasharray="4 4" />
              <Line
                type="monotone"
                dataKey="v"
                stroke="#818cf8"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-between text-[10px] mt-1" style={{ color: 'var(--text-secondary)' }}>
          <span>Older</span>
          <span>Now</span>
        </div>
      </div>

      {/* ── Controls ──────────────────────────────────────── */}
      <div className="glass-card p-4 space-y-4">
        <h4 className="text-xs font-semibold flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
          <SlidersHorizontal className="w-3.5 h-3.5" />
          Controls
        </h4>

        {/* Heatmap toggle */}
        <div className="flex items-center justify-between">
          <span className="text-xs flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            <Thermometer className="w-3.5 h-3.5" />
            Grad-CAM Heatmap
          </span>
          <button
            onClick={onToggleHeatmap}
            className="relative w-10 h-5 rounded-full transition-colors duration-200"
            style={{
              backgroundColor: heatmapEnabled ? 'var(--accent)' : 'rgba(255,255,255,0.1)',
            }}
          >
            <motion.div
              className="absolute top-0.5 w-4 h-4 rounded-full bg-white"
              animate={{ left: heatmapEnabled ? 22 : 2 }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
            />
          </button>
        </div>

        {/* Frame skip slider */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
              {heatmapEnabled ? (
                <Eye className="w-3.5 h-3.5" />
              ) : (
                <EyeOff className="w-3.5 h-3.5" />
              )}
              Frame Skip
            </span>
            <span className="text-xs font-semibold" style={{ color: 'var(--accent)' }}>
              1/{frameSkip}
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            value={frameSkip}
            onChange={(e) => onFrameSkipChange(Number(e.target.value))}
            className="w-full h-1 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, var(--accent) ${((frameSkip - 1) / 9) * 100}%, rgba(255,255,255,0.1) ${((frameSkip - 1) / 9) * 100}%)`,
            }}
          />
        </div>
      </div>
    </div>
  );
}
