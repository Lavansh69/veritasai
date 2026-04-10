'use client';
import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ScanningOverlayProps {
  isActive: boolean;
  file: File | null;
}

const SCAN_PHASES = [
  { label: 'Initializing AI Engine', icon: '⚡', detail: 'Loading EfficientNet-B4 neural network...' },
  { label: 'Scanning Facial Features', icon: '👁️', detail: 'Detecting face landmarks and geometry...' },
  { label: 'Analyzing Pixel Artifacts', icon: '🔬', detail: 'Checking for GAN fingerprints and compression anomalies...' },
  { label: 'Frequency Domain Analysis', icon: '📊', detail: 'Examining spectral patterns for manipulation traces...' },
  { label: 'Metadata Verification', icon: '🔐', detail: 'Inspecting EXIF data and file integrity...' },
  { label: 'Generating Heatmap', icon: '🗺️', detail: 'Computing Grad-CAM activation regions...' },
  { label: 'Computing Verdict', icon: '🛡️', detail: 'Aggregating scores and confidence intervals...' },
];

export default function ScanningOverlay({ isActive, file }: ScanningOverlayProps) {
  const [phase, setPhase] = useState(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [scanlinePos, setScanlinePos] = useState(0);
  const [gridPoints, setGridPoints] = useState<Array<{ x: number; y: number; delay: number }>>([]);
  const animRef = useRef<number>(0);

  // Generate preview URL
  useEffect(() => {
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setImageUrl(null);
    }
  }, [file]);

  // Generate randomized grid points for the scan effect
  useEffect(() => {
    if (isActive) {
      const points = Array.from({ length: 24 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        delay: Math.random() * 2,
      }));
      setGridPoints(points);
    }
  }, [isActive]);

  // Phase progression
  useEffect(() => {
    if (!isActive) {
      setPhase(0);
      return;
    }

    const interval = setInterval(() => {
      setPhase((prev) => {
        if (prev < SCAN_PHASES.length - 1) return prev + 1;
        return prev;
      });
    }, 2200);

    return () => clearInterval(interval);
  }, [isActive]);

  // Scanline animation
  useEffect(() => {
    if (!isActive) return;

    let pos = 0;
    let direction = 1;
    const animate = () => {
      pos += direction * 0.4;
      if (pos >= 100) { pos = 100; direction = -1; }
      if (pos <= 0) { pos = 0; direction = 1; }
      setScanlinePos(pos);
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, [isActive]);

  const progress = Math.min(((phase + 1) / SCAN_PHASES.length) * 100, 100);

  return (
    <AnimatePresence>
      {isActive && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(5, 2, 20, 0.92)', backdropFilter: 'blur(20px)' }}
        >
          {/* Ambient glow orbs */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div
              className="absolute w-96 h-96 rounded-full opacity-20"
              style={{
                background: 'radial-gradient(circle, rgba(99,102,241,0.4) 0%, transparent 70%)',
                top: '10%', left: '10%',
                animation: 'pulse-glow 4s ease-in-out infinite',
              }}
            />
            <div
              className="absolute w-80 h-80 rounded-full opacity-15"
              style={{
                background: 'radial-gradient(circle, rgba(168,85,247,0.4) 0%, transparent 70%)',
                bottom: '15%', right: '15%',
                animation: 'pulse-glow 5s ease-in-out infinite 1s',
              }}
            />
          </div>

          <div className="relative flex flex-col items-center gap-8 w-full max-w-lg px-6">
            {/* Image preview with scanning effect */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              className="relative w-72 h-72 md:w-80 md:h-80 rounded-2xl overflow-hidden"
              style={{
                boxShadow: '0 0 60px rgba(99,102,241,0.3), 0 0 120px rgba(99,102,241,0.1)',
              }}
            >
              {/* Image or placeholder */}
              {imageUrl ? (
                <img src={imageUrl} alt="Scanning" className="w-full h-full object-cover" />
              ) : (
                <div
                  className="w-full h-full flex items-center justify-center"
                  style={{ background: 'linear-gradient(135deg, #1a1a2e, #16213e)' }}
                >
                  <svg className="w-20 h-20 opacity-20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <path d="M21 15l-5-5L5 21" />
                  </svg>
                </div>
              )}

              {/* Scan overlay effects */}
              <div className="absolute inset-0">
                {/* Grid overlay */}
                <div
                  className="absolute inset-0 opacity-20"
                  style={{
                    backgroundImage: `
                      linear-gradient(rgba(99,102,241,0.3) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(99,102,241,0.3) 1px, transparent 1px)
                    `,
                    backgroundSize: '20px 20px',
                    animation: 'grid-shift 3s linear infinite',
                  }}
                />

                {/* Moving scanline */}
                <div
                  className="absolute left-0 right-0 h-1 pointer-events-none"
                  style={{
                    top: `${scanlinePos}%`,
                    background: 'linear-gradient(90deg, transparent, rgba(99,102,241,0.8), rgba(168,85,247,0.8), transparent)',
                    boxShadow: '0 0 20px rgba(99,102,241,0.6), 0 0 40px rgba(99,102,241,0.3)',
                  }}
                />
                {/* Scanline trail */}
                <div
                  className="absolute left-0 right-0 pointer-events-none"
                  style={{
                    top: `${scanlinePos}%`,
                    height: '40px',
                    marginTop: '-40px',
                    background: 'linear-gradient(to top, rgba(99,102,241,0.15), transparent)',
                  }}
                />

                {/* Animated detection points */}
                {gridPoints.map((pt, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{
                      opacity: [0, 1, 1, 0],
                      scale: [0, 1, 1.2, 0],
                    }}
                    transition={{
                      duration: 2,
                      delay: pt.delay + phase * 0.3,
                      repeat: Infinity,
                      repeatDelay: 3,
                    }}
                    className="absolute"
                    style={{
                      left: `${pt.x}%`,
                      top: `${pt.y}%`,
                    }}
                  >
                    {/* Crosshair */}
                    <div className="relative w-4 h-4 -ml-2 -mt-2">
                      <div className="absolute inset-0 border border-cyan-400/80 rounded-sm" />
                      <div className="absolute top-1/2 left-0 right-0 h-px bg-cyan-400/60" />
                      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-cyan-400/60" />
                    </div>
                  </motion.div>
                ))}

                {/* Corner brackets */}
                <div className="absolute top-3 left-3 w-6 h-6 border-t-2 border-l-2 border-cyan-400/70 rounded-tl-sm" />
                <div className="absolute top-3 right-3 w-6 h-6 border-t-2 border-r-2 border-cyan-400/70 rounded-tr-sm" />
                <div className="absolute bottom-3 left-3 w-6 h-6 border-b-2 border-l-2 border-cyan-400/70 rounded-bl-sm" />
                <div className="absolute bottom-3 right-3 w-6 h-6 border-b-2 border-r-2 border-cyan-400/70 rounded-br-sm" />

                {/* Pulsing border */}
                <div
                  className="absolute inset-0 rounded-2xl"
                  style={{
                    border: '2px solid rgba(99,102,241,0.4)',
                    animation: 'border-pulse 2s ease-in-out infinite',
                  }}
                />
              </div>

              {/* Top-left HUD label */}
              <div className="absolute top-4 left-8 text-[10px] font-mono tracking-wider text-cyan-400/80">
                VERITAS AI v7.0
              </div>
              {/* Top-right status */}
              <div className="absolute top-4 right-8 flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                <span className="text-[10px] font-mono tracking-wider text-green-400/80">SCANNING</span>
              </div>
            </motion.div>

            {/* Phase label */}
            <motion.div
              className="text-center"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <AnimatePresence mode="wait">
                <motion.div
                  key={phase}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="flex flex-col items-center gap-2"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{SCAN_PHASES[phase].icon}</span>
                    <h2 className="text-xl font-display font-bold text-white">
                      {SCAN_PHASES[phase].label}
                    </h2>
                  </div>
                  <p className="text-xs font-mono text-slate-400">
                    {SCAN_PHASES[phase].detail}
                  </p>
                </motion.div>
              </AnimatePresence>
            </motion.div>

            {/* Progress bar */}
            <div className="w-full max-w-xs">
              <div
                className="h-1.5 rounded-full overflow-hidden"
                style={{ background: 'rgba(255,255,255,0.08)' }}
              >
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    background: 'linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7)',
                    boxShadow: '0 0 12px rgba(99,102,241,0.5)',
                  }}
                  initial={{ width: '0%' }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.8, ease: 'easeInOut' }}
                />
              </div>
              <div className="flex justify-between mt-2">
                <span className="text-[11px] font-mono text-slate-500">
                  Phase {phase + 1}/{SCAN_PHASES.length}
                </span>
                <span className="text-[11px] font-mono text-indigo-400">
                  {Math.round(progress)}%
                </span>
              </div>
            </div>

            {/* Telemetry dots */}
            <div className="flex items-center gap-3 mt-2">
              {SCAN_PHASES.map((_, i) => (
                <motion.div
                  key={i}
                  className="w-2 h-2 rounded-full"
                  animate={{
                    backgroundColor: i <= phase ? '#818cf8' : 'rgba(255,255,255,0.15)',
                    scale: i === phase ? [1, 1.4, 1] : 1,
                  }}
                  transition={{
                    duration: i === phase ? 1 : 0.3,
                    repeat: i === phase ? Infinity : 0,
                  }}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
