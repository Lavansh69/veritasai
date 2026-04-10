'use client';
import { motion } from 'framer-motion';

interface HeatmapViewerProps {
  heatmapUrl: string | null;
  explanation: string;
  indicators: string[];
}

export default function HeatmapViewer({ heatmapUrl, explanation, indicators }: HeatmapViewerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="glass-card p-6"
    >
      <h3 className="text-lg font-display font-bold mb-4 gradient-text">
        🔍 Why AI Thinks This Media May Be Fake
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Heatmap Image */}
        <div>
          {heatmapUrl ? (
            <div className="rounded-xl overflow-hidden border border-white/10">
              <img
                src={`http://localhost:8000${heatmapUrl}`}
                alt="Grad-CAM Heatmap"
                className="w-full h-auto"
              />
              <div className="p-3 text-xs text-center" style={{ color: 'var(--text-secondary)' }}>
                Grad-CAM heatmap — red/yellow regions indicate suspicious areas
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-white/5 p-12 text-center" style={{ color: 'var(--text-secondary)' }}>
              <p>No heatmap available</p>
            </div>
          )}
        </div>

        {/* Explanation */}
        <div className="space-y-4">
          <div className="space-y-2">
            {explanation.split('\n').map((line, i) => (
              <p key={i} className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {line}
              </p>
            ))}
          </div>

          {indicators.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                Detected Indicators
              </h4>
              <ul className="space-y-2">
                {indicators.map((ind, i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + i * 0.1 }}
                    className="flex items-start gap-2 text-sm"
                  >
                    <span className="mt-1 w-1.5 h-1.5 rounded-full bg-brand-400 shrink-0" />
                    <span style={{ color: 'var(--text-secondary)' }}>{ind}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
