'use client';
import { motion } from 'framer-motion';

interface ScoreCardProps {
  overallScore: number;
  verdict: string;
  verdictColor: string;
}

const colorMap: Record<string, { bg: string; text: string; glow: string }> = {
  green:  { bg: 'rgba(52, 211, 153, 0.12)', text: '#34d399', glow: 'rgba(52, 211, 153, 0.3)' },
  orange: { bg: 'rgba(251, 191, 36, 0.12)', text: '#fbbf24', glow: 'rgba(251, 191, 36, 0.3)' },
  red:    { bg: 'rgba(248, 113, 113, 0.12)', text: '#f87171', glow: 'rgba(248, 113, 113, 0.3)' },
};

export default function ScoreCard({ overallScore, verdict, verdictColor }: ScoreCardProps) {
  const colors = colorMap[verdictColor] || colorMap.orange;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="glass-card p-8 text-center"
      style={{
        borderColor: colors.text,
        boxShadow: `0 0 40px ${colors.glow}`,
      }}
    >
      <p className="text-sm font-medium uppercase tracking-wider mb-4" style={{ color: 'var(--text-secondary)' }}>
        Overall Authenticity Score
      </p>
      <motion.div
        className="text-7xl font-display font-extrabold mb-2"
        style={{ color: colors.text }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        {overallScore}
        <span className="text-3xl opacity-60">/100</span>
      </motion.div>
      <motion.div
        className="inline-block px-4 py-2 rounded-full text-sm font-bold uppercase tracking-wider mt-3"
        style={{ backgroundColor: colors.bg, color: colors.text }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        {verdict}
      </motion.div>
    </motion.div>
  );
}
