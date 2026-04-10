'use client';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';
import { motion } from 'framer-motion';

interface RadarScoresProps {
  scores: {
    deepfake_probability: number;
    face_consistency: number;
    metadata_integrity: number;
    artifact_detection: number;
  };
}

export default function RadarScores({ scores }: RadarScoresProps) {
  const data = [
    { factor: 'Deepfake Prob.', value: scores.deepfake_probability },
    { factor: 'Face Match', value: scores.face_consistency },
    { factor: 'Metadata', value: scores.metadata_integrity },
    { factor: 'Artifacts', value: scores.artifact_detection },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="glass-card p-6"
    >
      <h3 className="text-sm font-semibold mb-4 text-center" style={{ color: 'var(--text-secondary)' }}>
        Multi-Factor Analysis Radar
      </h3>
      <ResponsiveContainer width="100%" height={280}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid
            stroke="rgba(255,255,255,0.08)"
            strokeDasharray="3 3"
          />
          <PolarAngleAxis
            dataKey="factor"
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: 'var(--text-secondary)', fontSize: 9 }}
          />
          <Radar
            name="Scores"
            dataKey="value"
            stroke="#818cf8"
            fill="#818cf8"
            fillOpacity={0.25}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
