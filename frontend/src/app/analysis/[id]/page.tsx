'use client';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import GaugeChart from '@/components/GaugeChart';
import ScoreCard from '@/components/ScoreCard';
import HeatmapViewer from '@/components/HeatmapViewer';
import RadarScores from '@/components/RadarScores';
import FeedbackPanel from '@/components/FeedbackPanel';
import { FileText, AlertTriangle, CheckCircle, Info } from 'lucide-react';

interface AnalysisResult {
  analysis_id: string;
  file_hash: string;
  deepfake: { probability: number; label: string; per_frame_scores: number[] };
  heatmap: { heatmap_url: string | null; explanation: string; indicators: string[] };
  face_consistency: { score: number; detail: string; match: boolean | null };
  metadata: { warnings: string[]; integrity_score: number; summary: string; metadata: Record<string, any> };
  scorecard: {
    scores: { deepfake_probability: number; face_consistency: number; metadata_integrity: number; artifact_detection: number };
    overall_score: number;
    verdict: string;
    verdict_color: string;
  };
}

export default function AnalysisPage({ params }: { params: { id: string } }) {
  const [result, setResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem('veritasai-result');
    if (stored) {
      try {
        setResult(JSON.parse(stored));
      } catch {}
    }
  }, []);

  if (!result) {
    return (
      <div className="min-h-[60vh] flex items-center justify-center px-4">
        <div className="glass-card p-12 text-center max-w-md">
          <Info className="w-12 h-12 mx-auto mb-4 text-brand-400" />
          <h2 className="text-xl font-display font-bold mb-2">No Analysis Data</h2>
          <p className="text-sm mb-6" style={{ color: 'var(--text-secondary)' }}>
            Upload media first to see analysis results.
          </p>
          <Link href="/upload" className="btn-primary inline-block">
            Go to Upload
          </Link>
        </div>
      </div>
    );
  }

  const sc = result.scorecard;
  const scores = sc.scores;

  return (
    <div className="max-w-6xl mx-auto px-4 py-12 space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl md:text-4xl font-display font-bold mb-2">
          Analysis <span className="gradient-text">Results</span>
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          ID: {result.analysis_id}
        </p>
      </motion.div>

      {/* Score Card */}
      <ScoreCard
        overallScore={sc.overall_score}
        verdict={sc.verdict}
        verdictColor={sc.verdict_color}
      />

      {/* Gauge charts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4"
      >
        <div className="glass-card p-4 flex justify-center">
          <GaugeChart value={scores.deepfake_probability} label="Deepfake Prob." color="#f87171" />
        </div>
        <div className="glass-card p-4 flex justify-center">
          <GaugeChart value={scores.face_consistency} label="Face Match" color="#818cf8" />
        </div>
        <div className="glass-card p-4 flex justify-center">
          <GaugeChart value={scores.metadata_integrity} label="Metadata" color="#fbbf24" />
        </div>
        <div className="glass-card p-4 flex justify-center">
          <GaugeChart value={scores.artifact_detection} label="Artifacts" color="#34d399" />
        </div>
      </motion.div>

      {/* Radar + Heatmap */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RadarScores scores={scores} />
        <HeatmapViewer
          heatmapUrl={result.heatmap.heatmap_url}
          explanation={result.heatmap.explanation}
          indicators={result.heatmap.indicators}
        />
      </div>

      {/* Metadata warnings */}
      {result.metadata.warnings.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-display font-bold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
            Metadata Warnings
          </h3>
          <ul className="space-y-2">
            {result.metadata.warnings.map((w, i) => (
              <li key={i} className="flex items-start gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
                <span className="mt-1 w-1.5 h-1.5 rounded-full bg-yellow-400 shrink-0" />
                {w}
              </li>
            ))}
          </ul>
        </motion.div>
      )}

      {/* Face consistency */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-card p-6"
      >
        <h3 className="text-lg font-display font-bold mb-3 flex items-center gap-2">
          {result.face_consistency.match === true ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : result.face_consistency.match === false ? (
            <AlertTriangle className="w-5 h-5 text-red-400" />
          ) : (
            <Info className="w-5 h-5 text-brand-400" />
          )}
          Face Consistency Analysis
        </h3>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          {result.face_consistency.detail}
        </p>
      </motion.div>

      {/* Feedback Panel */}
      <FeedbackPanel
        analysisId={result.analysis_id}
        prediction={result.deepfake.label}
        confidence={result.deepfake.probability}
      />

      {/* Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="flex flex-col sm:flex-row gap-4 justify-center"
      >
        <Link href={`/report/${result.analysis_id}`}>
          <button className="btn-primary flex items-center gap-2 w-full sm:w-auto justify-center">
            <FileText className="w-5 h-5" />
            View Evidence Report
          </button>
        </Link>
        <a
          href={`http://localhost:8000/api/report/${result.analysis_id}`}
          download
          className="btn-outline flex items-center gap-2 justify-center"
        >
          Download PDF Report
        </a>
      </motion.div>
    </div>
  );
}
