'use client';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { FileText, Download, ExternalLink, ArrowLeft } from 'lucide-react';

export default function ReportPage({ params }: { params: { id: string } }) {
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem('veritasai-result');
    if (stored) {
      try { setResult(JSON.parse(stored)); } catch {}
    }
  }, []);

  const sc = result?.scorecard;

  return (
    <div className="max-w-4xl mx-auto px-4 py-12 space-y-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Link href={`/analysis/${params.id}`} className="inline-flex items-center gap-1 text-sm mb-6 hover:text-brand-400 transition-colors" style={{ color: 'var(--text-secondary)' }}>
          <ArrowLeft className="w-4 h-4" /> Back to Analysis
        </Link>

        <h1 className="text-3xl font-display font-bold mb-2">
          <span className="gradient-text">Evidence Report</span>
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Digital Forensic Evidence Report — {params.id}
        </p>
      </motion.div>

      {/* Report preview card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-8 space-y-6"
      >
        <div className="flex items-center gap-3 pb-4" style={{ borderBottom: '1px solid var(--glass-border)' }}>
          <FileText className="w-8 h-8 text-brand-400" />
          <div>
            <h2 className="text-xl font-display font-bold">VeritasAI Forensic Report</h2>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>Generated automatically by VeritasAI analysis pipeline</p>
          </div>
        </div>

        {result ? (
          <div className="space-y-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span className="text-xs uppercase tracking-wider font-semibold block mb-1" style={{ color: 'var(--text-primary)' }}>Analysis ID</span>
                <code className="text-xs bg-white/5 px-2 py-1 rounded">{result.analysis_id}</code>
              </div>
              <div>
                <span className="text-xs uppercase tracking-wider font-semibold block mb-1" style={{ color: 'var(--text-primary)' }}>File SHA-256</span>
                <code className="text-xs bg-white/5 px-2 py-1 rounded break-all">{result.file_hash}</code>
              </div>
            </div>

            {sc && (
              <>
                <div style={{ borderTop: '1px solid var(--glass-border)' }} className="pt-4">
                  <span className="text-xs uppercase tracking-wider font-semibold block mb-2" style={{ color: 'var(--text-primary)' }}>Verdict</span>
                  <span className={`inline-block px-3 py-1 rounded-full text-sm font-bold ${
                    sc.verdict_color === 'green' ? 'bg-green-500/20 text-green-400' :
                    sc.verdict_color === 'red' ? 'bg-red-500/20 text-red-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {sc.verdict} — {sc.overall_score}/100
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3" style={{ borderTop: '1px solid var(--glass-border)', paddingTop: '1rem' }}>
                  {Object.entries(sc.scores).map(([key, val]) => (
                    <div key={key} className="bg-white/5 rounded-lg p-3 text-center">
                      <div className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{String(val)}%</div>
                      <div className="text-xs capitalize">{key.replace(/_/g, ' ')}</div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {result.metadata?.warnings?.length > 0 && (
              <div style={{ borderTop: '1px solid var(--glass-border)' }} className="pt-4">
                <span className="text-xs uppercase tracking-wider font-semibold block mb-2" style={{ color: 'var(--text-primary)' }}>Metadata Warnings</span>
                <ul className="space-y-1">
                  {result.metadata.warnings.map((w: string, i: number) => (
                    <li key={i} className="flex gap-2"><span className="text-yellow-400">⚠</span> {w}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.heatmap?.explanation && (
              <div style={{ borderTop: '1px solid var(--glass-border)' }} className="pt-4">
                <span className="text-xs uppercase tracking-wider font-semibold block mb-2" style={{ color: 'var(--text-primary)' }}>AI Explanation</span>
                <p className="whitespace-pre-line">{result.heatmap.explanation}</p>
              </div>
            )}
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)' }}>No analysis data available. Please run an analysis first.</p>
        )}
      </motion.div>

      {/* Action buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex flex-col sm:flex-row gap-4"
      >
        <a
          href={`http://localhost:8000/api/report/${params.id}`}
          download
          className="btn-primary flex items-center gap-2 justify-center text-center"
        >
          <Download className="w-5 h-5" />
          Download PDF Report
        </a>

        <a
          href="https://cybercrime.gov.in"
          target="_blank"
          rel="noopener noreferrer"
          className="btn-outline flex items-center gap-2 justify-center text-center"
          style={{ borderColor: '#f87171', color: '#f87171' }}
        >
          <ExternalLink className="w-5 h-5" />
          Report to Cyber Crime Portal
        </a>
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="text-xs text-center"
        style={{ color: 'var(--text-secondary)' }}
      >
        This report is generated by an automated AI system. Results should be reviewed by qualified experts before legal use.
      </motion.p>
    </div>
  );
}
