'use client';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, CheckCircle, Send } from 'lucide-react';

interface FeedbackPanelProps {
  analysisId: string;
  prediction: string;
  confidence: number;
}

export default function FeedbackPanel({ analysisId, prediction, confidence }: FeedbackPanelProps) {
  const [submitted, setSubmitted] = useState(false);
  const [showCorrection, setShowCorrection] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitFeedback = async (isCorrect: boolean, correctedLabel?: string) => {
    setLoading(true);
    setError(null);

    try {
      const body: Record<string, any> = {
        analysis_id: analysisId,
        is_correct: isCorrect,
        prediction: prediction,
        confidence: confidence,
      };
      if (correctedLabel) {
        body.corrected_label = correctedLabel;
      }

      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        throw new Error('Failed to submit feedback');
      }

      setSubmitted(true);
    } catch (err: any) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass-card p-6 text-center"
      >
        <CheckCircle className="w-10 h-10 mx-auto mb-3 text-green-400" />
        <h3 className="text-lg font-display font-bold mb-1">Thank You!</h3>
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          Your feedback helps improve our AI model over time.
        </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.55 }}
      className="glass-card p-6"
    >
      <h3 className="text-lg font-display font-bold mb-3 text-center">
        🔍 Was this prediction correct?
      </h3>
      <p className="text-sm text-center mb-5" style={{ color: 'var(--text-secondary)' }}>
        Your feedback helps our model learn and improve accuracy.
      </p>

      <AnimatePresence mode="wait">
        {!showCorrection ? (
          <motion.div
            key="buttons"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, y: -10 }}
            className="flex gap-4 justify-center"
          >
            <button
              onClick={() => submitFeedback(true)}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200"
              style={{
                background: 'rgba(52, 211, 153, 0.15)',
                border: '1px solid rgba(52, 211, 153, 0.4)',
                color: '#34d399',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(52, 211, 153, 0.25)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(52, 211, 153, 0.15)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <ThumbsUp className="w-5 h-5" />
              Correct
            </button>

            <button
              onClick={() => setShowCorrection(true)}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200"
              style={{
                background: 'rgba(248, 113, 113, 0.15)',
                border: '1px solid rgba(248, 113, 113, 0.4)',
                color: '#f87171',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(248, 113, 113, 0.25)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(248, 113, 113, 0.15)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              <ThumbsDown className="w-5 h-5" />
              Incorrect
            </button>
          </motion.div>
        ) : (
          <motion.div
            key="correction"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <p className="text-sm text-center font-semibold" style={{ color: 'var(--text-secondary)' }}>
              What is the correct classification?
            </p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={() => submitFeedback(false, 'real')}
                disabled={loading}
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200"
                style={{
                  background: 'rgba(52, 211, 153, 0.15)',
                  border: '1px solid rgba(52, 211, 153, 0.4)',
                  color: '#34d399',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(52, 211, 153, 0.25)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(52, 211, 153, 0.15)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                <Send className="w-4 h-4" />
                It&apos;s Real
              </button>

              <button
                onClick={() => submitFeedback(false, 'fake')}
                disabled={loading}
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200"
                style={{
                  background: 'rgba(248, 113, 113, 0.15)',
                  border: '1px solid rgba(248, 113, 113, 0.4)',
                  color: '#f87171',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(248, 113, 113, 0.25)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(248, 113, 113, 0.15)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                <Send className="w-4 h-4" />
                It&apos;s Fake
              </button>
            </div>

            <button
              onClick={() => setShowCorrection(false)}
              className="block mx-auto text-xs underline opacity-60 hover:opacity-100 transition-opacity"
              style={{ color: 'var(--text-secondary)' }}
            >
              ← Go back
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <p className="text-xs text-red-400 text-center mt-3">{error}</p>
      )}
    </motion.div>
  );
}
