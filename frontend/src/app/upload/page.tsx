'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import FileUploader from '@/components/FileUploader';
import AudioUploader from '@/components/AudioUploader';
import ScanningOverlay from '@/components/ScanningOverlay';
import { Info, Music } from 'lucide-react';

export default function UploadPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanFile, setScanFile] = useState<File | null>(null);

  const handleAnalyze = async (media: File, reference?: File) => {
    setLoading(true);
    setError(null);
    setScanFile(media);

    try {
      const formData = new FormData();
      formData.append('media', media);
      if (reference) formData.append('reference', reference);

      const res = await fetch('/api/analyze', { method: 'POST', body: formData });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Analysis failed (${res.status})`);
      }

      const result = await res.json();
      // Store result in sessionStorage for the analysis page
      sessionStorage.setItem('veritasai-result', JSON.stringify(result));
      router.push(`/analysis/${result.analysis_id}`);
    } catch (err: any) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <ScanningOverlay isActive={loading} file={scanFile} />
      <div className="min-h-[80vh] flex flex-col items-center justify-start px-4 py-12 gap-16">
      {/* ═══════════════════════ Image / Video Section ═══════════════════════ */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-2xl"
      >
        <div className="text-center mb-10">
          <h1 className="text-4xl font-display font-bold mb-3">
            Upload <span className="gradient-text">Suspicious Media</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)' }}>
            Upload an image or video to analyze for deepfake manipulation
          </p>
        </div>

        <div className="glass-card p-8">
          <FileUploader onFilesSelected={handleAnalyze} isLoading={loading} />

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm"
            >
              {error}
            </motion.div>
          )}
        </div>

        {/* Info card */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="glass-card p-5 mt-6 flex gap-3"
        >
          <Info className="w-5 h-5 text-brand-400 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1" style={{ color: 'var(--text-secondary)' }}>
            <p><strong>Supported formats:</strong> JPG, PNG, MP4, MOV (max 100 MB)</p>
            <p><strong>Reference image:</strong> Optionally upload a photo of the real person to compare identity consistency</p>
            <p><strong>Privacy:</strong> All files are encrypted and automatically deleted after analysis</p>
          </div>
        </motion.div>
      </motion.div>

      {/* ═══════════════════════ Divider ═══════════════════════ */}
      <div className="w-full max-w-2xl flex items-center gap-4">
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        <span className="text-xs font-medium tracking-widest uppercase" style={{ color: 'var(--text-secondary)' }}>
          or analyze audio
        </span>
        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      </div>

      {/* ═══════════════════════ Audio Section ═══════════════════════ */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="w-full max-w-2xl"
      >
        <div className="text-center mb-10">
          <h2 className="text-3xl font-display font-bold mb-3 flex items-center justify-center gap-3">
            <Music className="w-8 h-8 text-emerald-400" />
            Audio <span style={{ background: 'linear-gradient(135deg, #34d399, #06b6d4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Deepfake Detection</span>
          </h2>
          <p style={{ color: 'var(--text-secondary)' }}>
            Upload an audio file to detect AI-generated or cloned voices
          </p>
        </div>

        <div className="glass-card p-8">
          <AudioUploader />
        </div>

        {/* Audio info card */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="glass-card p-5 mt-6 flex gap-3"
        >
          <Music className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1" style={{ color: 'var(--text-secondary)' }}>
            <p><strong>Supported formats:</strong> WAV, MP3, FLAC, OGG (max 100 MB)</p>
            <p><strong>Detection:</strong> Analyzes Mel-spectrogram patterns to identify AI-generated or voice-cloned audio</p>
            <p><strong>Privacy:</strong> Audio files are automatically deleted after analysis</p>
          </div>
        </motion.div>
      </motion.div>
      </div>
    </>
  );
}
