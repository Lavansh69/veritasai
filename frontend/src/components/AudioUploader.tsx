'use client';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Music, X, AlertTriangle, CheckCircle, Info, Loader2 } from 'lucide-react';

interface AudioResult {
  analysis_id: string;
  deepfake_probability: number;
  verdict: string;
  demo_mode: boolean;
}

export default function AudioUploader() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AudioResult | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length > 0) {
      setAudioFile(accepted[0]);
      setResult(null);
      setError(null);
    }
  }, []);

  const dropzone = useDropzone({
    onDrop,
    accept: {
      'audio/wav': ['.wav'],
      'audio/mpeg': ['.mp3'],
      'audio/flac': ['.flac'],
      'audio/ogg': ['.ogg'],
    },
    maxSize: 100 * 1024 * 1024,
    multiple: false,
  });

  const handleAnalyzeAudio = async () => {
    if (!audioFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('audio', audioFile);

      const res = await fetch('/api/audio/analyze', { method: 'POST', body: formData });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Audio analysis failed (${res.status})`);
      }

      const data: AudioResult = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Audio analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getVerdictColor = (verdict: string) => {
    if (verdict.includes('Deepfake')) return 'text-red-400';
    if (verdict.includes('Suspicious')) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getVerdictIcon = (verdict: string) => {
    if (verdict.includes('Deepfake')) return <AlertTriangle className="w-6 h-6 text-red-400" />;
    if (verdict.includes('Suspicious')) return <Info className="w-6 h-6 text-yellow-400" />;
    return <CheckCircle className="w-6 h-6 text-green-400" />;
  };

  const getProbColor = (prob: number) => {
    if (prob >= 0.85) return '#f87171';
    if (prob >= 0.65) return '#fbbf24';
    return '#34d399';
  };

  return (
    <div className="space-y-6">
      {/* Audio upload area */}
      <div>
        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Suspicious Audio <span className="text-red-400">*</span>
        </label>
        <motion.div
          whileHover={{ scale: 1.01 }}
          className={`glass-card p-8 text-center cursor-pointer transition-all duration-300 border-2 border-dashed ${
            dropzone.isDragActive ? 'border-emerald-400 bg-emerald-500/10' : 'border-white/10'
          }`}
          {...(dropzone.getRootProps() as any)}
        >
          <input {...dropzone.getInputProps()} id="audio-upload" />
          <AnimatePresence mode="wait">
            {audioFile ? (
              <motion.div
                key="file"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center justify-center gap-3"
              >
                <Music className="w-8 h-8 text-emerald-400" />
                <span className="font-medium">{audioFile.name}</span>
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  ({(audioFile.size / 1024 / 1024).toFixed(1)} MB)
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); setAudioFile(null); setResult(null); }}
                  className="p-1 rounded-full hover:bg-red-500/20 text-red-400"
                >
                  <X className="w-4 h-4" />
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-emerald-500/10 flex items-center justify-center">
                  <Music className="w-8 h-8 text-emerald-400 opacity-70" />
                </div>
                <p className="font-medium">Drop your suspicious audio file here</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                  Supports WAV, MP3, FLAC, OGG — up to 100 MB
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Analyze button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        disabled={!audioFile || loading}
        onClick={handleAnalyzeAudio}
        className={`w-full py-4 text-lg font-bold flex items-center justify-center gap-2 rounded-xl transition-all duration-300 ${
          (!audioFile || loading)
            ? 'opacity-50 cursor-not-allowed bg-emerald-800/30 text-emerald-300'
            : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-lg shadow-emerald-500/20'
        }`}
        id="analyze-audio-button"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing Audio…
          </>
        ) : (
          <>
            <WaveformIcon className="w-5 h-5" />
            Analyze Audio
          </>
        )}
      </motion.button>

      {/* Error message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm"
        >
          {error}
        </motion.div>
      )}

      {/* Result card */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            transition={{ duration: 0.5, type: 'spring' }}
            className="glass-card p-6 space-y-4"
          >
            <div className="flex items-center gap-3">
              {getVerdictIcon(result.verdict)}
              <h3 className={`text-xl font-display font-bold ${getVerdictColor(result.verdict)}`}>
                {result.verdict}
              </h3>
            </div>

            {/* Probability bar */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span style={{ color: 'var(--text-secondary)' }}>Deepfake Probability</span>
                <span className="font-bold" style={{ color: getProbColor(result.deepfake_probability) }}>
                  {(result.deepfake_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${result.deepfake_probability * 100}%` }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full rounded-full"
                  style={{
                    background: `linear-gradient(90deg, ${getProbColor(result.deepfake_probability)}88, ${getProbColor(result.deepfake_probability)})`,
                  }}
                />
              </div>
            </div>

            {/* Analysis ID */}
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Analysis ID: {result.analysis_id}
            </p>

            {/* Demo mode warning */}
            {result.demo_mode && (
              <div className="flex items-start gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                <Info className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" />
                <p className="text-xs text-yellow-300">
                  <strong>Demo Mode:</strong> No trained audio model found. These results are generated randomly.
                  Train the audio model using <code className="px-1 py-0.5 rounded bg-white/5">train_audio.py</code> for real predictions.
                </p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function WaveformIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M2 12h2m4-7v14m4-10v6m4-8v10m4-4v-2" />
    </svg>
  );
}
