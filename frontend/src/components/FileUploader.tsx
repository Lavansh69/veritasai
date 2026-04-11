'use client';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileImage, FileVideo, X, Image as ImageIcon } from 'lucide-react';

interface FileUploaderProps {
  onFilesSelected: (media: File, reference?: File) => void;
  isLoading?: boolean;
}

export default function FileUploader({ onFilesSelected, isLoading }: FileUploaderProps) {
  const [mediaFile, setMediaFile] = useState<File | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);

  const onDropMedia = useCallback((accepted: File[]) => {
    if (accepted.length > 0) setMediaFile(accepted[0]);
  }, []);

  const onDropRef = useCallback((accepted: File[]) => {
    if (accepted.length > 0) setRefFile(accepted[0]);
  }, []);

  const mediaDz = useDropzone({
    onDrop: onDropMedia,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'video/mp4': ['.mp4'],
      'video/quicktime': ['.mov'],
    },
    maxSize: 100 * 1024 * 1024,
    multiple: false,
  });

  const refDz = useDropzone({
    onDrop: onDropRef,
    accept: { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] },
    maxSize: 100 * 1024 * 1024,
    multiple: false,
  });

  const handleAnalyze = () => {
    if (mediaFile) onFilesSelected(mediaFile, refFile ?? undefined);
  };

  const fileIcon = (file: File) => {
    if (file.type.startsWith('video')) return <FileVideo className="w-8 h-8 text-purple-400" />;
    return <FileImage className="w-8 h-8 text-brand-400" />;
  };

  return (
    <div className="space-y-6">
      {/* Main media upload */}
      <div>
        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Suspicious Media <span className="text-red-400">*</span>
        </label>
        <div
          className={`glass-card p-8 text-center cursor-pointer transition-all duration-300 hover:scale-[1.01] border-2 border-dashed ${
            mediaDz.isDragActive ? 'border-brand-400 bg-brand-500/10' : 'border-white/10'
          }`}
          {...mediaDz.getRootProps()}
        >
          <input {...mediaDz.getInputProps()} id="media-upload" />
          <AnimatePresence mode="wait">
            {mediaFile ? (
              <motion.div
                key="file"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center justify-center gap-3"
              >
                {fileIcon(mediaFile)}
                <span className="font-medium">{mediaFile.name}</span>
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  ({(mediaFile.size / 1024 / 1024).toFixed(1)} MB)
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); setMediaFile(null); }}
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
                <Upload className="w-12 h-12 mx-auto mb-3 text-brand-400 opacity-60" />
                <p className="font-medium">Drop your suspicious image or video here</p>
                <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                  Supports JPG, PNG, MP4, MOV — up to 100 MB
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Reference image upload */}
      <div>
        <label className="block text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
          Reference Image <span className="text-xs opacity-60">(optional)</span>
        </label>
        <div
          className={`glass-card p-6 text-center cursor-pointer transition-all duration-300 hover:scale-[1.01] border-2 border-dashed ${
            refDz.isDragActive ? 'border-purple-400 bg-purple-500/10' : 'border-white/10'
          }`}
          {...refDz.getRootProps()}
        >
          <input {...refDz.getInputProps()} id="reference-upload" />
          {refFile ? (
            <div className="flex items-center justify-center gap-3">
              <ImageIcon className="w-6 h-6 text-purple-400" />
              <span className="text-sm font-medium">{refFile.name}</span>
              <button
                onClick={(e) => { e.stopPropagation(); setRefFile(null); }}
                className="p-1 rounded-full hover:bg-red-500/20 text-red-400"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
              <ImageIcon className="w-5 h-5 opacity-60" />
              <span>Upload a reference photo of the real person for identity comparison</span>
            </div>
          )}
        </div>
      </div>

      {/* Analyze button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        disabled={!mediaFile || isLoading}
        onClick={handleAnalyze}
        className={`w-full btn-primary py-4 text-lg font-bold flex items-center justify-center gap-2 ${
          (!mediaFile || isLoading) ? 'opacity-50 cursor-not-allowed' : ''
        }`}
        id="analyze-button"
      >
        {isLoading ? (
          <>
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
              className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
            />
            Analyzing…
          </>
        ) : (
          <>
            <Shield className="w-5 h-5" />
            Analyze Media
          </>
        )}
      </motion.button>
    </div>
  );
}

function Shield(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
