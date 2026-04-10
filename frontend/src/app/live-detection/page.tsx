'use client';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { VideoOff, ArrowLeft, Clock } from 'lucide-react';

export default function LiveDetectionPage() {
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center px-4 text-center gap-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col items-center gap-6"
      >
        {/* Icon */}
        <div
          className="w-24 h-24 rounded-full flex items-center justify-center"
          style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.25)' }}
        >
          <VideoOff className="w-10 h-10" style={{ color: 'var(--accent)' }} />
        </div>

        {/* Heading */}
        <div>
          <h1 className="text-3xl md:text-4xl font-display font-bold mb-3">
            Live Detection <span className="gradient-text">Coming Soon</span>
          </h1>
          <p className="max-w-md text-sm" style={{ color: 'var(--text-secondary)' }}>
            Real-time webcam deepfake detection is temporarily disabled while we improve
            accuracy and performance. It will be back shortly.
          </p>
        </div>

        {/* Badge */}
        <div
          className="flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold"
          style={{
            background: 'rgba(251,191,36,0.1)',
            border: '1px solid rgba(251,191,36,0.3)',
            color: '#fbbf24',
          }}
        >
          <Clock className="w-3.5 h-3.5" />
          Temporarily Disabled
        </div>

        {/* CTA */}
        <Link href="/upload" className="btn-primary flex items-center gap-2 text-sm !px-6 !py-3">
          <ArrowLeft className="w-4 h-4" />
          Analyse Media Instead
        </Link>
      </motion.div>
    </div>
  );
}
