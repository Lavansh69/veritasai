'use client';
import { motion } from 'framer-motion';
import Link from 'next/link';
import AnimatedCounter from '@/components/AnimatedCounter';
import { Shield, Scan, FileText, BarChart3, ArrowRight, Zap, Eye, Lock } from 'lucide-react';

const features = [
  {
    icon: Scan,
    title: 'AI Detection',
    desc: 'EfficientNet & XceptionNet models trained on millions of deepfake samples detect manipulations with high accuracy.',
  },
  {
    icon: Eye,
    title: 'Explainable AI',
    desc: 'Grad-CAM heatmaps visually reveal which facial regions triggered the AI, making decisions transparent.',
  },
  {
    icon: BarChart3,
    title: 'Multi-Factor Scoring',
    desc: 'Authenticity scorecard combines deepfake probability, face consistency, metadata integrity, and artifact detection.',
  },
  {
    icon: FileText,
    title: 'Forensic Reports',
    desc: 'Download structured PDF evidence reports ready for cybercrime reporting and legal proceedings.',
  },
  {
    icon: Lock,
    title: 'Privacy First',
    desc: 'All uploads are encrypted, temporarily stored, and automatically deleted after analysis completes.',
  },
  {
    icon: Zap,
    title: 'Real-Time Analysis',
    desc: 'Get results in seconds. Our optimized pipeline processes images and videos at blazing speed.',
  },
];

const stats = [
  { value: 96, suffix: '%', label: 'Detection Accuracy' },
  { value: 500, suffix: 'K+', label: 'Deepfakes Analyzed' },
  { value: 50, suffix: 'M+', label: 'Frames Processed' },
  { value: 128, suffix: '+', label: 'Countries Served' },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};
const item = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function HomePage() {
  return (
    <div className="relative">
      {/* ── Hero ──────────────────────────────────────────── */}
      <section className="min-h-[90vh] flex items-center justify-center px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <motion.div
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-xs font-medium mb-8"
              style={{ color: 'var(--text-secondary)' }}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              Powered by The Verifiers
            </motion.div>

            <h1 className="text-5xl md:text-7xl font-display font-extrabold leading-tight mb-6">
              Unmask the{' '}
              <span className="gradient-text">Deepfakes</span>
              <br />
              Protect the Truth
            </h1>

            <p
              className="text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed"
              style={{ color: 'var(--text-secondary)' }}
            >
              VeritasAI uses cutting-edge neural networks to detect AI-generated media,
              analyze forensic metadata, and generate court-ready evidence reports —
              all in seconds.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/upload">
                <motion.button
                  className="btn-primary text-lg flex items-center gap-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.97 }}
                >
                  <Shield className="w-5 h-5" />
                  Analyze Media
                  <ArrowRight className="w-4 h-4" />
                </motion.button>
              </Link>
              <Link href="/statistics">
                <motion.button
                  className="btn-outline text-lg"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.97 }}
                >
                  View Global Threat Data
                </motion.button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Stats bar ────────────────────────────────────── */}
      <section className="py-12 px-4">
        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-80px' }}
          className="max-w-5xl mx-auto glass-card py-8 px-6 grid grid-cols-2 md:grid-cols-4 gap-6 text-center"
        >
          {stats.map((s, i) => (
            <motion.div key={i} variants={item}>
              <AnimatedCounter
                to={s.value}
                suffix={s.suffix}
                className="text-4xl gradient-text"
              />
              <p className="text-xs mt-2 font-medium" style={{ color: 'var(--text-secondary)' }}>
                {s.label}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ── Features ─────────────────────────────────────── */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl md:text-4xl font-display font-bold text-center mb-16"
          >
            How <span className="gradient-text">VeritasAI</span> Works
          </motion.h2>

          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {features.map((f, i) => (
              <motion.div
                key={i}
                variants={item}
                className="glass-card-hover p-6 cursor-default"
              >
                <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                  style={{ background: 'var(--accent-glow)' }}
                >
                  <f.icon className="w-6 h-6" style={{ color: 'var(--accent)' }} />
                </div>
                <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {f.desc}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────── */}
      <section className="py-20 px-4">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-3xl mx-auto glass-card p-12 text-center"
          style={{ borderColor: 'var(--accent)', boxShadow: '0 0 60px var(--accent-glow)' }}
        >
          <h2 className="text-3xl font-display font-bold mb-4">
            Ready to Verify?
          </h2>
          <p className="mb-8" style={{ color: 'var(--text-secondary)' }}>
            Upload any suspicious image or video and let our AI reveal the truth.
          </p>
          <Link href="/upload">
            <motion.button
              className="btn-primary text-lg"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
            >
              Start Analysis →
            </motion.button>
          </Link>
        </motion.div>
      </section>
    </div>
  );
}
