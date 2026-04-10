'use client';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, AreaChart, Area,
} from 'recharts';
import AnimatedCounter from '@/components/AnimatedCounter';
import { TrendingUp, AlertTriangle, DollarSign, Users, Globe, ShieldAlert } from 'lucide-react';

const yearlyGrowth = [
  { year: '2019', deepfakes: 14678 },
  { year: '2020', deepfakes: 49081 },
  { year: '2021', deepfakes: 145230 },
  { year: '2022', deepfakes: 423000 },
  { year: '2023', deepfakes: 950000 },
  { year: '2024', deepfakes: 2100000 },
  { year: '2025', deepfakes: 4800000 },
];

const categoryData = [
  { name: 'Non-consensual', value: 48, color: '#f87171' },
  { name: 'Financial fraud', value: 27, color: '#fbbf24' },
  { name: 'Political', value: 15, color: '#818cf8' },
  { name: 'Entertainment', value: 10, color: '#34d399' },
];

const financialLoss = [
  { year: '2020', loss: 0.8 },
  { year: '2021', loss: 2.1 },
  { year: '2022', loss: 5.4 },
  { year: '2023', loss: 12.3 },
  { year: '2024', loss: 25.6 },
  { year: '2025', loss: 40.2 },
];

const monthlyDetections = [
  { month: 'Jan', detected: 12400, reported: 3200 },
  { month: 'Feb', detected: 14200, reported: 3800 },
  { month: 'Mar', detected: 18900, reported: 5100 },
  { month: 'Apr', detected: 22300, reported: 6400 },
  { month: 'May', detected: 28100, reported: 8200 },
  { month: 'Jun', detected: 31600, reported: 9700 },
  { month: 'Jul', detected: 35400, reported: 11200 },
  { month: 'Aug', detected: 39800, reported: 13500 },
  { month: 'Sep', detected: 44200, reported: 15800 },
  { month: 'Oct', detected: 48900, reported: 17200 },
  { month: 'Nov', detected: 52300, reported: 19100 },
  { month: 'Dec', detected: 58700, reported: 22400 },
];

const statCards = [
  { icon: TrendingUp, value: 550, suffix: '%', label: 'YoY Growth in Deepfakes', color: '#f87171' },
  { icon: DollarSign, value: 40, suffix: 'B+', label: 'Annual Financial Losses (USD)', color: '#fbbf24' },
  { icon: Users, value: 96, suffix: '%', label: 'Target Women & Minors', color: '#c084fc' },
  { icon: Globe, value: 190, suffix: '+', label: 'Countries Affected', color: '#34d399' },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};
const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

const tooltipStyle = {
  backgroundColor: 'rgba(15, 23, 42, 0.9)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: '8px',
  color: '#f1f5f9',
  fontSize: '12px',
};

export default function StatisticsPage() {
  return (
    <div className="max-w-7xl mx-auto px-4 py-12 space-y-12">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-4xl md:text-5xl font-display font-bold mb-3">
          Global <span className="gradient-text">Deepfake Threat</span> Dashboard
        </h1>
        <p className="max-w-2xl mx-auto" style={{ color: 'var(--text-secondary)' }}>
          Real-time statistics on the rise of AI-generated disinformation,
          deepfake-enabled fraud, and non-consensual synthetic media worldwide.
        </p>
      </motion.div>

      {/* Stat cards */}
      <motion.div
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {statCards.map((s, i) => (
          <motion.div key={i} variants={item} className="glass-card-hover p-6 text-center">
            <s.icon className="w-8 h-8 mx-auto mb-3" style={{ color: s.color }} />
            <AnimatedCounter to={s.value} suffix={s.suffix} className="text-3xl" />
            <p className="text-xs mt-2 font-medium" style={{ color: 'var(--text-secondary)' }}>
              {s.label}
            </p>
          </motion.div>
        ))}
      </motion.div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Yearly growth */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-red-400" />
            <span>Exponential Rise of Deepfakes (2019–2025)</span>
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={yearlyGrowth}>
              <defs>
                <linearGradient id="gradRed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f87171" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="year" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v) => `${(v / 1000000).toFixed(1)}M`} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [v.toLocaleString(), 'Deepfakes']} />
              <Area type="monotone" dataKey="deepfakes" stroke="#f87171" fill="url(#gradRed)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Category breakdown */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-400" />
            <span>Deepfake Content Categories</span>
          </h3>
          <div className="flex items-center justify-center gap-8">
            <ResponsiveContainer width="50%" height={250}>
              <PieChart>
                <Pie
                  data={categoryData}
                  cx="50%" cy="50%"
                  outerRadius={90}
                  innerRadius={55}
                  paddingAngle={4}
                  dataKey="value"
                  stroke="none"
                >
                  {categoryData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [`${v}%`, 'Share']} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-3">
              {categoryData.map((c, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: c.color }} />
                  <span style={{ color: 'var(--text-secondary)' }}>{c.name}</span>
                  <span className="font-semibold">{c.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Financial losses */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="w-4 h-4 text-yellow-400" />
            <span>Financial Fraud Losses (Billions USD)</span>
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={financialLoss}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="year" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v) => `$${v}B`} />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [`$${v}B`, 'Losses']} />
              <Bar dataKey="loss" fill="#fbbf24" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Monthly detections */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-brand-400" />
            <span>Monthly Detection vs. Reporting (2025)</span>
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={monthlyDetections}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="detected" stroke="#818cf8" strokeWidth={2} dot={false} name="Detected" />
              <Line type="monotone" dataKey="reported" stroke="#34d399" strokeWidth={2} dot={false} name="Reported" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-6 justify-center mt-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-brand-400 inline-block" /> Detected</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-green-400 inline-block" /> Reported</span>
          </div>
        </motion.div>
      </div>

      {/* Awareness CTA */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 text-center"
        style={{ borderColor: 'var(--accent)', boxShadow: '0 0 40px var(--accent-glow)' }}
      >
        <h2 className="text-2xl font-display font-bold mb-3">
          Don&apos;t be a victim. <span className="gradient-text">Verify before you trust.</span>
        </h2>
        <p className="mb-6 max-w-lg mx-auto" style={{ color: 'var(--text-secondary)' }}>
          Every day, millions of AI-generated images and videos are created. Protect yourself
          and your community by verifying suspicious media.
        </p>
        <a href="/upload" className="btn-primary inline-block">
          Analyze Suspicious Media →
        </a>
      </motion.div>
    </div>
  );
}
