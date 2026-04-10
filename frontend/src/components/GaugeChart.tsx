'use client';
import { motion, useMotionValue, useTransform } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

interface GaugeChartProps {
  value: number;       // 0-100
  label: string;
  color?: string;
  size?: number;
}

export default function GaugeChart({ value, label, color = '#818cf8', size = 160 }: GaugeChartProps) {
  const [displayed, setDisplayed] = useState(0);
  const radius = (size - 20) / 2;
  const circumference = Math.PI * radius;

  useEffect(() => {
    const timer = setTimeout(() => setDisplayed(value), 100);
    return () => clearTimeout(timer);
  }, [value]);

  const offset = circumference - (displayed / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size / 2 + 20 }}>
        <svg
          width={size}
          height={size / 2 + 20}
          viewBox={`0 0 ${size} ${size / 2 + 20}`}
        >
          {/* Background arc */}
          <path
            d={`M 10 ${size / 2 + 10} A ${radius} ${radius} 0 0 1 ${size - 10} ${size / 2 + 10}`}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth="12"
            strokeLinecap="round"
          />
          {/* Value arc */}
          <motion.path
            d={`M 10 ${size / 2 + 10} A ${radius} ${radius} 0 0 1 ${size - 10} ${size / 2 + 10}`}
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
            style={{ filter: `drop-shadow(0 0 8px ${color}40)` }}
          />
        </svg>
        {/* Center value */}
        <div
          className="absolute inset-0 flex items-end justify-center pb-2"
          style={{ fontFamily: 'Outfit, sans-serif' }}
        >
          <motion.span
            className="text-3xl font-bold"
            style={{ color }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {displayed.toFixed(0)}%
          </motion.span>
        </div>
      </div>
      <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </span>
    </div>
  );
}
