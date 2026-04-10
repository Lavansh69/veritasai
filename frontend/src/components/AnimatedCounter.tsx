'use client';
import { motion, useInView } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

interface AnimatedCounterProps {
  to: number;
  suffix?: string;
  prefix?: string;
  duration?: number;
  decimals?: number;
  className?: string;
}

export default function AnimatedCounter({
  to,
  suffix = '',
  prefix = '',
  duration = 2,
  decimals = 0,
  className = '',
}: AnimatedCounterProps) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-50px' });
  const [value, setValue] = useState(0);

  useEffect(() => {
    if (!inView) return;

    let start = 0;
    const step = to / (duration * 60);
    const interval = setInterval(() => {
      start += step;
      if (start >= to) {
        start = to;
        clearInterval(interval);
      }
      setValue(start);
    }, 1000 / 60);

    return () => clearInterval(interval);
  }, [inView, to, duration]);

  return (
    <motion.span
      ref={ref}
      className={`font-display font-bold ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.4 }}
    >
      {prefix}{value.toFixed(decimals)}{suffix}
    </motion.span>
  );
}
