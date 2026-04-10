/**
 * VeritasAI — Color Palette
 * Matches the web app's CSS variables for dark/light themes.
 */

export const Colors = {
  dark: {
    bgGradientFrom: '#0f0c29',
    bgGradientVia: '#302b63',
    bgGradientTo: '#24243e',
    glassBg: 'rgba(255, 255, 255, 0.06)',
    glassBorder: 'rgba(255, 255, 255, 0.1)',
    glassShadow: 'rgba(0, 0, 0, 0.37)',
    textPrimary: '#f1f5f9',
    textSecondary: '#94a3b8',
    accent: '#818cf8',
    accentGlow: 'rgba(129, 140, 248, 0.3)',
    danger: '#f87171',
    success: '#34d399',
    warning: '#fbbf24',
    cardBg: 'rgba(255, 255, 255, 0.04)',
    tabBarBg: 'rgba(15, 12, 41, 0.95)',
    inputBg: 'rgba(255, 255, 255, 0.08)',
    overlay: 'rgba(0, 0, 0, 0.6)',
  },
  light: {
    bgGradientFrom: '#f0f4ff',
    bgGradientVia: '#e0e7ff',
    bgGradientTo: '#f8fafc',
    glassBg: 'rgba(255, 255, 255, 0.7)',
    glassBorder: 'rgba(148, 163, 184, 0.25)',
    glassShadow: 'rgba(0, 0, 0, 0.08)',
    textPrimary: '#0f172a',
    textSecondary: '#475569',
    accent: '#6366f1',
    accentGlow: 'rgba(99, 102, 241, 0.2)',
    danger: '#ef4444',
    success: '#10b981',
    warning: '#f59e0b',
    cardBg: 'rgba(255, 255, 255, 0.8)',
    tabBarBg: 'rgba(240, 244, 255, 0.95)',
    inputBg: 'rgba(0, 0, 0, 0.04)',
    overlay: 'rgba(0, 0, 0, 0.3)',
  },
} as const;

export const Gradients = {
  primary: ['#6366f1', '#8b5cf6'] as const,
  accent: ['#818cf8', '#c084fc', '#f472b6'] as const,
  background: {
    dark: ['#0f0c29', '#302b63', '#24243e'] as const,
    light: ['#f0f4ff', '#e0e7ff', '#f8fafc'] as const,
  },
  danger: ['#ef4444', '#f87171'] as const,
  success: ['#10b981', '#34d399'] as const,
} as const;

export type ThemeColors = typeof Colors.dark;
