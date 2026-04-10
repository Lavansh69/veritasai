/**
 * VeritasAI — ScoreCard Component
 * Displays the verdict with animated score and color-coded styling.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { GlassCard } from './ui/GlassCard';
import { useTheme } from '../theme/ThemeContext';
import { FontSizes, Spacing, BorderRadius } from '../theme/typography';

interface ScoreCardProps {
  verdict: string;
  overallScore: number;
  confidence: string;
  deepfakeProbability: number;
}

export function ScoreCard({
  verdict,
  overallScore,
  confidence,
  deepfakeProbability,
}: ScoreCardProps) {
  const { colors } = useTheme();

  const getVerdictStyle = () => {
    const prob = deepfakeProbability;
    if (prob < 0.3)
      return {
        icon: 'shield-checkmark' as const,
        color: colors.success,
        gradient: ['#10b981', '#34d399'] as string[],
        label: 'Likely Authentic',
      };
    if (prob < 0.7)
      return {
        icon: 'warning' as const,
        color: colors.warning,
        gradient: ['#f59e0b', '#fbbf24'] as string[],
        label: 'Suspicious',
      };
    return {
      icon: 'alert-circle' as const,
      color: colors.danger,
      gradient: ['#ef4444', '#f87171'] as string[],
      label: 'Likely Deepfake',
    };
  };

  const vs = getVerdictStyle();

  return (
    <GlassCard glowing style={styles.card}>
      <View style={styles.content}>
        {/* Icon */}
        <View style={[styles.iconContainer, { backgroundColor: vs.color + '20' }]}>
          <Ionicons name={vs.icon} size={40} color={vs.color} />
        </View>

        {/* Verdict */}
        <Text style={[styles.verdict, { color: vs.color }]}>{vs.label}</Text>

        {/* Score */}
        <Text style={[styles.score, { color: colors.textPrimary }]}>
          {Math.round(overallScore)}
          <Text style={styles.scoreUnit}>/100</Text>
        </Text>
        <Text style={[styles.scoreLabel, { color: colors.textSecondary }]}>
          Authenticity Score
        </Text>

        {/* Confidence badge */}
        <View style={[styles.badge, { backgroundColor: vs.color + '20' }]}>
          <Text style={[styles.badgeText, { color: vs.color }]}>
            {confidence} Confidence
          </Text>
        </View>

        {/* Probability bar */}
        <View style={styles.probSection}>
          <Text style={[styles.probLabel, { color: colors.textSecondary }]}>
            Deepfake Probability
          </Text>
          <View style={[styles.probBar, { backgroundColor: colors.glassBorder }]}>
            <LinearGradient
              colors={vs.gradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={[
                styles.probFill,
                { width: `${Math.round(deepfakeProbability * 100)}%` as any },
              ]}
            />
          </View>
          <Text style={[styles.probValue, { color: vs.color }]}>
            {(deepfakeProbability * 100).toFixed(1)}%
          </Text>
        </View>
      </View>
    </GlassCard>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.xl,
  },
  content: {
    alignItems: 'center',
    gap: Spacing.sm,
  },
  iconContainer: {
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: Spacing.xs,
  },
  verdict: {
    fontSize: FontSizes.xl,
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  score: {
    fontSize: FontSizes['4xl'],
    fontWeight: '800',
  },
  scoreUnit: {
    fontSize: FontSizes.lg,
    fontWeight: '400',
    opacity: 0.5,
  },
  scoreLabel: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  badge: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
    marginTop: Spacing.xs,
  },
  badgeText: {
    fontSize: FontSizes.xs,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  probSection: {
    width: '100%',
    marginTop: Spacing.base,
    gap: Spacing.xs,
  },
  probLabel: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  probBar: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  probFill: {
    height: '100%',
    borderRadius: 4,
  },
  probValue: {
    fontSize: FontSizes.md,
    fontWeight: '700',
    textAlign: 'right',
  },
});
