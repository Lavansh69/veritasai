/**
 * VeritasAI — FeedbackPanel Component
 * Allows users to confirm or correct analysis results.
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GlassCard } from './ui/GlassCard';
import { Button } from './ui/Button';
import { useTheme } from '../theme/ThemeContext';
import { submitFeedback } from '../services/api';
import { FontSizes, Spacing, BorderRadius } from '../theme/typography';

interface FeedbackPanelProps {
  analysisId: string;
  prediction: string;
  confidence: number;
}

export function FeedbackPanel({ analysisId, prediction, confidence }: FeedbackPanelProps) {
  const { colors } = useTheme();
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleFeedback = async (isCorrect: boolean, correctedLabel?: string) => {
    setLoading(true);
    try {
      await submitFeedback({
        analysis_id: analysisId,
        is_correct: isCorrect,
        corrected_label: correctedLabel || null,
        prediction,
        confidence,
      });
      setSubmitted(true);
    } catch (err: any) {
      Alert.alert('Error', err.message || 'Failed to submit feedback');
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <GlassCard style={styles.card}>
        <View style={styles.successContainer}>
          <Ionicons name="checkmark-circle" size={32} color={colors.success} />
          <Text style={[styles.successText, { color: colors.success }]}>
            Thank you for your feedback!
          </Text>
        </View>
      </GlassCard>
    );
  }

  return (
    <GlassCard style={styles.card}>
      <Text style={[styles.title, { color: colors.textPrimary }]}>
        Was this analysis correct?
      </Text>
      <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
        Your feedback helps improve our AI model
      </Text>

      <View style={styles.buttonRow}>
        <Button
          title="Correct"
          variant="success"
          icon={<Ionicons name="checkmark" size={18} color={colors.success} />}
          onPress={() => handleFeedback(true)}
          loading={loading}
          style={styles.feedbackBtn}
          size="sm"
        />
        <Button
          title="It's Real"
          variant="outline"
          icon={<Ionicons name="person" size={18} color={colors.textSecondary} />}
          onPress={() => handleFeedback(false, 'real')}
          loading={loading}
          style={styles.feedbackBtn}
          size="sm"
        />
        <Button
          title="It's Fake"
          variant="danger"
          icon={<Ionicons name="alert" size={18} color={colors.danger} />}
          onPress={() => handleFeedback(false, 'fake')}
          loading={loading}
          style={styles.feedbackBtn}
          size="sm"
        />
      </View>
    </GlassCard>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.lg,
  },
  title: {
    fontSize: FontSizes.md,
    fontWeight: '700',
    marginBottom: Spacing.xs,
  },
  subtitle: {
    fontSize: FontSizes.sm,
    marginBottom: Spacing.base,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  feedbackBtn: {
    flex: 1,
  },
  successContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.sm,
    paddingVertical: Spacing.sm,
  },
  successText: {
    fontSize: FontSizes.base,
    fontWeight: '600',
  },
});
