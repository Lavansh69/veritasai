/**
 * VeritasAI — Analysis Results Screen
 * Displays full analysis results: verdict, scores, heatmap, feedback.
 */

import React, { useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, router } from 'expo-router';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../../theme/typography';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { ScoreCard } from '../../components/ScoreCard';
import { GaugeChart } from '../../components/GaugeChart';
import { HeatmapViewer } from '../../components/HeatmapViewer';
import { FeedbackPanel } from '../../components/FeedbackPanel';
import { getHeatmapUrl, type AnalysisResult } from '../../services/api';

export default function AnalysisResultScreen() {
  const { colors, isDark } = useTheme();
  const { id, data } = useLocalSearchParams<{ id: string; data: string }>();

  const result: AnalysisResult | null = useMemo(() => {
    try {
      return data ? JSON.parse(data) : null;
    } catch {
      return null;
    }
  }, [data]);

  if (!result) {
    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={48} color={colors.danger} />
          <Text style={[styles.errorText, { color: colors.textPrimary }]}>
            No analysis data available
          </Text>
          <Button
            title="Go Back"
            variant="outline"
            onPress={() => router.back()}
          />
        </View>
      </LinearGradient>
    );
  }

  const scores = result.scorecard?.scores;

  return (
    <LinearGradient
      colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
      style={styles.gradient}
    >
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Main Verdict Card */}
        <ScoreCard
          verdict={result.scorecard?.verdict || result.deepfake?.label || 'Unknown'}
          overallScore={result.scorecard?.overall_score || 0}
          confidence={result.scorecard?.confidence || 'Medium'}
          deepfakeProbability={result.deepfake?.probability || 0}
        />

        {/* Individual Score Gauges */}
        {scores && (
          <GlassCard style={styles.gaugesCard}>
            <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
              Detailed Scores
            </Text>
            <View style={styles.gaugesGrid}>
              <GaugeChart
                value={scores.ai_detection}
                label="AI Detection"
                size={90}
                strokeWidth={7}
              />
              <GaugeChart
                value={scores.face_analysis}
                label="Face Analysis"
                size={90}
                strokeWidth={7}
              />
              <GaugeChart
                value={scores.metadata_integrity}
                label="Metadata"
                size={90}
                strokeWidth={7}
              />
              <GaugeChart
                value={scores.artifact_detection}
                label="Artifacts"
                size={90}
                strokeWidth={7}
              />
            </View>
          </GlassCard>
        )}

        {/* Face Consistency */}
        {result.face_consistency && (
          <GlassCard style={styles.detailCard}>
            <View style={styles.detailHeader}>
              <Ionicons name="person" size={20} color={colors.accent} />
              <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
                Face Consistency
              </Text>
            </View>
            <View style={styles.scoreRow}>
              <Text style={[styles.scoreLabel, { color: colors.textSecondary }]}>Score</Text>
              <Text
                style={[
                  styles.scoreValue,
                  {
                    color:
                      result.face_consistency.score >= 70
                        ? colors.success
                        : result.face_consistency.score >= 40
                        ? colors.warning
                        : colors.danger,
                  },
                ]}
              >
                {result.face_consistency.score}/100
              </Text>
            </View>
          </GlassCard>
        )}

        {/* Metadata Warnings */}
        {result.metadata?.warnings && result.metadata.warnings.length > 0 && (
          <GlassCard style={styles.detailCard}>
            <View style={styles.detailHeader}>
              <Ionicons name="warning" size={20} color={colors.warning} />
              <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
                Metadata Warnings
              </Text>
            </View>
            {result.metadata.warnings.map((warning, i) => (
              <View key={i} style={styles.warningRow}>
                <Ionicons name="ellipse" size={6} color={colors.warning} />
                <Text style={[styles.warningText, { color: colors.textSecondary }]}>
                  {warning}
                </Text>
              </View>
            ))}
            <View style={styles.scoreRow}>
              <Text style={[styles.scoreLabel, { color: colors.textSecondary }]}>
                Integrity Score
              </Text>
              <Text style={[styles.scoreValue, { color: colors.accent }]}>
                {result.metadata.integrity_score}/100
              </Text>
            </View>
          </GlassCard>
        )}

        {/* Heatmap */}
        {result.heatmap?.heatmap_url && (
          <HeatmapViewer
            heatmapUrl={result.heatmap.heatmap_url}
            overlayUrl={result.heatmap.overlay_url}
          />
        )}

        {/* Feedback */}
        <FeedbackPanel
          analysisId={result.analysis_id}
          prediction={result.deepfake?.label || 'unknown'}
          confidence={result.deepfake?.probability || 0}
        />

        {/* Actions */}
        <View style={styles.actionsRow}>
          <Button
            title="Download Report"
            variant="outline"
            onPress={() =>
              router.push({
                pathname: '/report/[id]',
                params: { id: result.analysis_id },
              })
            }
            icon={<Ionicons name="download" size={18} color={colors.textPrimary} />}
            style={styles.actionBtn}
          />
          <Button
            title="New Analysis"
            onPress={() => router.replace('/upload')}
            icon={<Ionicons name="add" size={18} color="#fff" />}
            style={styles.actionBtn}
          />
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  container: { flex: 1 },
  content: {
    padding: Spacing.lg,
    paddingBottom: 40,
    gap: Spacing.base,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: Spacing.base,
    padding: Spacing['2xl'],
  },
  errorText: {
    fontSize: FontSizes.md,
    fontWeight: '600',
    textAlign: 'center',
  },
  gaugesCard: {
    padding: Spacing.lg,
    gap: Spacing.base,
  },
  sectionTitle: {
    fontSize: FontSizes.md,
    fontWeight: '700',
  },
  gaugesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    gap: Spacing.base,
  },
  detailCard: {
    padding: Spacing.lg,
    gap: Spacing.sm,
  },
  detailHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  scoreRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: Spacing.xs,
  },
  scoreLabel: {
    fontSize: FontSizes.base,
    fontWeight: '500',
  },
  scoreValue: {
    fontSize: FontSizes.lg,
    fontWeight: '800',
  },
  warningRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: Spacing.sm,
    paddingLeft: Spacing.xs,
  },
  warningText: {
    flex: 1,
    fontSize: FontSizes.sm,
    lineHeight: 20,
  },
  actionsRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  actionBtn: {
    flex: 1,
  },
});
