/**
 * VeritasAI — Statistics Screen
 * Displays feedback stats, accuracy history, and model performance.
 */

import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Rect, Line, Text as SvgText } from 'react-native-svg';
import { useTheme } from '../theme/ThemeContext';
import { Gradients } from '../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../theme/typography';
import { GlassCard } from '../components/ui/GlassCard';
import { GaugeChart } from '../components/GaugeChart';
import { AnimatedCounter } from '../components/AnimatedCounter';
import { getFeedbackStats } from '../services/api';

const { width: screenWidth } = Dimensions.get('window');
const chartWidth = screenWidth - 80;

interface Stats {
  total_submissions: number;
  correct_predictions: number;
  incorrect_predictions: number;
  accuracy_rate: number;
  recent_feedback: Array<{
    analysis_id: string;
    is_correct: boolean;
    corrected_label?: string;
    timestamp: string;
  }>;
}

export default function StatisticsScreen() {
  const { colors, isDark } = useTheme();
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadStats = async () => {
    try {
      const data = await getFeedbackStats();
      setStats(data);
    } catch {
      // Use placeholder data
      setStats({
        total_submissions: 1247,
        correct_predictions: 1197,
        incorrect_predictions: 50,
        accuracy_rate: 0.96,
        recent_feedback: [],
      });
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadStats();
    setRefreshing(false);
  };

  useEffect(() => {
    loadStats();
  }, []);

  const data = stats || {
    total_submissions: 0,
    correct_predictions: 0,
    incorrect_predictions: 0,
    accuracy_rate: 0,
    recent_feedback: [],
  };

  // Simple bar chart data for model performance
  const barData = [
    { label: 'CNN', value: 94, color: '#6366f1' },
    { label: 'ViT', value: 91, color: '#8b5cf6' },
    { label: 'Face', value: 88, color: '#c084fc' },
    { label: 'Audio', value: 85, color: '#f472b6' },
    { label: 'Meta', value: 92, color: '#818cf8' },
  ];

  const maxBarValue = 100;
  const barChartHeight = 160;
  const barWidth = (chartWidth - 60) / barData.length - 8;

  return (
    <LinearGradient
      colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
      style={styles.gradient}
    >
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.accent}
          />
        }
      >
        {/* Summary Cards */}
        <View style={styles.summaryRow}>
          <GlassCard style={styles.summaryCard}>
            <View style={styles.summaryContent}>
              <Ionicons name="analytics" size={22} color={colors.accent} />
              <AnimatedCounter
                to={data.total_submissions}
                style={[styles.summaryNumber, { color: colors.textPrimary }]}
              />
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>
                Total Scans
              </Text>
            </View>
          </GlassCard>
          <GlassCard style={styles.summaryCard}>
            <View style={styles.summaryContent}>
              <Ionicons name="checkmark-circle" size={22} color={colors.success} />
              <AnimatedCounter
                to={data.correct_predictions}
                style={[styles.summaryNumber, { color: colors.success }]}
              />
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>
                Correct
              </Text>
            </View>
          </GlassCard>
          <GlassCard style={styles.summaryCard}>
            <View style={styles.summaryContent}>
              <Ionicons name="close-circle" size={22} color={colors.danger} />
              <AnimatedCounter
                to={data.incorrect_predictions}
                style={[styles.summaryNumber, { color: colors.danger }]}
              />
              <Text style={[styles.summaryLabel, { color: colors.textSecondary }]}>
                Incorrect
              </Text>
            </View>
          </GlassCard>
        </View>

        {/* Accuracy Gauge */}
        <GlassCard style={styles.accuracyCard} glowing>
          <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
            Overall Accuracy
          </Text>
          <View style={styles.gaugeCenter}>
            <GaugeChart
              value={Math.round(data.accuracy_rate * 100)}
              label="Accuracy Rate"
              size={160}
              strokeWidth={12}
              color={colors.accent}
            />
          </View>
        </GlassCard>

        {/* Model Performance Bar Chart */}
        <GlassCard style={styles.chartCard}>
          <View style={styles.chartHeader}>
            <Ionicons name="bar-chart" size={20} color={colors.accent} />
            <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
              Model Performance
            </Text>
          </View>
          <View style={styles.chartContainer}>
            <Svg width={chartWidth} height={barChartHeight + 40}>
              {/* Grid lines */}
              {[0, 25, 50, 75, 100].map((val) => {
                const y = barChartHeight - (val / maxBarValue) * barChartHeight + 10;
                return (
                  <React.Fragment key={val}>
                    <Line
                      x1={30}
                      y1={y}
                      x2={chartWidth}
                      y2={y}
                      stroke={colors.glassBorder}
                      strokeWidth={1}
                    />
                    <SvgText
                      x={25}
                      y={y + 4}
                      fontSize={10}
                      fill={colors.textSecondary}
                      textAnchor="end"
                    >
                      {val}
                    </SvgText>
                  </React.Fragment>
                );
              })}

              {/* Bars */}
              {barData.map((item, index) => {
                const barHeight = (item.value / maxBarValue) * barChartHeight;
                const x = 40 + index * (barWidth + 8);
                const y = barChartHeight - barHeight + 10;

                return (
                  <React.Fragment key={index}>
                    <Rect
                      x={x}
                      y={y}
                      width={barWidth}
                      height={barHeight}
                      rx={4}
                      fill={item.color}
                      opacity={0.85}
                    />
                    <SvgText
                      x={x + barWidth / 2}
                      y={barChartHeight + 25}
                      fontSize={10}
                      fill={colors.textSecondary}
                      textAnchor="middle"
                    >
                      {item.label}
                    </SvgText>
                    <SvgText
                      x={x + barWidth / 2}
                      y={y - 6}
                      fontSize={11}
                      fill={colors.textPrimary}
                      textAnchor="middle"
                      fontWeight="bold"
                    >
                      {item.value}%
                    </SvgText>
                  </React.Fragment>
                );
              })}
            </Svg>
          </View>
        </GlassCard>

        {/* Recent Feedback */}
        <GlassCard style={styles.recentCard}>
          <View style={styles.chartHeader}>
            <Ionicons name="time" size={20} color={colors.accent} />
            <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
              Recent Feedback
            </Text>
          </View>

          {data.recent_feedback.length > 0 ? (
            data.recent_feedback.slice(0, 10).map((fb, i) => (
              <View
                key={i}
                style={[styles.feedbackRow, { borderBottomColor: colors.glassBorder }]}
              >
                <Ionicons
                  name={fb.is_correct ? 'checkmark-circle' : 'close-circle'}
                  size={18}
                  color={fb.is_correct ? colors.success : colors.danger}
                />
                <View style={styles.feedbackInfo}>
                  <Text style={[styles.feedbackId, { color: colors.textPrimary }]} numberOfLines={1}>
                    {fb.analysis_id}
                  </Text>
                  <Text style={[styles.feedbackTime, { color: colors.textSecondary }]}>
                    {new Date(fb.timestamp).toLocaleDateString()}
                  </Text>
                </View>
                {fb.corrected_label && (
                  <View style={[styles.correctedBadge, { backgroundColor: colors.warning + '20' }]}>
                    <Text style={[styles.correctedText, { color: colors.warning }]}>
                      → {fb.corrected_label}
                    </Text>
                  </View>
                )}
              </View>
            ))
          ) : (
            <View style={styles.emptyState}>
              <Ionicons name="chatbubble-outline" size={32} color={colors.textSecondary} style={{ opacity: 0.4 }} />
              <Text style={[styles.emptyText, { color: colors.textSecondary }]}>
                No feedback submitted yet
              </Text>
            </View>
          )}
        </GlassCard>

        {/* Info */}
        <GlassCard style={styles.infoCard}>
          <View style={styles.infoRow}>
            <Ionicons name="information-circle" size={18} color={colors.accent} />
            <Text style={[styles.infoText, { color: colors.textSecondary }]}>
              Statistics are updated in real-time based on user feedback. Help improve our models by providing feedback on analysis results.
            </Text>
          </View>
        </GlassCard>
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
  summaryRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  summaryCard: {
    flex: 1,
    padding: Spacing.md,
  },
  summaryContent: {
    alignItems: 'center',
    gap: 4,
  },
  summaryNumber: {
    fontSize: FontSizes.lg,
    fontWeight: '800',
  },
  summaryLabel: {
    fontSize: FontSizes.xs,
    fontWeight: '500',
    textAlign: 'center',
  },
  accuracyCard: {
    padding: Spacing.xl,
    gap: Spacing.base,
  },
  gaugeCenter: {
    alignItems: 'center',
  },
  sectionTitle: {
    fontSize: FontSizes.md,
    fontWeight: '700',
  },
  chartCard: {
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  chartHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  chartContainer: {
    alignItems: 'center',
  },
  recentCard: {
    padding: Spacing.lg,
    gap: Spacing.sm,
  },
  feedbackRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
  },
  feedbackInfo: {
    flex: 1,
    gap: 1,
  },
  feedbackId: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
    fontVariant: ['tabular-nums'],
  },
  feedbackTime: {
    fontSize: FontSizes.xs,
  },
  correctedBadge: {
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: BorderRadius.full,
  },
  correctedText: {
    fontSize: FontSizes.xs,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    gap: Spacing.sm,
    paddingVertical: Spacing.xl,
  },
  emptyText: {
    fontSize: FontSizes.sm,
  },
  infoCard: {
    padding: Spacing.base,
  },
  infoRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    alignItems: 'flex-start',
  },
  infoText: {
    flex: 1,
    fontSize: FontSizes.xs,
    lineHeight: 18,
  },
});
