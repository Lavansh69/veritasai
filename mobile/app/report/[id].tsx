/**
 * VeritasAI — Report Screen
 * Downloads and shares the analysis PDF report.
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams } from 'expo-router';
import * as Sharing from 'expo-sharing';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../../theme/typography';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { GradientText } from '../../components/ui/GradientText';
import { LoadingSpinner } from '../../components/ui/LoadingSpinner';
import { downloadReport } from '../../services/api';

export default function ReportScreen() {
  const { colors, isDark } = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const [downloading, setDownloading] = useState(false);
  const [downloadedUri, setDownloadedUri] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    if (!id) return;

    setDownloading(true);
    setError(null);

    try {
      const uri = await downloadReport(id);
      setDownloadedUri(uri);
    } catch (err: any) {
      setError(err.message || 'Failed to download report');
    } finally {
      setDownloading(false);
    }
  };

  const handleShare = async () => {
    if (!downloadedUri) return;

    const isAvailable = await Sharing.isAvailableAsync();
    if (!isAvailable) {
      Alert.alert('Sharing not available', 'Sharing is not supported on this device.');
      return;
    }

    try {
      await Sharing.shareAsync(downloadedUri, {
        mimeType: 'application/pdf',
        dialogTitle: 'Share VeritasAI Report',
      });
    } catch (err: any) {
      Alert.alert('Error', 'Failed to share report');
    }
  };

  if (downloading) {
    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <LoadingSpinner message="Generating PDF report..." />
      </LinearGradient>
    );
  }

  return (
    <LinearGradient
      colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
      style={styles.gradient}
    >
      <View style={styles.container}>
        <GlassCard style={styles.card} glowing={!!downloadedUri}>
          <View style={styles.content}>
            {/* Icon */}
            <View style={[styles.iconBg, { backgroundColor: colors.accentGlow }]}>
              <Ionicons
                name={downloadedUri ? 'document-text' : 'document-text-outline'}
                size={48}
                color={colors.accent}
              />
            </View>

            {/* Title */}
            <GradientText style={styles.title}>Analysis Report</GradientText>

            <Text style={[styles.analysisId, { color: colors.textSecondary }]}>
              ID: {id || 'N/A'}
            </Text>

            {/* Status */}
            {downloadedUri ? (
              <>
                <View style={[styles.successBadge, { backgroundColor: colors.success + '20' }]}>
                  <Ionicons name="checkmark-circle" size={20} color={colors.success} />
                  <Text style={[styles.successText, { color: colors.success }]}>
                    Report Downloaded
                  </Text>
                </View>

                <Text style={[styles.description, { color: colors.textSecondary }]}>
                  Your comprehensive analysis report is ready. Share it or open it from your device's file manager.
                </Text>

                <Button
                  title="Share Report"
                  onPress={handleShare}
                  size="lg"
                  icon={<Ionicons name="share" size={20} color="#fff" />}
                  style={styles.btn}
                />

                <Button
                  title="Download Again"
                  variant="outline"
                  onPress={handleDownload}
                  icon={<Ionicons name="refresh" size={18} color={colors.textPrimary} />}
                  style={styles.btn}
                />
              </>
            ) : (
              <>
                <Text style={[styles.description, { color: colors.textSecondary }]}>
                  Generate a comprehensive PDF report containing the full analysis results, scores, heatmaps, and metadata findings.
                </Text>

                {error && (
                  <View style={[styles.errorBadge, { backgroundColor: colors.danger + '20' }]}>
                    <Ionicons name="alert-circle" size={16} color={colors.danger} />
                    <Text style={[styles.errorText, { color: colors.danger }]}>{error}</Text>
                  </View>
                )}

                <Button
                  title="Download PDF Report"
                  onPress={handleDownload}
                  size="lg"
                  icon={<Ionicons name="download" size={20} color="#fff" />}
                  style={styles.btn}
                />
              </>
            )}
          </View>
        </GlassCard>

        {/* Report Contents Info */}
        <GlassCard style={styles.infoCard}>
          <Text style={[styles.infoTitle, { color: colors.textPrimary }]}>Report Includes</Text>
          {[
            { icon: 'shield-checkmark', label: 'Overall verdict and confidence score' },
            { icon: 'bar-chart', label: 'Individual analysis scores breakdown' },
            { icon: 'eye', label: 'Grad-CAM heatmap visualization' },
            { icon: 'person', label: 'Face consistency analysis' },
            { icon: 'document-text', label: 'Metadata forensic findings' },
          ].map((item, i) => (
            <View key={i} style={styles.infoRow}>
              <Ionicons name={item.icon as any} size={16} color={colors.accent} />
              <Text style={[styles.infoLabel, { color: colors.textSecondary }]}>
                {item.label}
              </Text>
            </View>
          ))}
        </GlassCard>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  container: {
    flex: 1,
    padding: Spacing.lg,
    gap: Spacing.base,
    justifyContent: 'center',
  },
  card: {
    padding: Spacing.xl,
  },
  content: {
    alignItems: 'center',
    gap: Spacing.md,
  },
  iconBg: {
    width: 88,
    height: 88,
    borderRadius: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: FontSizes.xl,
  },
  analysisId: {
    fontSize: FontSizes.xs,
    fontWeight: '500',
    fontVariant: ['tabular-nums'],
  },
  description: {
    fontSize: FontSizes.sm,
    textAlign: 'center',
    lineHeight: 22,
  },
  successBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  successText: {
    fontSize: FontSizes.sm,
    fontWeight: '600',
  },
  errorBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.md,
  },
  errorText: {
    fontSize: FontSizes.xs,
    fontWeight: '500',
    flex: 1,
  },
  btn: {
    width: '100%',
  },
  infoCard: {
    padding: Spacing.base,
    gap: Spacing.sm,
  },
  infoTitle: {
    fontSize: FontSizes.base,
    fontWeight: '700',
    marginBottom: Spacing.xs,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  infoLabel: {
    fontSize: FontSizes.sm,
  },
});
