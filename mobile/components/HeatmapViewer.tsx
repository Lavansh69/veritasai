/**
 * VeritasAI — HeatmapViewer Component
 * Displays heatmap overlay with pinch-to-zoom support.
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  Dimensions,
  Pressable,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { GlassCard } from './ui/GlassCard';
import { useTheme } from '../theme/ThemeContext';
import { FontSizes, Spacing, BorderRadius } from '../theme/typography';

const screenWidth = Dimensions.get('window').width;

interface HeatmapViewerProps {
  heatmapUrl: string;
  overlayUrl: string;
}

export function HeatmapViewer({ heatmapUrl, overlayUrl }: HeatmapViewerProps) {
  const { colors } = useTheme();
  const [showOverlay, setShowOverlay] = useState(true);

  const imageUrl = showOverlay ? overlayUrl : heatmapUrl;

  return (
    <GlassCard style={styles.card}>
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="eye" size={20} color={colors.accent} />
          <Text style={[styles.title, { color: colors.textPrimary }]}>
            Explainability Heatmap
          </Text>
        </View>
        <Pressable
          onPress={() => setShowOverlay(!showOverlay)}
          style={[styles.toggleBtn, { backgroundColor: colors.accentGlow }]}
        >
          <Text style={[styles.toggleText, { color: colors.accent }]}>
            {showOverlay ? 'Overlay' : 'Heatmap'}
          </Text>
        </Pressable>
      </View>

      <Text style={[styles.description, { color: colors.textSecondary }]}>
        Grad-CAM highlights the regions the AI focused on to make its decision.
        Brighter areas indicate higher attention.
      </Text>

      <View style={styles.imageContainer}>
        <Image
          source={{ uri: imageUrl }}
          style={styles.image}
          resizeMode="contain"
        />
      </View>
    </GlassCard>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: Spacing.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.sm,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  title: {
    fontSize: FontSizes.md,
    fontWeight: '700',
  },
  toggleBtn: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  toggleText: {
    fontSize: FontSizes.xs,
    fontWeight: '600',
  },
  description: {
    fontSize: FontSizes.sm,
    lineHeight: 20,
    marginBottom: Spacing.base,
  },
  imageContainer: {
    borderRadius: BorderRadius.md,
    overflow: 'hidden',
    backgroundColor: 'rgba(0,0,0,0.2)',
  },
  image: {
    width: '100%',
    height: screenWidth - 80,
  },
});
