/**
 * VeritasAI — GlassCard Component
 * Glassmorphic card using expo-blur for the frosted glass effect.
 */

import React from 'react';
import { StyleSheet, View, ViewStyle } from 'react-native';
import { BlurView } from 'expo-blur';
import { useTheme } from '../../theme/ThemeContext';
import { BorderRadius } from '../../theme/typography';

interface GlassCardProps {
  children: React.ReactNode;
  style?: ViewStyle;
  intensity?: number;
  glowing?: boolean;
  onPress?: () => void;
}

export function GlassCard({ children, style, intensity = 20, glowing = false }: GlassCardProps) {
  const { colors, isDark } = useTheme();

  return (
    <View
      style={[
        styles.container,
        {
          borderColor: glowing ? colors.accent : colors.glassBorder,
          shadowColor: glowing ? colors.accent : colors.glassShadow,
          backgroundColor: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.6)',
        },
        glowing && {
          shadowOpacity: 0.6,
          shadowRadius: 30,
          elevation: 12,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderWidth: 1,
    borderRadius: BorderRadius.lg,
    overflow: 'hidden',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.25,
    shadowRadius: 16,
    elevation: 6,
  },
});
