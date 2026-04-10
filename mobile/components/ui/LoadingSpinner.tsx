/**
 * VeritasAI — LoadingSpinner Component
 * Animated loading indicator with pulsing text.
 */

import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated, ActivityIndicator } from 'react-native';
import { useTheme } from '../../theme/ThemeContext';
import { FontSizes, Spacing } from '../../theme/typography';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'small' | 'large';
}

export function LoadingSpinner({
  message = 'Analyzing...',
  size = 'large',
}: LoadingSpinnerProps) {
  const { colors } = useTheme();
  const pulseAnim = useRef(new Animated.Value(0.6)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 0.6,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, [pulseAnim]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size={size} color={colors.accent} />
      {message && (
        <Animated.Text
          style={[
            styles.message,
            { color: colors.textSecondary, opacity: pulseAnim },
          ]}
        >
          {message}
        </Animated.Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing['2xl'],
    gap: Spacing.base,
  },
  message: {
    fontSize: FontSizes.base,
    fontWeight: '500',
    marginTop: Spacing.md,
  },
});
