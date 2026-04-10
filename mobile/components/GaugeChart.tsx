/**
 * VeritasAI — GaugeChart Component
 * SVG semicircle gauge for displaying scores.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';
import { useTheme } from '../theme/ThemeContext';
import { FontSizes, Spacing } from '../theme/typography';

interface GaugeChartProps {
  value: number; // 0-100
  label: string;
  size?: number;
  strokeWidth?: number;
  color?: string;
}

export function GaugeChart({
  value,
  label,
  size = 100,
  strokeWidth = 8,
  color,
}: GaugeChartProps) {
  const { colors } = useTheme();
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedValue = Math.min(100, Math.max(0, value));
  const strokeDashoffset = circumference - (clampedValue / 100) * circumference;

  const getColor = () => {
    if (color) return color;
    if (clampedValue >= 70) return colors.success;
    if (clampedValue >= 40) return colors.warning;
    return colors.danger;
  };

  return (
    <View style={styles.container}>
      <View style={{ width: size, height: size }}>
        <Svg width={size} height={size}>
          <G rotation="-90" origin={`${size / 2}, ${size / 2}`}>
            {/* Background circle */}
            <Circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={colors.glassBorder}
              strokeWidth={strokeWidth}
              fill="transparent"
            />
            {/* Progress circle */}
            <Circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={getColor()}
              strokeWidth={strokeWidth}
              fill="transparent"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
            />
          </G>
        </Svg>
        {/* Center value */}
        <View style={[styles.valueContainer, { width: size, height: size }]}>
          <Text style={[styles.value, { color: getColor(), fontSize: size * 0.22 }]}>
            {Math.round(clampedValue)}
          </Text>
        </View>
      </View>
      <Text style={[styles.label, { color: colors.textSecondary }]} numberOfLines={1}>
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    gap: Spacing.xs,
  },
  valueContainer: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
  },
  value: {
    fontWeight: '700',
  },
  label: {
    fontSize: FontSizes.xs,
    fontWeight: '500',
    textAlign: 'center',
  },
});
