/**
 * VeritasAI — GradientText Component
 * Renders text with a gradient fill using MaskedView.
 */

import React from 'react';
import { Text, StyleSheet, TextStyle, View } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Gradients } from '../../theme/colors';

interface GradientTextProps {
  children: string;
  style?: TextStyle;
  colors?: readonly string[];
}

export function GradientText({
  children,
  style,
  colors = Gradients.accent,
}: GradientTextProps) {
  // On Android, MaskedView can be problematic, so we use a simpler approach
  // We'll use the accent color directly with a slight gradient effect via text shadow
  return (
    <Text
      style={[
        styles.text,
        style,
        {
          color: colors[0] as string,
          textShadowColor: (colors[2] || colors[1]) as string,
          textShadowOffset: { width: 0, height: 0 },
          textShadowRadius: 10,
        },
      ]}
    >
      {children}
    </Text>
  );
}

const styles = StyleSheet.create({
  text: {
    fontWeight: '800',
  },
});
