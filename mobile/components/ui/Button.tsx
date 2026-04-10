/**
 * VeritasAI — Button Components
 * Primary and outline button variants with press animations.
 */

import React from 'react';
import {
  StyleSheet,
  Text,
  Pressable,
  ViewStyle,
  TextStyle,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { BorderRadius, Spacing, FontSizes } from '../../theme/typography';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'outline' | 'danger' | 'success';
  icon?: React.ReactNode;
  iconRight?: React.ReactNode;
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
  size?: 'sm' | 'md' | 'lg';
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  icon,
  iconRight,
  disabled = false,
  loading = false,
  style,
  textStyle,
  size = 'md',
}: ButtonProps) {
  const { colors } = useTheme();

  const sizeStyles = {
    sm: { paddingVertical: 8, paddingHorizontal: 16, fontSize: FontSizes.sm },
    md: { paddingVertical: 14, paddingHorizontal: 24, fontSize: FontSizes.base },
    lg: { paddingVertical: 18, paddingHorizontal: 32, fontSize: FontSizes.md },
  };

  if (variant === 'primary') {
    return (
      <Pressable
        onPress={onPress}
        disabled={disabled || loading}
        style={({ pressed }) => [
          { opacity: pressed ? 0.85 : disabled ? 0.5 : 1, transform: [{ scale: pressed ? 0.97 : 1 }] },
          style,
        ]}
      >
        <LinearGradient
          colors={Gradients.primary as unknown as string[]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={[
            styles.button,
            {
              paddingVertical: sizeStyles[size].paddingVertical,
              paddingHorizontal: sizeStyles[size].paddingHorizontal,
              shadowColor: '#6366f1',
              shadowOffset: { width: 0, height: 4 },
              shadowOpacity: 0.4,
              shadowRadius: 15,
              elevation: 8,
            },
          ]}
        >
          {loading ? (
            <ActivityIndicator color="#fff" size="small" />
          ) : (
            <>
              {icon}
              <Text style={[styles.primaryText, { fontSize: sizeStyles[size].fontSize }, textStyle]}>
                {title}
              </Text>
              {iconRight}
            </>
          )}
        </LinearGradient>
      </Pressable>
    );
  }

  const borderColorMap = {
    outline: colors.glassBorder,
    danger: colors.danger,
    success: colors.success,
  };

  const textColorMap = {
    outline: colors.textPrimary,
    danger: colors.danger,
    success: colors.success,
  };

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled || loading}
      style={({ pressed }) => [
        styles.button,
        {
          borderWidth: 1,
          borderColor: borderColorMap[variant],
          paddingVertical: sizeStyles[size].paddingVertical,
          paddingHorizontal: sizeStyles[size].paddingHorizontal,
          opacity: pressed ? 0.7 : disabled ? 0.5 : 1,
          transform: [{ scale: pressed ? 0.97 : 1 }],
          backgroundColor: pressed ? colors.accentGlow : 'transparent',
        },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={textColorMap[variant]} size="small" />
      ) : (
        <>
          {icon}
          <Text
            style={[
              styles.outlineText,
              { color: textColorMap[variant], fontSize: sizeStyles[size].fontSize },
              textStyle,
            ]}
          >
            {title}
          </Text>
          {iconRight}
        </>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: BorderRadius.md,
    gap: 8,
  },
  primaryText: {
    color: '#ffffff',
    fontWeight: '700',
  },
  outlineText: {
    fontWeight: '600',
  },
});
