/**
 * VeritasAI — AnimatedCounter Component
 * Counts up from 0 to a target value with animation.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Text, Animated, TextStyle } from 'react-native';

interface AnimatedCounterProps {
  to: number;
  suffix?: string;
  duration?: number;
  style?: TextStyle;
}

export function AnimatedCounter({
  to,
  suffix = '',
  duration = 1500,
  style,
}: AnimatedCounterProps) {
  const [displayValue, setDisplayValue] = useState(0);
  const animValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const listener = animValue.addListener(({ value }) => {
      setDisplayValue(Math.round(value));
    });

    Animated.timing(animValue, {
      toValue: to,
      duration,
      useNativeDriver: false,
    }).start();

    return () => {
      animValue.removeListener(listener);
    };
  }, [to, duration, animValue]);

  return (
    <Text style={style}>
      {displayValue}
      {suffix}
    </Text>
  );
}
