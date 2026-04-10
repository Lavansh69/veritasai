/**
 * VeritasAI — Root Layout
 * Wraps the app with ThemeProvider and sets up the navigation stack.
 */

import React from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ThemeProvider, useTheme } from '../theme/ThemeContext';

function RootNavigator() {
  const { colors, isDark } = useTheme();

  return (
    <>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: colors.bgGradientFrom },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="(tabs)" />
        <Stack.Screen
          name="analysis/[id]"
          options={{
            headerShown: true,
            headerTitle: 'Analysis Results',
            headerStyle: { backgroundColor: colors.bgGradientFrom },
            headerTintColor: colors.textPrimary,
            headerShadowVisible: false,
            presentation: 'card',
          }}
        />
        <Stack.Screen
          name="report/[id]"
          options={{
            headerShown: true,
            headerTitle: 'Report',
            headerStyle: { backgroundColor: colors.bgGradientFrom },
            headerTintColor: colors.textPrimary,
            headerShadowVisible: false,
            presentation: 'modal',
          }}
        />
        <Stack.Screen
          name="statistics"
          options={{
            headerShown: true,
            headerTitle: 'Statistics',
            headerStyle: { backgroundColor: colors.bgGradientFrom },
            headerTintColor: colors.textPrimary,
            headerShadowVisible: false,
          }}
        />
      </Stack>
    </>
  );
}

export default function RootLayout() {
  return (
    <ThemeProvider>
      <RootNavigator />
    </ThemeProvider>
  );
}
