/**
 * VeritasAI — Theme Context
 * Provides dark/light mode toggle and color access throughout the app.
 */

import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { useColorScheme } from 'react-native';
import { Colors, type ThemeColors } from './colors';

type ThemeMode = 'dark' | 'light';

interface ThemeContextType {
  mode: ThemeMode;
  colors: ThemeColors;
  toggle: () => void;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextType>({
  mode: 'dark',
  colors: Colors.dark,
  toggle: () => {},
  isDark: true,
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemScheme = useColorScheme();
  const [mode, setMode] = useState<ThemeMode>(systemScheme === 'light' ? 'light' : 'dark');

  const toggle = useCallback(() => {
    setMode((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const value = useMemo(
    () => ({
      mode,
      colors: Colors[mode],
      toggle,
      isDark: mode === 'dark',
    }),
    [mode, toggle]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  return useContext(ThemeContext);
}

export default ThemeContext;
