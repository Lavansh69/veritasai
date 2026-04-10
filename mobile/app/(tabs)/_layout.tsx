/**
 * VeritasAI — Tab Layout
 * Bottom tab navigation with glassmorphic styling.
 */

import React from 'react';
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../theme/ThemeContext';
import { FontSizes } from '../../theme/typography';

type TabIconName = React.ComponentProps<typeof Ionicons>['name'];

interface TabConfig {
  name: string;
  title: string;
  icon: TabIconName;
  iconFocused: TabIconName;
}

const tabs: TabConfig[] = [
  { name: 'index', title: 'Home', icon: 'home-outline', iconFocused: 'home' },
  { name: 'upload', title: 'Detect', icon: 'scan-outline', iconFocused: 'scan' },
  { name: 'audio', title: 'Audio', icon: 'mic-outline', iconFocused: 'mic' },
  { name: 'live', title: 'Live', icon: 'videocam-outline', iconFocused: 'videocam' },
  { name: 'settings', title: 'Settings', icon: 'settings-outline', iconFocused: 'settings' },
];

export default function TabLayout() {
  const { colors, isDark } = useTheme();

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: colors.tabBarBg,
          borderTopColor: colors.glassBorder,
          borderTopWidth: 1,
          height: 88,
          paddingBottom: 28,
          paddingTop: 8,
          elevation: 0,
          shadowColor: isDark ? '#6366f1' : '#000',
          shadowOffset: { width: 0, height: -4 },
          shadowOpacity: isDark ? 0.15 : 0.05,
          shadowRadius: 20,
        },
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors.textSecondary,
        tabBarLabelStyle: {
          fontSize: FontSizes.xs,
          fontWeight: '600',
        },
      }}
    >
      {tabs.map((tab) => (
        <Tabs.Screen
          key={tab.name}
          name={tab.name}
          options={{
            title: tab.title,
            tabBarIcon: ({ color, focused }) => (
              <Ionicons
                name={focused ? tab.iconFocused : tab.icon}
                size={24}
                color={color}
              />
            ),
          }}
        />
      ))}
    </Tabs>
  );
}
