/**
 * VeritasAI — Settings Screen
 * API URL configuration, theme toggle, and app info.
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Alert,
  Switch,
  Linking,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../../theme/typography';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { GradientText } from '../../components/ui/GradientText';
import { getApiUrl, setApiUrl, healthCheck } from '../../services/api';

export default function SettingsScreen() {
  const { colors, isDark, toggle } = useTheme();
  const [apiUrl, setApiUrlState] = useState('');
  const [testing, setTesting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error'>('idle');

  useEffect(() => {
    const loadUrl = async () => {
      const url = await getApiUrl();
      setApiUrlState(url);
    };
    loadUrl();
  }, []);

  const saveUrl = async () => {
    if (!apiUrl.trim()) {
      Alert.alert('Error', 'API URL cannot be empty');
      return;
    }
    await setApiUrl(apiUrl.trim());
    Alert.alert('Saved', 'API URL updated successfully');
  };

  const testConnection = async () => {
    setTesting(true);
    setConnectionStatus('idle');
    try {
      await setApiUrl(apiUrl.trim());
      await healthCheck();
      setConnectionStatus('success');
    } catch {
      setConnectionStatus('error');
    } finally {
      setTesting(false);
    }
  };

  return (
    <LinearGradient
      colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
      style={styles.gradient}
    >
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Ionicons name="settings" size={28} color={colors.accent} />
          <Text style={[styles.title, { color: colors.textPrimary }]}>Settings</Text>
        </View>

        {/* API Configuration */}
        <GlassCard style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="server" size={20} color={colors.accent} />
            <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
              Server Configuration
            </Text>
          </View>

          <Text style={[styles.label, { color: colors.textSecondary }]}>Backend API URL</Text>
          <TextInput
            style={[
              styles.input,
              {
                color: colors.textPrimary,
                backgroundColor: colors.inputBg,
                borderColor: colors.glassBorder,
              },
            ]}
            value={apiUrl}
            onChangeText={setApiUrlState}
            placeholder="http://192.168.1.100:8000"
            placeholderTextColor={colors.textSecondary}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
          />

          {connectionStatus !== 'idle' && (
            <View style={styles.statusRow}>
              <Ionicons
                name={connectionStatus === 'success' ? 'checkmark-circle' : 'close-circle'}
                size={18}
                color={connectionStatus === 'success' ? colors.success : colors.danger}
              />
              <Text
                style={[
                  styles.statusText,
                  { color: connectionStatus === 'success' ? colors.success : colors.danger },
                ]}
              >
                {connectionStatus === 'success' ? 'Connected successfully!' : 'Connection failed'}
              </Text>
            </View>
          )}

          <View style={styles.buttonRow}>
            <Button
              title="Test"
              variant="outline"
              size="sm"
              onPress={testConnection}
              loading={testing}
              icon={<Ionicons name="pulse" size={16} color={colors.textPrimary} />}
              style={styles.actionBtn}
            />
            <Button
              title="Save"
              size="sm"
              onPress={saveUrl}
              icon={<Ionicons name="save" size={16} color="#fff" />}
              style={styles.actionBtn}
            />
          </View>
        </GlassCard>

        {/* Appearance */}
        <GlassCard style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="color-palette" size={20} color={colors.accent} />
            <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
              Appearance
            </Text>
          </View>

          <View style={styles.settingRow}>
            <View style={styles.settingInfo}>
              <Text style={[styles.settingLabel, { color: colors.textPrimary }]}>Dark Mode</Text>
              <Text style={[styles.settingHint, { color: colors.textSecondary }]}>
                Toggle between dark and light theme
              </Text>
            </View>
            <Switch
              value={isDark}
              onValueChange={toggle}
              trackColor={{ false: '#767577', true: colors.accent }}
              thumbColor={isDark ? '#f4f3f4' : '#f4f3f4'}
            />
          </View>
        </GlassCard>

        {/* About */}
        <GlassCard style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="information-circle" size={20} color={colors.accent} />
            <Text style={[styles.sectionTitle, { color: colors.textPrimary }]}>
              About
            </Text>
          </View>

          <View style={styles.aboutInfo}>
            <GradientText style={styles.aboutTitle}>VeritasAI</GradientText>
            <Text style={[styles.aboutVersion, { color: colors.textSecondary }]}>
              Version 1.0.0
            </Text>
            <Text style={[styles.aboutDesc, { color: colors.textSecondary }]}>
              AI-Powered Multi-Modal Deepfake Detection Platform. Protecting truth in the age of synthetic media.
            </Text>
          </View>

          <View style={styles.techStack}>
            {[
              { icon: 'hardware-chip', label: 'EfficientNet-B4 + ViT' },
              { icon: 'musical-notes', label: 'WaveFake Audio Classifier' },
              { icon: 'eye', label: 'Grad-CAM Explainability' },
              { icon: 'analytics', label: 'Face Consistency Analysis' },
              { icon: 'document-text', label: 'Metadata Forensics' },
            ].map((item, i) => (
              <View key={i} style={styles.techRow}>
                <Ionicons name={item.icon as any} size={16} color={colors.accent} />
                <Text style={[styles.techLabel, { color: colors.textSecondary }]}>
                  {item.label}
                </Text>
              </View>
            ))}
          </View>
        </GlassCard>

        {/* Footer */}
        <Text style={[styles.footer, { color: colors.textSecondary }]}>
          Built with 🛡️ for truth verification
        </Text>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  container: { flex: 1 },
  content: {
    padding: Spacing.lg,
    paddingTop: 60,
    paddingBottom: 40,
    gap: Spacing.base,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
    marginBottom: Spacing.sm,
  },
  title: {
    fontSize: FontSizes['2xl'],
    fontWeight: '800',
  },
  section: {
    padding: Spacing.lg,
    gap: Spacing.md,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  sectionTitle: {
    fontSize: FontSizes.md,
    fontWeight: '700',
  },
  label: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
    marginTop: Spacing.xs,
  },
  input: {
    borderWidth: 1,
    borderRadius: BorderRadius.md,
    paddingHorizontal: Spacing.base,
    paddingVertical: Spacing.md,
    fontSize: FontSizes.base,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.xs,
  },
  statusText: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  actionBtn: {
    flex: 1,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  settingInfo: {
    flex: 1,
    gap: 2,
  },
  settingLabel: {
    fontSize: FontSizes.base,
    fontWeight: '600',
  },
  settingHint: {
    fontSize: FontSizes.xs,
  },
  aboutInfo: {
    alignItems: 'center',
    gap: Spacing.xs,
  },
  aboutTitle: {
    fontSize: FontSizes['2xl'],
  },
  aboutVersion: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  aboutDesc: {
    fontSize: FontSizes.sm,
    textAlign: 'center',
    lineHeight: 20,
  },
  techStack: {
    gap: Spacing.sm,
    marginTop: Spacing.xs,
  },
  techRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  techLabel: {
    fontSize: FontSizes.sm,
  },
  footer: {
    fontSize: FontSizes.xs,
    textAlign: 'center',
    marginTop: Spacing.lg,
    opacity: 0.6,
  },
});
