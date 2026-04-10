/**
 * VeritasAI — Audio Analysis Screen
 * Record audio or pick audio file for deepfake voice detection.
 */

import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  Pressable,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../../theme/typography';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { GradientText } from '../../components/ui/GradientText';
import { LoadingSpinner } from '../../components/ui/LoadingSpinner';
import { GaugeChart } from '../../components/GaugeChart';
import { FeedbackPanel } from '../../components/FeedbackPanel';
import { analyzeAudio, type AudioResult } from '../../services/api';

export default function AudioScreen() {
  const { colors, isDark } = useTheme();
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [selectedAudio, setSelectedAudio] = useState<{ uri: string; name: string } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AudioResult | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const durationInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = async () => {
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Permission Required', 'Microphone access is needed for audio recording.');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording: rec } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(rec);
      setIsRecording(true);
      setRecordingDuration(0);

      // Pulse animation
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.15, duration: 600, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
        ])
      ).start();

      // Duration counter
      durationInterval.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      Alert.alert('Error', 'Failed to start recording');
    }
  };

  const stopRecording = async () => {
    if (!recording) return;

    try {
      await recording.stopAndUnloadAsync();
      const uri = recording.getURI();
      pulseAnim.stopAnimation();
      pulseAnim.setValue(1);

      if (durationInterval.current) {
        clearInterval(durationInterval.current);
        durationInterval.current = null;
      }

      setIsRecording(false);
      setRecording(null);

      if (uri) {
        setSelectedAudio({
          uri,
          name: `recording_${Date.now()}.m4a`,
        });
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to stop recording');
    }
  };

  const pickAudioFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['audio/*'],
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        setSelectedAudio({
          uri: asset.uri,
          name: asset.name || `audio_${Date.now()}.wav`,
        });
      }
    } catch {
      Alert.alert('Error', 'Failed to pick audio file');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedAudio) return;

    setAnalyzing(true);
    try {
      const res = await analyzeAudio(selectedAudio.uri, selectedAudio.name);
      setResult(res);
    } catch (err: any) {
      Alert.alert('Analysis Failed', err.message || 'Audio analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAll = () => {
    setSelectedAudio(null);
    setResult(null);
    setRecordingDuration(0);
  };

  const formatDuration = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  if (analyzing) {
    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <LoadingSpinner message="Analyzing audio for AI-generated speech patterns..." />
      </LinearGradient>
    );
  }

  // Results view
  if (result) {
    const prob = result.deepfake_probability;
    const verdictColor = prob < 0.3 ? colors.success : prob < 0.7 ? colors.warning : colors.danger;
    const verdictLabel = prob < 0.3 ? 'Likely Authentic' : prob < 0.7 ? 'Suspicious' : 'Likely AI-Generated';
    const verdictIcon = prob < 0.3 ? 'shield-checkmark' : prob < 0.7 ? 'warning' : 'alert-circle';

    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <ScrollView style={styles.container} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <Ionicons name="mic" size={28} color={colors.accent} />
            <Text style={[styles.title, { color: colors.textPrimary }]}>Audio Results</Text>
          </View>

          {/* Main verdict */}
          <GlassCard style={styles.verdictCard} glowing>
            <View style={styles.verdictContent}>
              <View style={[styles.verdictIconBg, { backgroundColor: verdictColor + '20' }]}>
                <Ionicons name={verdictIcon as any} size={40} color={verdictColor} />
              </View>
              <Text style={[styles.verdictLabel, { color: verdictColor }]}>{verdictLabel}</Text>
              <GaugeChart
                value={Math.round((1 - prob) * 100)}
                label="Authenticity Score"
                size={140}
                strokeWidth={10}
              />
              {result.demo_mode && (
                <View style={[styles.demoBadge, { backgroundColor: colors.warning + '20' }]}>
                  <Ionicons name="flask" size={14} color={colors.warning} />
                  <Text style={[styles.demoText, { color: colors.warning }]}>Demo Mode</Text>
                </View>
              )}
            </View>
          </GlassCard>

          {/* Feedback */}
          <FeedbackPanel
            analysisId={result.analysis_id}
            prediction={result.verdict}
            confidence={prob}
          />

          <Button
            title="Analyze Another"
            variant="outline"
            onPress={resetAll}
            icon={<Ionicons name="refresh" size={18} color={colors.textPrimary} />}
          />
        </ScrollView>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient
      colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
      style={styles.gradient}
    >
      <ScrollView style={styles.container} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Ionicons name="mic" size={28} color={colors.accent} />
          <Text style={[styles.title, { color: colors.textPrimary }]}>Audio Detection</Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
            Record or upload audio to detect AI-generated speech
          </Text>
        </View>

        {/* Recording Control */}
        <GlassCard style={styles.recordCard}>
          <View style={styles.recordContent}>
            <Animated.View style={{ transform: [{ scale: isRecording ? pulseAnim : 1 }] }}>
              <Pressable
                onPress={isRecording ? stopRecording : startRecording}
                style={[
                  styles.recordButton,
                  {
                    backgroundColor: isRecording ? colors.danger : colors.accent,
                    shadowColor: isRecording ? colors.danger : colors.accent,
                  },
                ]}
              >
                <Ionicons
                  name={isRecording ? 'stop' : 'mic'}
                  size={36}
                  color="#fff"
                />
              </Pressable>
            </Animated.View>

            <Text style={[styles.recordLabel, { color: colors.textPrimary }]}>
              {isRecording ? 'Recording...' : 'Tap to Record'}
            </Text>

            {(isRecording || recordingDuration > 0) && (
              <Text style={[styles.duration, { color: isRecording ? colors.danger : colors.textSecondary }]}>
                {formatDuration(recordingDuration)}
              </Text>
            )}
          </View>
        </GlassCard>

        {/* Divider */}
        <View style={styles.dividerRow}>
          <View style={[styles.dividerLine, { backgroundColor: colors.glassBorder }]} />
          <Text style={[styles.dividerText, { color: colors.textSecondary }]}>or</Text>
          <View style={[styles.dividerLine, { backgroundColor: colors.glassBorder }]} />
        </View>

        {/* File picker */}
        <Button
          title="Pick Audio File"
          variant="outline"
          onPress={pickAudioFile}
          icon={<Ionicons name="document-attach" size={20} color={colors.textPrimary} />}
          size="lg"
        />

        {/* Selected audio info */}
        {selectedAudio && (
          <GlassCard style={styles.selectedCard} glowing>
            <View style={styles.selectedRow}>
              <View style={[styles.audioIconBg, { backgroundColor: colors.accentGlow }]}>
                <Ionicons name="musical-notes" size={24} color={colors.accent} />
              </View>
              <View style={styles.selectedInfo}>
                <Text style={[styles.selectedName, { color: colors.textPrimary }]} numberOfLines={1}>
                  {selectedAudio.name}
                </Text>
                <Text style={[styles.selectedReady, { color: colors.success }]}>Ready to analyze</Text>
              </View>
              <Pressable onPress={() => setSelectedAudio(null)}>
                <Ionicons name="close-circle" size={24} color={colors.textSecondary} />
              </Pressable>
            </View>
          </GlassCard>
        )}

        {/* Analyze */}
        {selectedAudio && (
          <Button
            title="🎤  Analyze Audio"
            onPress={handleAnalyze}
            size="lg"
            style={styles.analyzeBtn}
          />
        )}

        {/* Info */}
        <GlassCard style={styles.infoCard}>
          <View style={styles.infoRow}>
            <Ionicons name="information-circle" size={20} color={colors.accent} />
            <Text style={[styles.infoText, { color: colors.textSecondary }]}>
              Our audio classifier uses WaveFake-trained neural networks to detect AI-generated speech, voice cloning, and text-to-speech synthesis.
            </Text>
          </View>
        </GlassCard>
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
    alignItems: 'center',
    gap: Spacing.xs,
    marginBottom: Spacing.sm,
  },
  title: {
    fontSize: FontSizes['2xl'],
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: FontSizes.sm,
    textAlign: 'center',
    lineHeight: 20,
  },
  recordCard: {
    padding: Spacing['2xl'],
  },
  recordContent: {
    alignItems: 'center',
    gap: Spacing.md,
  },
  recordButton: {
    width: 88,
    height: 88,
    borderRadius: 44,
    justifyContent: 'center',
    alignItems: 'center',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
  },
  recordLabel: {
    fontSize: FontSizes.md,
    fontWeight: '600',
  },
  duration: {
    fontSize: FontSizes['2xl'],
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  dividerLine: {
    flex: 1,
    height: 1,
  },
  dividerText: {
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  selectedCard: {
    padding: Spacing.base,
  },
  selectedRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  audioIconBg: {
    width: 48,
    height: 48,
    borderRadius: BorderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  selectedInfo: {
    flex: 1,
    gap: 2,
  },
  selectedName: {
    fontSize: FontSizes.base,
    fontWeight: '600',
  },
  selectedReady: {
    fontSize: FontSizes.xs,
    fontWeight: '500',
  },
  analyzeBtn: {
    marginTop: Spacing.xs,
  },
  infoCard: {
    padding: Spacing.base,
  },
  infoRow: {
    flexDirection: 'row',
    gap: Spacing.sm,
    alignItems: 'flex-start',
  },
  infoText: {
    flex: 1,
    fontSize: FontSizes.xs,
    lineHeight: 18,
  },
  verdictCard: {
    padding: Spacing.xl,
  },
  verdictContent: {
    alignItems: 'center',
    gap: Spacing.md,
  },
  verdictIconBg: {
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: 'center',
    alignItems: 'center',
  },
  verdictLabel: {
    fontSize: FontSizes.xl,
    fontWeight: '800',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  demoBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  demoText: {
    fontSize: FontSizes.xs,
    fontWeight: '600',
  },
});
