/**
 * VeritasAI — Live Detection Screen
 * Real-time camera-based deepfake detection via WebSocket.
 */

import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  Pressable,
  Animated,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions, CameraType } from 'expo-camera';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../../theme/typography';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { getLiveDetectionWsUrl } from '../../services/api';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

interface LiveResult {
  deepfake_probability: number;
  fps?: number;
}

export default function LiveScreen() {
  const { colors, isDark } = useTheme();
  const [permission, requestPermission] = useCameraPermissions();
  const [isStreaming, setIsStreaming] = useState(false);
  const [facing, setFacing] = useState<CameraType>('front');
  const [liveResult, setLiveResult] = useState<LiveResult | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, []);

  const startStreaming = async () => {
    try {
      const wsUrl = await getLiveDetectionWsUrl();
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsConnected(true);
        setIsStreaming(true);

        // Pulse animation for live indicator
        Animated.loop(
          Animated.sequence([
            Animated.timing(pulseAnim, { toValue: 0.3, duration: 500, useNativeDriver: true }),
            Animated.timing(pulseAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
          ])
        ).start();

        // Capture and send frames
        streamInterval.current = setInterval(async () => {
          if (cameraRef.current) {
            try {
              const photo = await cameraRef.current.takePictureAsync({
                quality: 0.4,
                base64: true,
                skipProcessing: true,
              });
              if (photo?.base64 && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ frame: photo.base64 }));
              }
            } catch {
              // Skip frame on error
            }
          }
        }, 500); // ~2 fps
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.deepfake_probability !== undefined) {
            setLiveResult({
              deepfake_probability: data.deepfake_probability,
              fps: data.fps,
            });
          }
        } catch {}
      };

      ws.onerror = () => {
        Alert.alert('Connection Error', 'Failed to connect to live detection server.');
        stopStreaming();
      };

      ws.onclose = () => {
        setWsConnected(false);
        stopStreaming();
      };

      wsRef.current = ws;
    } catch (err: any) {
      Alert.alert('Error', err.message || 'Failed to start live detection');
    }
  };

  const stopStreaming = useCallback(() => {
    if (streamInterval.current) {
      clearInterval(streamInterval.current);
      streamInterval.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    pulseAnim.stopAnimation();
    pulseAnim.setValue(1);
    setIsStreaming(false);
    setWsConnected(false);
    setLiveResult(null);
  }, [pulseAnim]);

  const toggleFacing = () => {
    setFacing((prev) => (prev === 'front' ? 'back' : 'front'));
  };

  if (!permission) {
    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <View style={styles.centerContainer}>
          <Text style={[styles.permText, { color: colors.textSecondary }]}>Loading...</Text>
        </View>
      </LinearGradient>
    );
  }

  if (!permission.granted) {
    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <View style={styles.centerContainer}>
          <Ionicons name="videocam-off" size={64} color={colors.textSecondary} style={{ opacity: 0.4 }} />
          <Text style={[styles.permTitle, { color: colors.textPrimary }]}>
            Camera Access Required
          </Text>
          <Text style={[styles.permText, { color: colors.textSecondary }]}>
            Live detection needs camera access to analyze video frames in real-time
          </Text>
          <Button
            title="Grant Camera Access"
            onPress={requestPermission}
            size="lg"
            icon={<Ionicons name="videocam" size={20} color="#fff" />}
          />
        </View>
      </LinearGradient>
    );
  }

  const prob = liveResult?.deepfake_probability ?? 0;
  const borderColor =
    !liveResult ? colors.glassBorder
    : prob < 0.3 ? colors.success
    : prob < 0.7 ? colors.warning
    : colors.danger;

  return (
    <View style={styles.fullScreen}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
      >
        {/* Overlay */}
        <View style={styles.overlay}>
          {/* Top bar */}
          <View style={styles.topBar}>
            <View style={styles.liveBadge}>
              {isStreaming && (
                <Animated.View
                  style={[styles.liveDot, { backgroundColor: colors.danger, opacity: pulseAnim }]}
                />
              )}
              <Text style={styles.liveText}>
                {isStreaming ? 'LIVE' : 'READY'}
              </Text>
            </View>
            {liveResult?.fps && (
              <Text style={styles.fpsText}>{liveResult.fps.toFixed(1)} FPS</Text>
            )}
          </View>

          {/* Detection frame border */}
          <View style={[styles.detectionFrame, { borderColor }]} />

          {/* Live result overlay */}
          {liveResult && isStreaming && (
            <View style={styles.resultOverlay}>
              <GlassCard style={styles.resultCard}>
                <View style={styles.resultContent}>
                  <Ionicons
                    name={prob < 0.3 ? 'shield-checkmark' : prob < 0.7 ? 'warning' : 'alert-circle'}
                    size={28}
                    color={borderColor}
                  />
                  <View>
                    <Text style={[styles.resultVerdict, { color: borderColor }]}>
                      {prob < 0.3 ? 'Authentic' : prob < 0.7 ? 'Suspicious' : 'Deepfake'}
                    </Text>
                    <Text style={styles.resultProb}>
                      {(prob * 100).toFixed(1)}% probability
                    </Text>
                  </View>
                </View>
              </GlassCard>
            </View>
          )}

          {/* Bottom controls */}
          <View style={styles.bottomControls}>
            <Pressable onPress={toggleFacing} style={styles.controlBtn}>
              <Ionicons name="camera-reverse" size={28} color="#fff" />
            </Pressable>

            <Pressable
              onPress={isStreaming ? stopStreaming : startStreaming}
              style={[
                styles.mainBtn,
                { backgroundColor: isStreaming ? colors.danger : colors.accent },
              ]}
            >
              <Ionicons
                name={isStreaming ? 'stop' : 'play'}
                size={32}
                color="#fff"
              />
            </Pressable>

            <View style={styles.controlBtn}>
              {wsConnected && (
                <View style={styles.connectedBadge}>
                  <Ionicons name="wifi" size={16} color={colors.success} />
                  <Text style={[styles.connectedText, { color: colors.success }]}>WS</Text>
                </View>
              )}
            </View>
          </View>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  gradient: { flex: 1 },
  fullScreen: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing['2xl'],
    gap: Spacing.base,
  },
  permTitle: {
    fontSize: FontSizes.xl,
    fontWeight: '700',
    textAlign: 'center',
  },
  permText: {
    fontSize: FontSizes.base,
    textAlign: 'center',
    lineHeight: 24,
  },
  overlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: Spacing.lg,
    paddingTop: 60,
  },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.full,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  liveText: {
    color: '#fff',
    fontSize: FontSizes.xs,
    fontWeight: '700',
    letterSpacing: 1,
  },
  fpsText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: FontSizes.xs,
    fontWeight: '600',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: BorderRadius.full,
  },
  detectionFrame: {
    position: 'absolute',
    top: '15%',
    left: '10%',
    right: '10%',
    bottom: '25%',
    borderWidth: 2,
    borderRadius: BorderRadius.lg,
    borderStyle: 'dashed',
  },
  resultOverlay: {
    position: 'absolute',
    bottom: 140,
    left: Spacing.lg,
    right: Spacing.lg,
  },
  resultCard: {
    padding: Spacing.base,
  },
  resultContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.md,
  },
  resultVerdict: {
    fontSize: FontSizes.md,
    fontWeight: '800',
    textTransform: 'uppercase',
  },
  resultProb: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: FontSizes.sm,
  },
  bottomControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 40,
    paddingHorizontal: Spacing.xl,
  },
  controlBtn: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: 'rgba(255,255,255,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  mainBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
  },
  connectedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  connectedText: {
    fontSize: FontSizes.xs,
    fontWeight: '700',
  },
});
