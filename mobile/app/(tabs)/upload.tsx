/**
 * VeritasAI — Upload & Detect Screen
 * Camera capture, gallery pick, and file upload for deepfake analysis.
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Image,
  Alert,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { router } from 'expo-router';
import { useTheme } from '../../theme/ThemeContext';
import { Gradients } from '../../theme/colors';
import { FontSizes, Spacing, BorderRadius } from '../../theme/typography';
import { GlassCard } from '../../components/ui/GlassCard';
import { Button } from '../../components/ui/Button';
import { LoadingSpinner } from '../../components/ui/LoadingSpinner';
import { analyzeMedia } from '../../services/api';

const { width } = Dimensions.get('window');

export default function UploadScreen() {
  const { colors, isDark } = useTheme();
  const [selectedMedia, setSelectedMedia] = useState<{
    uri: string;
    name: string;
    type: 'image' | 'video';
  } | null>(null);
  const [referenceMedia, setReferenceMedia] = useState<{
    uri: string;
    name: string;
  } | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState('');

  const pickFromCamera = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Camera access is needed to capture photos for analysis.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.9,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      setSelectedMedia({
        uri: asset.uri,
        name: asset.fileName || `capture_${Date.now()}.jpg`,
        type: 'image',
      });
    }
  };

  const pickFromGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Photo library access is needed to select media.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images', 'videos'],
      quality: 0.9,
      allowsEditing: false,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      const isVideo = asset.type === 'video' || asset.uri.endsWith('.mp4');
      setSelectedMedia({
        uri: asset.uri,
        name: asset.fileName || `media_${Date.now()}.${isVideo ? 'mp4' : 'jpg'}`,
        type: isVideo ? 'video' : 'image',
      });
    }
  };

  const pickFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['image/*', 'video/*'],
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        const isVideo = asset.mimeType?.startsWith('video') || asset.name?.endsWith('.mp4');
        setSelectedMedia({
          uri: asset.uri,
          name: asset.name || `file_${Date.now()}`,
          type: isVideo ? 'video' : 'image',
        });
      }
    } catch (err) {
      Alert.alert('Error', 'Failed to pick file');
    }
  };

  const pickReference = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.9,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      setReferenceMedia({
        uri: asset.uri,
        name: asset.fileName || `reference_${Date.now()}.jpg`,
      });
    }
  };

  const handleAnalyze = async () => {
    if (!selectedMedia) return;

    setAnalyzing(true);
    setProgress('Uploading media...');

    try {
      setProgress('Running AI analysis...');
      const result = await analyzeMedia(
        selectedMedia.uri,
        selectedMedia.name,
        referenceMedia?.uri,
        referenceMedia?.name
      );

      setProgress('');
      setAnalyzing(false);

      // Navigate to results, passing the analysis data
      router.push({
        pathname: '/analysis/[id]',
        params: {
          id: result.analysis_id,
          data: JSON.stringify(result),
        },
      });
    } catch (err: any) {
      setAnalyzing(false);
      setProgress('');
      Alert.alert('Analysis Failed', err.message || 'Failed to analyze media. Check server connection.');
    }
  };

  const clearSelection = () => {
    setSelectedMedia(null);
    setReferenceMedia(null);
  };

  if (analyzing) {
    return (
      <LinearGradient
        colors={isDark ? [...Gradients.background.dark] : [...Gradients.background.light]}
        style={styles.gradient}
      >
        <LoadingSpinner message={progress || 'Analyzing media for deepfake signatures...'} />
      </LinearGradient>
    );
  }

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
          <Ionicons name="shield-checkmark" size={28} color={colors.accent} />
          <Text style={[styles.title, { color: colors.textPrimary }]}>
            Deepfake Detection
          </Text>
          <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
            Upload or capture media to analyze with our multi-model AI pipeline
          </Text>
        </View>

        {/* Selected Media Preview */}
        {selectedMedia ? (
          <GlassCard style={styles.previewCard} glowing>
            <Image
              source={{ uri: selectedMedia.uri }}
              style={styles.previewImage}
              resizeMode="cover"
            />
            <View style={styles.previewInfo}>
              <View style={styles.previewBadge}>
                <Ionicons
                  name={selectedMedia.type === 'video' ? 'videocam' : 'image'}
                  size={14}
                  color={colors.accent}
                />
                <Text style={[styles.previewType, { color: colors.accent }]}>
                  {selectedMedia.type.toUpperCase()}
                </Text>
              </View>
              <Text style={[styles.previewName, { color: colors.textPrimary }]} numberOfLines={1}>
                {selectedMedia.name}
              </Text>
              <Button
                title="Remove"
                variant="danger"
                size="sm"
                onPress={clearSelection}
                icon={<Ionicons name="close" size={14} color={colors.danger} />}
              />
            </View>
          </GlassCard>
        ) : (
          /* Upload Options */
          <GlassCard style={styles.uploadZone}>
            <Ionicons name="cloud-upload" size={48} color={colors.accent} style={{ opacity: 0.6 }} />
            <Text style={[styles.uploadTitle, { color: colors.textPrimary }]}>
              Select Media to Analyze
            </Text>
            <Text style={[styles.uploadHint, { color: colors.textSecondary }]}>
              Supports JPEG, PNG, MP4, AVI, MOV
            </Text>
          </GlassCard>
        )}

        {/* Source Buttons */}
        <View style={styles.sourceButtons}>
          <Button
            title="Camera"
            variant={selectedMedia ? 'outline' : 'primary'}
            onPress={pickFromCamera}
            icon={<Ionicons name="camera" size={20} color={selectedMedia ? colors.textPrimary : '#fff'} />}
            style={styles.sourceBtn}
          />
          <Button
            title="Gallery"
            variant={selectedMedia ? 'outline' : 'primary'}
            onPress={pickFromGallery}
            icon={<Ionicons name="images" size={20} color={selectedMedia ? colors.textPrimary : '#fff'} />}
            style={styles.sourceBtn}
          />
          <Button
            title="File"
            variant={selectedMedia ? 'outline' : 'primary'}
            onPress={pickFile}
            icon={<Ionicons name="document" size={20} color={selectedMedia ? colors.textPrimary : '#fff'} />}
            style={styles.sourceBtn}
          />
        </View>

        {/* Reference Image (optional) */}
        {selectedMedia && (
          <GlassCard style={styles.referenceCard}>
            <View style={styles.referenceHeader}>
              <Text style={[styles.referenceTitle, { color: colors.textPrimary }]}>
                Reference Image (Optional)
              </Text>
              <Text style={[styles.referenceHint, { color: colors.textSecondary }]}>
                Add an authentic reference for face comparison
              </Text>
            </View>
            {referenceMedia ? (
              <View style={styles.referencePreview}>
                <Image
                  source={{ uri: referenceMedia.uri }}
                  style={styles.referenceThumb}
                  resizeMode="cover"
                />
                <Text style={[styles.referenceName, { color: colors.textSecondary }]} numberOfLines={1}>
                  {referenceMedia.name}
                </Text>
                <Ionicons
                  name="checkmark-circle"
                  size={20}
                  color={colors.success}
                />
              </View>
            ) : null}
            <Button
              title={referenceMedia ? 'Change Reference' : 'Add Reference'}
              variant="outline"
              size="sm"
              onPress={pickReference}
              icon={<Ionicons name="person-add" size={16} color={colors.textPrimary} />}
            />
          </GlassCard>
        )}

        {/* Analyze Button */}
        {selectedMedia && (
          <Button
            title="🔍  Analyze for Deepfakes"
            onPress={handleAnalyze}
            size="lg"
            style={styles.analyzeBtn}
          />
        )}

        {/* Info cards */}
        <GlassCard style={styles.infoCard}>
          <View style={styles.infoRow}>
            <Ionicons name="information-circle" size={20} color={colors.accent} />
            <Text style={[styles.infoText, { color: colors.textSecondary }]}>
              Our AI pipeline combines CNN detection, face consistency analysis, metadata inspection, and Grad-CAM explainability for comprehensive results.
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
  previewCard: {
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: width * 0.55,
    borderTopLeftRadius: BorderRadius.lg,
    borderTopRightRadius: BorderRadius.lg,
  },
  previewInfo: {
    padding: Spacing.base,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  previewBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: BorderRadius.full,
    backgroundColor: 'rgba(129, 140, 248, 0.15)',
  },
  previewType: {
    fontSize: FontSizes.xs,
    fontWeight: '700',
  },
  previewName: {
    flex: 1,
    fontSize: FontSizes.sm,
    fontWeight: '500',
  },
  uploadZone: {
    padding: Spacing['2xl'],
    alignItems: 'center',
    gap: Spacing.sm,
    borderStyle: 'dashed',
  },
  uploadTitle: {
    fontSize: FontSizes.md,
    fontWeight: '700',
  },
  uploadHint: {
    fontSize: FontSizes.sm,
  },
  sourceButtons: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  sourceBtn: {
    flex: 1,
  },
  referenceCard: {
    padding: Spacing.base,
    gap: Spacing.sm,
  },
  referenceHeader: {
    gap: 2,
  },
  referenceTitle: {
    fontSize: FontSizes.base,
    fontWeight: '600',
  },
  referenceHint: {
    fontSize: FontSizes.xs,
  },
  referencePreview: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.sm,
  },
  referenceThumb: {
    width: 48,
    height: 48,
    borderRadius: BorderRadius.sm,
  },
  referenceName: {
    flex: 1,
    fontSize: FontSizes.sm,
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
});
