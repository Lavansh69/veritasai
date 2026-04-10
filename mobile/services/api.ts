/**
 * VeritasAI — API Service Layer
 * Centralized HTTP client for all backend communication.
 */

import * as FileSystem from 'expo-file-system';
import AsyncStorage from '@react-native-async-storage/async-storage';

const DEFAULT_API_URL = 'http://192.168.1.100:8000';
const API_URL_KEY = '@veritasai_api_url';

let _cachedUrl: string | null = null;

export async function getApiUrl(): Promise<string> {
  if (_cachedUrl) return _cachedUrl;
  try {
    const stored = await AsyncStorage.getItem(API_URL_KEY);
    _cachedUrl = stored || DEFAULT_API_URL;
  } catch {
    _cachedUrl = DEFAULT_API_URL;
  }
  return _cachedUrl;
}

export async function setApiUrl(url: string): Promise<void> {
  _cachedUrl = url;
  await AsyncStorage.setItem(API_URL_KEY, url);
}

// ─── Health Check ────────────────────────────────────────────────
export async function healthCheck(): Promise<{ status: string; service: string }> {
  const base = await getApiUrl();
  const res = await fetch(`${base}/api/health`, { method: 'GET' });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// ─── Image/Video Analysis ────────────────────────────────────────
export interface AnalysisResult {
  analysis_id: string;
  file_hash: string;
  deepfake: {
    probability: number;
    label: string;
    frame_count?: number;
    frame_probabilities?: number[];
  };
  heatmap: {
    heatmap_url: string;
    overlay_url: string;
  };
  face_consistency: {
    score: number;
    details: Record<string, any>;
  };
  metadata: {
    integrity_score: number;
    warnings: string[];
    details: Record<string, any>;
  };
  scorecard: {
    overall_score: number;
    verdict: string;
    confidence: string;
    scores: {
      ai_detection: number;
      face_analysis: number;
      metadata_integrity: number;
      artifact_detection: number;
    };
  };
}

export async function analyzeMedia(
  mediaUri: string,
  mediaName: string,
  referenceUri?: string,
  referenceName?: string
): Promise<AnalysisResult> {
  const base = await getApiUrl();
  const formData = new FormData();

  formData.append('media', {
    uri: mediaUri,
    name: mediaName || 'media.jpg',
    type: mediaName?.endsWith('.mp4') ? 'video/mp4' : 'image/jpeg',
  } as any);

  if (referenceUri) {
    formData.append('reference', {
      uri: referenceUri,
      name: referenceName || 'reference.jpg',
      type: 'image/jpeg',
    } as any);
  }

  const res = await fetch(`${base}/api/analyze`, {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(err.detail || `Analysis failed: ${res.status}`);
  }

  return res.json();
}

// ─── Simple Predict ──────────────────────────────────────────────
export async function predictImage(
  imageUri: string,
  imageName: string
): Promise<{ deepfake_probability: number; verdict: string }> {
  const base = await getApiUrl();
  const formData = new FormData();

  formData.append('file', {
    uri: imageUri,
    name: imageName || 'image.jpg',
    type: 'image/jpeg',
  } as any);

  const res = await fetch(`${base}/api/predict`, {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed' }));
    throw new Error(err.detail || `Prediction failed: ${res.status}`);
  }

  return res.json();
}

// ─── Audio Analysis ──────────────────────────────────────────────
export interface AudioResult {
  analysis_id: string;
  deepfake_probability: number;
  verdict: string;
  demo_mode: boolean;
}

export async function analyzeAudio(
  audioUri: string,
  audioName: string
): Promise<AudioResult> {
  const base = await getApiUrl();
  const formData = new FormData();

  formData.append('audio', {
    uri: audioUri,
    name: audioName || 'audio.wav',
    type: 'audio/wav',
  } as any);

  const res = await fetch(`${base}/api/audio/analyze`, {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Audio analysis failed' }));
    throw new Error(err.detail || `Audio analysis failed: ${res.status}`);
  }

  return res.json();
}

// ─── Feedback ────────────────────────────────────────────────────
export async function submitFeedback(data: {
  analysis_id: string;
  is_correct: boolean;
  corrected_label?: string | null;
  prediction?: string | null;
  confidence?: number | null;
}): Promise<any> {
  const base = await getApiUrl();
  const res = await fetch(`${base}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Feedback failed' }));
    throw new Error(err.detail || `Feedback failed: ${res.status}`);
  }

  return res.json();
}

// ─── Feedback Stats ──────────────────────────────────────────────
export async function getFeedbackStats(): Promise<any> {
  const base = await getApiUrl();
  const res = await fetch(`${base}/api/feedback/stats`);
  if (!res.ok) throw new Error(`Failed to fetch stats: ${res.status}`);
  return res.json();
}

// ─── Report Download ─────────────────────────────────────────────
export async function downloadReport(analysisId: string): Promise<string> {
  const base = await getApiUrl();
  const url = `${base}/api/report/${analysisId}`;
  const fileUri = `${FileSystem.documentDirectory}VeritasAI_Report_${analysisId}.pdf`;

  const download = await FileSystem.downloadAsync(url, fileUri);

  if (download.status !== 200) {
    throw new Error(`Report download failed: ${download.status}`);
  }

  return download.uri;
}

// ─── WebSocket URL ───────────────────────────────────────────────
export async function getLiveDetectionWsUrl(): Promise<string> {
  const base = await getApiUrl();
  const wsBase = base.replace(/^http/, 'ws');
  return `${wsBase}/api/live-detection`;
}

// ─── Heatmap URL helper ──────────────────────────────────────────
export async function getHeatmapUrl(relativePath: string): Promise<string> {
  const base = await getApiUrl();
  return `${base}${relativePath}`;
}
