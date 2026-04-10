'use client';
import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, CameraOff, User } from 'lucide-react';

interface BBox {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

interface FaceResult {
  bbox: BBox;
  confidence: number;
  label: string;
}

interface WebcamViewProps {
  faces: FaceResult[];
  heatmapSrc: string | null;
  showHeatmap: boolean;
  facesDetected: number;
  isActive: boolean;
  label: string;
}

export interface WebcamViewHandle {
  captureFrame: () => Promise<Blob | null>;
  startCamera: () => Promise<boolean>;
  stopCamera: () => void;
  getVideoSize: () => { width: number; height: number } | null;
}

const WebcamView = forwardRef<WebcamViewHandle, WebcamViewProps>(
  ({ faces, heatmapSrc, showHeatmap, facesDetected, isActive, label }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const overlayRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const animFrameRef = useRef<number>(0);
    const facesRef = useRef<FaceResult[]>([]);
    const labelRef = useRef<string>('');
    const [cameraReady, setCameraReady] = useState(false);
    const [cameraError, setCameraError] = useState<string | null>(null);
    const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 });

    // Keep refs in sync to avoid re-renders in the draw loop
    useEffect(() => { facesRef.current = faces; }, [faces]);
    useEffect(() => { labelRef.current = label; }, [label]);

    // ── Start / Stop camera ────────────────────────────────────────
    const startCamera = useCallback(async (): Promise<boolean> => {
      try {
        setCameraError(null);
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
            // Disable auto frame-rate reduction on mobile to prevent blinking
            frameRate: { ideal: 15, max: 30 },
          },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Use a promise to wait for the video to fully start
          await new Promise<void>((resolve) => {
            const v = videoRef.current!;
            const onPlay = () => { v.removeEventListener('playing', onPlay); resolve(); };
            v.addEventListener('playing', onPlay);
            v.play().catch(() => resolve());
          });
        }
        setCameraReady(true);
        return true;
      } catch (err: any) {
        const msg =
          err.name === 'NotAllowedError'
            ? 'Camera permission denied. Please allow camera access and reload.'
            : err.name === 'NotFoundError'
            ? 'No camera found on this device.'
            : `Camera error: ${err.message}`;
        setCameraError(msg);
        setCameraReady(false);
        return false;
      }
    }, []);

    const stopCamera = useCallback(() => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraReady(false);
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = 0;
      }
    }, []);

    // Track actual video dimensions once metadata loads
    useEffect(() => {
      const video = videoRef.current;
      if (!video) return;
      const handler = () => {
        setVideoDimensions({ width: video.videoWidth, height: video.videoHeight });
      };
      video.addEventListener('loadedmetadata', handler);
      return () => video.removeEventListener('loadedmetadata', handler);
    }, []);

    // ── Capture a frame as JPEG Blob ──────────────────────────────
    const captureFrame = useCallback(async (): Promise<Blob | null> => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !cameraReady) return null;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0);
      return new Promise((resolve) => canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.7));
    }, [cameraReady]);

    const getVideoSize = useCallback(() => {
      if (!cameraReady) return null;
      return videoDimensions;
    }, [cameraReady, videoDimensions]);

    useImperativeHandle(ref, () => ({ captureFrame, startCamera, stopCamera, getVideoSize }), [
      captureFrame,
      startCamera,
      stopCamera,
      getVideoSize,
    ]);

    // ── Continuous overlay draw loop (RAF-based, prevents blinking) ──
    useEffect(() => {
      const overlay = overlayRef.current;
      const video = videoRef.current;
      if (!overlay || !video || !cameraReady) return;

      let lastWidth = 0;
      let lastHeight = 0;

      const draw = () => {
        const rect = video.getBoundingClientRect();
        // Only resize canvas when the container actually changes size
        if (Math.round(rect.width) !== lastWidth || Math.round(rect.height) !== lastHeight) {
          lastWidth = Math.round(rect.width);
          lastHeight = Math.round(rect.height);
          overlay.width = lastWidth;
          overlay.height = lastHeight;
        }

        const ctx = overlay.getContext('2d');
        if (!ctx) { animFrameRef.current = requestAnimationFrame(draw); return; }
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const currentFaces = facesRef.current;
        if (currentFaces.length === 0) {
          animFrameRef.current = requestAnimationFrame(draw);
          return;
        }

        const scaleX = rect.width / videoDimensions.width;
        const scaleY = rect.height / videoDimensions.height;

        for (const face of currentFaces) {
          const { bbox, label: faceLabel, confidence } = face;

          // Mirror horizontally for selfie view
          const bx = rect.width - bbox.right * scaleX;
          const by = bbox.top * scaleY;
          const bw = (bbox.right - bbox.left) * scaleX;
          const bh = (bbox.bottom - bbox.top) * scaleY;

          // Pick color based on per-face label
          const color =
            faceLabel === 'Likely Deepfake'
              ? '#f87171'
              : faceLabel === 'Suspicious'
              ? '#fbbf24'
              : '#34d399';

          // ── Neon glow rectangle ──
          ctx.save();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.5;
          ctx.shadowColor = color;
          ctx.shadowBlur = 14;
          ctx.beginPath();
          ctx.roundRect(bx, by, bw, bh, 8);
          ctx.stroke();

          // ── Corner brackets (premium look) ──
          const cornerLen = Math.min(bw, bh) * 0.2;
          ctx.lineWidth = 3;
          ctx.shadowBlur = 8;
          // Top-left
          ctx.beginPath();
          ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
          ctx.stroke();
          // Top-right
          ctx.beginPath();
          ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
          ctx.stroke();
          // Bottom-left
          ctx.beginPath();
          ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
          ctx.stroke();
          // Bottom-right
          ctx.beginPath();
          ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
          ctx.stroke();

          // ── Label badge above box ──
          const confText = `${faceLabel} ${Math.round(confidence * 100)}%`;
          ctx.font = '600 12px Inter, system-ui, sans-serif';
          const metrics = ctx.measureText(confText);
          const padX = 8;
          const padY = 4;
          const badgeW = metrics.width + padX * 2;
          const badgeH = 22;
          const badgeX = bx;
          const badgeY = by - badgeH - 4;

          // Badge background
          ctx.shadowBlur = 0;
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 6);
          ctx.fill();

          // Badge text
          ctx.globalAlpha = 1;
          ctx.fillStyle = '#000';
          ctx.fillText(confText, badgeX + padX, badgeY + badgeH - padY - 2);
          ctx.restore();
        }

        animFrameRef.current = requestAnimationFrame(draw);
      };

      animFrameRef.current = requestAnimationFrame(draw);
      return () => {
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      };
    }, [cameraReady, videoDimensions]);

    return (
      <div className="relative w-full rounded-2xl overflow-hidden bg-black/40"
        style={{ aspectRatio: `${videoDimensions.width} / ${videoDimensions.height}` }}
      >
        {/* Video */}
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
          muted
          playsInline
          autoPlay
        />

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Bounding-box overlay canvas */}
        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 10 }}
        />

        {/* Face count badge */}
        <AnimatePresence>
          {isActive && cameraReady && facesDetected === 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold"
              style={{
                background: 'rgba(251,191,36,0.15)',
                border: '1px solid rgba(251,191,36,0.4)',
                color: '#fbbf24',
                backdropFilter: 'blur(8px)',
              }}
            >
              <User className="w-3.5 h-3.5" />
              No face detected
            </motion.div>
          )}
        </AnimatePresence>

        {/* Multi-face counter */}
        <AnimatePresence>
          {isActive && cameraReady && facesDetected > 1 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold"
              style={{
                background: 'rgba(99,102,241,0.15)',
                border: '1px solid rgba(99,102,241,0.4)',
                color: '#818cf8',
                backdropFilter: 'blur(8px)',
              }}
            >
              <User className="w-3.5 h-3.5" />
              {facesDetected} faces detected
            </motion.div>
          )}
        </AnimatePresence>

        {/* Heatmap PiP */}
        <AnimatePresence>
          {showHeatmap && heatmapSrc && (
            <motion.div
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.85 }}
              className="absolute bottom-3 right-3 z-20 rounded-xl overflow-hidden border"
              style={{
                width: 160,
                height: 160,
                borderColor: 'rgba(129,140,248,0.5)',
                boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
              }}
            >
              <img
                src={`data:image/jpeg;base64,${heatmapSrc}`}
                alt="Grad-CAM heatmap"
                className="w-full h-full object-cover"
              />
              <div
                className="absolute bottom-0 inset-x-0 text-center text-[10px] font-medium py-0.5"
                style={{ background: 'rgba(0,0,0,0.6)', color: '#818cf8' }}
              >
                Grad-CAM
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Camera error overlay */}
        {cameraError && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-4 bg-black/70 text-center px-8">
            <CameraOff className="w-12 h-12 text-red-400" />
            <p className="text-sm text-red-300 max-w-xs">{cameraError}</p>
            <button
              onClick={startCamera}
              className="btn-primary text-sm !px-4 !py-2"
            >
              Retry
            </button>
          </div>
        )}

        {/* Idle state */}
        {!cameraReady && !cameraError && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3"
            style={{ aspectRatio: '4 / 3', minHeight: '240px' }}
          >
            <Camera className="w-10 h-10" style={{ color: 'var(--text-secondary)' }} />
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Camera will activate when you start detection
            </p>
          </div>
        )}
      </div>
    );
  }
);

WebcamView.displayName = 'WebcamView';
export default WebcamView;
