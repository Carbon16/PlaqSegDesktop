// Types matching the Rust structs in lib.rs

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  size_mb: number;
  file: string;
  backend: 'onnx';
  gpu_capable: boolean;
  path: string;
}

export interface InferenceConfig {
  model_path: string;
  backend: 'onnx';
  device: 'auto' | 'cpu' | 'cuda' | 'directml';
  workers: number;
  tile_size: number;
  stride: number;
  conf_threshold: number;
  nms_threshold: number;
  images: string[];
  output_dir: string | null;
  // Crop
  crop_enabled: boolean;
  crop_scale: number;
  crop_param1: number;
  crop_param2: number;
  crop_padding: number;
}

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  area: number;
}

export interface ImageResult {
  image: string;
  count: number;
  detections: Detection[];
  duration_s: number;
  error: string | null;
}

export type ProgressEvent =
  | { type: 'image_start'; image: string; index: number; total: number }
  | { type: 'image_done'; image: string; index: number; total: number; count: number; duration_s: number }
  | { type: 'image_error'; image: string; index: number; total: number; error: string }
  | { type: 'done'; results: ImageResult[]; total_plaques: number; duration_s: number }
  | { type: 'cancelled' };

export type ImageStatus = 'queued' | 'running' | 'done' | 'error';

export interface QueuedImage {
  path: string;
  name: string;
  status: ImageStatus;
  count: number | null;
  error: string | null;
  duration_s: number | null;
}

export const DEFAULT_CONFIG: InferenceConfig = {
  model_path: '',
  backend: 'onnx',
  device: 'auto',
  workers: 4,
  tile_size: 1280,
  stride: 1024,
  conf_threshold: 0.15,
  nms_threshold: 0.50,
  images: [],
  output_dir: null,
  crop_enabled: false,
  crop_scale: 0.25,
  crop_param1: 100,
  crop_param2: 30,
  crop_padding: 0,
};
