import { invoke } from '@tauri-apps/api/core';
import type { ModelInfo, InferenceConfig } from './types';

export const commands = {
  scanModels: (folder: string) =>
    invoke<ModelInfo[]>('scan_models', { folder }),

  runInference: (config: InferenceConfig) =>
    invoke<void>('run_inference', { config }),

  cancelInference: () =>
    invoke<void>('cancel_inference'),

  selectImages: () =>
    invoke<string[]>('select_images'),

  selectFolder: () =>
    invoke<string | null>('select_folder'),

  selectModelsFolder: () =>
    invoke<string | null>('select_models_folder'),

  getImagesInFolder: (folder: string) =>
    invoke<string[]>('get_images_in_folder', { folder }),

  saveResultsJson: (content: string) =>
    invoke<void>('save_results_json', { content }),

  saveResultsCsv: (content: string) =>
    invoke<void>('save_results_csv', { content }),

  openInExplorer: (path: string) =>
    invoke<void>('open_in_explorer', { path }),

  readImagePreview: (path: string) =>
    invoke<[number[], number, number]>('read_image_preview', { path }),
};
