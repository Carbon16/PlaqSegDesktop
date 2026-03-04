use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{s, Array, Array4, ArrayView};
use ort::{
    execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider},
    session::{builder::SessionBuilder, Session},
    value::Tensor,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};
use tauri::{AppHandle, Emitter, Manager, State};


// ──────────────────────────────────────────────────────────────────────────────
// Types mirrored on the frontend
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub label: String,
    pub description: String,
    pub size_mb: f64,
    pub file: String,
    pub backend: String,
    pub gpu_capable: bool,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub model_path: String,
    pub backend: String,
    pub device: String,        // "cpu" | "cuda" | "directml"
    pub workers: u32,
    pub tile_size: u32,
    pub stride: u32,
    pub conf_threshold: f32,
    pub nms_threshold: f32,
    pub images: Vec<String>,
    pub output_dir: Option<String>,
    // Crop
    pub crop_enabled: bool,
    pub crop_scale: f32,
    pub crop_param1: f32,
    pub crop_param2: f32,
    pub crop_padding: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub confidence: f32,
    pub area: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResult {
    pub image: String,
    pub count: usize,
    pub detections: Vec<Detection>,
    pub duration_s: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressEvent {
    ImageStart {
        image: String,
        index: usize,
        total: usize,
    },
    ImageDone {
        image: String,
        index: usize,
        total: usize,
        count: usize,
        duration_s: f64,
    },
    ImageError {
        image: String,
        index: usize,
        total: usize,
        error: String,
    },
    Done {
        results: Vec<ImageResult>,
        total_plaques: usize,
        duration_s: f64,
    },
    Cancelled,
}

// ──────────────────────────────────────────────────────────────────────────────
// App state — cancel flag + live results
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct AppState {
    pub cancel_flag: Arc<AtomicBool>,
    pub results: Arc<Mutex<Vec<ImageResult>>>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Model scanning
// ──────────────────────────────────────────────────────────────────────────────

static MODEL_CATALOG: &[(&str, &str, &str, &str, f64, bool)] = &[
    // (filename_pattern, id, label, description, size_mb, gpu_capable)
    ("nano.onnx", "nano", "Nano · ONNX", "Fastest – GPU/CPU, most accurate for size", 12.0, true),
    ("small.onnx", "small", "Small · ONNX", "Higher accuracy – GPU/CPU, larger model", 44.0, true),
];

#[tauri::command]
fn scan_models(folder: String) -> Result<Vec<ModelInfo>, String> {
    let dir = Path::new(&folder);
    if !dir.is_dir() {
        return Err(format!("Not a directory: {folder}"));
    }

    let mut found = Vec::new();
    for (filename, id, label, description, size_mb, gpu_capable) in MODEL_CATALOG {
        let path = dir.join(filename);
        if path.exists() {
            let backend = "onnx";
            let actual_mb = std::fs::metadata(&path)
                .map(|m| m.len() as f64 / 1_048_576.0)
                .unwrap_or(*size_mb);
            found.push(ModelInfo {
                id: id.to_string(),
                label: label.to_string(),
                description: description.to_string(),
                size_mb: (actual_mb * 10.0).round() / 10.0,
                file: filename.to_string(),
                backend: backend.to_string(),
                gpu_capable: *gpu_capable,
                path: path.to_string_lossy().to_string(),
            });
        }
    }
    Ok(found)
}

// ──────────────────────────────────────────────────────────────────────────────
// ONNX Inference
// ──────────────────────────────────────────────────────────────────────────────

fn build_ort_session(model_path: &str, device: &str) -> Result<Session> {
    let mut builder = Session::builder()
        .with_context(|| format!("Failed to create session builder for {model_path}"))?;

    match device {
        "auto" => {
            builder = builder.with_execution_providers([
                CUDAExecutionProvider::default().build(),
                DirectMLExecutionProvider::default().build(),
            ])?;
        }
        "cuda" | "cuda:0" => {
            builder = builder.with_execution_providers([
                CUDAExecutionProvider::default().build(),
            ])?;
        }
        "directml" => {
            builder = builder.with_execution_providers([
                DirectMLExecutionProvider::default().build(),
            ])?;
        }
        _ => {}
    }

    let mut session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load ONNX model: {model_path}"))?;

    // Warm up the session so the first image isn't terribly slow
    if !session.inputs().is_empty() {
        let input_name = session.inputs()[0].name().to_string();
        let shape = vec![1, 3, 640, 640];
        let zero_data = vec![0.0f32; 1 * 3 * 640 * 640];
        if let Ok(tensor_val) = Tensor::from_array((shape, zero_data)) {
            let _ = session.run(ort::inputs! { input_name => tensor_val });
        }
    }

    Ok(session)
}

fn preprocess_tile_onnx(tile: &DynamicImage, size: u32) -> Array4<f32> {
    let resized = tile.resize_exact(size, size, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let (w, h) = (size as usize, size as usize);
    let mut arr = Array4::<f32>::zeros((1, 3, h, w));
    for (x, y, px) in rgb.enumerate_pixels() {
        let (xi, yi) = (x as usize, y as usize);
        arr[[0, 0, yi, xi]] = px[0] as f32 / 255.0;
        arr[[0, 1, yi, xi]] = px[1] as f32 / 255.0;
        arr[[0, 2, yi, xi]] = px[2] as f32 / 255.0;
    }
    arr
}

fn run_onnx_tile(session: &mut Session, input: Array4<f32>) -> Result<Vec<Detection>> {
    let input_name = session.inputs()[0].name().to_string();
    let shape = vec![1, 3, 640, 640];
    let tensor_val = Tensor::from_array((shape, input.into_raw_vec_and_offset().0))?;
    let outputs = session.run(ort::inputs! { input_name => tensor_val })?;
    let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
    parse_yolo_output_onnx(data, 0, 0, 640, 640, 0.15)
}

/// Parse YOLO E2E output: shape (1, 300, 38) — [x1, y1, x2, y2, cls0, cls1, ...32 mask coeffs]
/// tileX/Y are offsets in the full image coordinate system.
fn parse_yolo_output_onnx(
    flat: &[f32],
    tile_x: u32,
    tile_y: u32,
    tile_w: u32,
    tile_h: u32,
    conf_threshold: f32,
) -> Result<Vec<Detection>> {
    let num_dets = 300usize;
    let num_attribs = 38usize;
    let num_classes = 2usize;

    let mut dets = Vec::new();

    for i in 0..num_dets {
        let off = i * num_attribs;
        if off + num_attribs > flat.len() {
            break;
        }
        let x1n = flat[off];
        let y1n = flat[off + 1];
        let x2n = flat[off + 2];
        let y2n = flat[off + 3];

        // Max class confidence
        let mut max_conf = 0f32;
        for c in 0..num_classes {
            let p = flat[off + 4 + c];
            if p > max_conf {
                max_conf = p;
            }
        }
        if max_conf < conf_threshold {
            continue;
        }

        // Skip zero placeholders
        let sum = x1n.abs() + y1n.abs() + x2n.abs() + y2n.abs();
        if sum < 0.01 {
            continue;
        }

        let x1 = x1n * tile_w as f32;
        let y1 = y1n * tile_h as f32;
        let x2 = x2n * tile_w as f32;
        let y2 = y2n * tile_h as f32;
        let w = x2 - x1;
        let h = y2 - y1;

        if w <= 0.0 || h <= 0.0 { continue; }
        if w > tile_w as f32 * 0.3 || h > tile_h as f32 * 0.3 { continue; }
        if w < 10.0 || h < 10.0 { continue; }
        let aspect = w.max(h) / w.min(h);
        if aspect > 3.0 { continue; }

        dets.push(Detection {
            x: (tile_x as f32 + x1) as i32,
            y: (tile_y as f32 + y1) as i32,
            width: w as i32,
            height: h as i32,
            confidence: max_conf,
            area: (w * h) as i32,
        });
    }

    Ok(dets)
}


// ──────────────────────────────────────────────────────────────────────────────
// Tiled ONNX inference for a single image
// ──────────────────────────────────────────────────────────────────────────────

fn run_onnx_image(
    session: &mut Session,
    image: &DynamicImage,
    tile_size: u32,
    stride: u32,
    conf_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Detection>> {
    let (img_w, img_h) = image.dimensions();
    let input_size = 640u32;
    let mut all_dets: Vec<Detection> = Vec::new();

    let overlap = tile_size - stride;
    let tiles_x = ((img_w as f32 - overlap as f32) / stride as f32).ceil().max(1.0) as u32;
    let tiles_y = ((img_h as f32 - overlap as f32) / stride as f32).ceil().max(1.0) as u32;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x = (tx * stride).min(img_w.saturating_sub(tile_size));
            let y = (ty * stride).min(img_h.saturating_sub(tile_size));
            let cw = tile_size.min(img_w - x);
            let ch = tile_size.min(img_h - y);

            let tile = image.crop_imm(x, y, cw, ch);
            let tensor = preprocess_tile_onnx(&tile, input_size);

            let input_name = session.inputs()[0].name().to_string();
            let shape = vec![1, 3, 640, 640];
            let tensor_val = Tensor::from_array((shape, tensor.into_raw_vec_and_offset().0))?;
            let outputs = session
                .run(ort::inputs! { input_name => tensor_val })
                .context("ONNX inference failed")?;

            let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
            let mut tile_dets = parse_yolo_output_onnx(data, x, y, cw, ch, conf_threshold)?;
            all_dets.append(&mut tile_dets);
        }
    }

    Ok(cross_tile_nms(all_dets, nms_threshold))
}

// ──────────────────────────────────────────────────────────────────────────────
// Cross-tile NMS
// ──────────────────────────────────────────────────────────────────────────────

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ax2 = a.x + a.width;
    let ay2 = a.y + a.height;
    let bx2 = b.x + b.width;
    let by2 = b.y + b.height;

    let inter_x1 = a.x.max(b.x);
    let inter_y1 = a.y.max(b.y);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
        return 0.0;
    }
    let inter = ((inter_x2 - inter_x1) * (inter_y2 - inter_y1)) as f32;
    let a_area = (a.width * a.height) as f32;
    let b_area = (b.width * b.height) as f32;
    inter / (a_area + b_area - inter)
}

fn cross_tile_nms(mut dets: Vec<Detection>, iou_thresh: f32) -> Vec<Detection> {
    dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<Detection> = Vec::new();
    'outer: for det in dets {
        for k in &kept {
            if iou(k, &det) > iou_thresh {
                continue 'outer;
            }
        }
        kept.push(det);
    }
    kept
}

// ──────────────────────────────────────────────────────────────────────────────
// Process a single image — dispatches to ONNX
// ──────────────────────────────────────────────────────────────────────────────

fn process_image_onnx(
    path: &str,
    session: &mut Session,
    config: &InferenceConfig,
) -> ImageResult {
    let t0 = Instant::now();
    match image::open(path) {
        Err(e) => ImageResult {
            image: path.to_string(),
            count: 0,
            detections: vec![],
            duration_s: t0.elapsed().as_secs_f64(),
            error: Some(e.to_string()),
        },
        Ok(img) => {
            match run_onnx_image(session, &img, config.tile_size, config.stride, config.conf_threshold, config.nms_threshold) {
                Err(e) => ImageResult {
                    image: path.to_string(),
                    count: 0,
                    detections: vec![],
                    duration_s: t0.elapsed().as_secs_f64(),
                    error: Some(e.to_string()),
                },
                Ok(dets) => {
                    let count = dets.len();
                    ImageResult {
                        image: path.to_string(),
                        count,
                        detections: dets,
                        duration_s: t0.elapsed().as_secs_f64(),
                        error: None,
                    }
                }
            }
        }
    }
}



// ──────────────────────────────────────────────────────────────────────────────
// CSV/JSON helpers
// ──────────────────────────────────────────────────────────────────────────────

pub fn results_to_json(results: &[ImageResult]) -> String {
    serde_json::to_string_pretty(results).unwrap_or_default()
}

pub fn results_to_csv(results: &[ImageResult]) -> String {
    let mut out = String::from("image,plaque_count,x,y,width,height,confidence,area,duration_s,error\n");
    for r in results {
        if r.detections.is_empty() {
            out.push_str(&format!(
                "{},{},,,,,,,{},{}\n",
                r.image, r.count, r.duration_s,
                r.error.as_deref().unwrap_or("")
            ));
        } else {
            for d in &r.detections {
                out.push_str(&format!(
                    "{},{},{},{},{},{},{:.4},{},{:.4},{}\n",
                    r.image, r.count, d.x, d.y, d.width, d.height, d.confidence, d.area, r.duration_s,
                    r.error.as_deref().unwrap_or("")
                ));
            }
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// Tauri commands
// ──────────────────────────────────────────────────────────────────────────────

#[tauri::command]
async fn select_images(app: AppHandle) -> Result<Vec<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let files = app
        .dialog()
        .file()
        .add_filter("Images", &["jpg", "jpeg", "png", "tiff", "tif", "bmp"])
        .set_title("Select images")
        .blocking_pick_files();

    Ok(files
        .map(|v| v.into_iter().map(|p| p.to_string()).collect())
        .unwrap_or_default())
}

#[tauri::command]
async fn select_folder(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let folder = app
        .dialog()
        .file()
        .set_title("Select image folder")
        .blocking_pick_folder();
    Ok(folder.map(|p| p.to_string()))
}

#[tauri::command]
async fn select_models_folder(app: AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let folder = app
        .dialog()
        .file()
        .set_title("Select models folder")
        .blocking_pick_folder();
    Ok(folder.map(|p| p.to_string()))
}

#[tauri::command]
async fn save_results_json(app: AppHandle, content: String) -> Result<(), String> {
    use tauri_plugin_dialog::DialogExt;
    if let Some(path) = app
        .dialog()
        .file()
        .add_filter("JSON", &["json"])
        .set_title("Save results as JSON")
        .blocking_save_file()
    {
        std::fs::write(path.to_string(), content).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn read_image_preview(path: String) -> Result<(Vec<u8>, u32, u32), String> {
    use image::GenericImageView;
    let img = image::open(&path).map_err(|e| e.to_string())?;
    let (original_w, original_h) = img.dimensions();
    
    let max_dim = 1200;
    let resized = if original_w > max_dim || original_h > max_dim {
        img.resize(max_dim, max_dim, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    
    let mut cursor = std::io::Cursor::new(Vec::new());
    resized.write_to(&mut cursor, image::ImageFormat::Jpeg).map_err(|e| e.to_string())?;
    
    Ok((cursor.into_inner(), original_w, original_h))
}

#[tauri::command]
async fn save_results_csv(app: AppHandle, content: String) -> Result<(), String> {
    use tauri_plugin_dialog::DialogExt;
    if let Some(path) = app
        .dialog()
        .file()
        .add_filter("CSV", &["csv"])
        .set_title("Save results as CSV")
        .blocking_save_file()
    {
        std::fs::write(path.to_string(), content).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn open_in_explorer(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn cancel_inference(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel_flag.store(true, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
async fn get_images_in_folder(folder: String) -> Result<Vec<String>, String> {
    let dir = Path::new(&folder);
    if !dir.is_dir() {
        return Err(format!("Not a directory: {folder}"));
    }
    let exts = ["jpg", "jpeg", "png", "tiff", "tif", "bmp"];
    let mut images = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if exts.contains(&ext.to_lowercase().as_str()) {
                    images.push(path.to_string_lossy().to_string());
                }
            }
        }
    }
    images.sort();
    Ok(images)
}

#[tauri::command]
async fn run_inference(
    app: AppHandle,
    state: State<'_, AppState>,
    config: InferenceConfig,
) -> Result<(), String> {
    // Reset cancel flag and results
    state.cancel_flag.store(false, Ordering::Relaxed);
    *state.results.lock().unwrap() = Vec::new();

    let cancel = Arc::clone(&state.cancel_flag);
    let results_store = Arc::clone(&state.results);
    let app_clone = app.clone();
    let config = Arc::new(config);

    tokio::spawn(async move {
        let batch_start = Instant::now();
        let images = config.images.clone();
        let total = images.len();
        let mut all_results: Vec<ImageResult> = Vec::new();

        match config.backend.as_str() {
            "onnx" => {
                // Build session(s) based on worker count
                let mut session = match build_ort_session(&config.model_path, &config.device) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = app_clone.emit("inference_error", format!("Failed to load model: {e}"));
                        return;
                    }
                };

                // For GPU single session; for CPU use rayon
                if config.device == "cpu" {
                    let workers = config.workers as usize;
                    // Use rayon with N threads, each gets its own session clone
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap_or(rayon::ThreadPoolBuilder::new().build().unwrap());

                    let results_mutex: Arc<Mutex<Vec<(usize, ImageResult)>>> = Arc::new(Mutex::new(Vec::new()));
                    let cancel_r = Arc::clone(&cancel);
                    let config_r = Arc::clone(&config);

                    pool.install(|| {
                        images.par_iter().enumerate().for_each(|(idx, img_path)| {
                            if cancel_r.load(Ordering::Relaxed) { return; }

                            let _ = app_clone.emit("inference_progress", ProgressEvent::ImageStart {
                                image: img_path.clone(),
                                index: idx,
                                total,
                            });

                            // Each rayon thread builds its own session to avoid Send issues
                            let result = match build_ort_session(&config_r.model_path, "cpu") {
                                Ok(mut sess) => process_image_onnx(img_path, &mut sess, &config_r),
                                Err(e) => ImageResult {
                                    image: img_path.clone(),
                                    count: 0,
                                    detections: vec![],
                                    duration_s: 0.0,
                                    error: Some(e.to_string()),
                                },
                            };

                            let event = if result.error.is_some() {
                                ProgressEvent::ImageError {
                                    image: img_path.clone(),
                                    index: idx,
                                    total,
                                    error: result.error.clone().unwrap(),
                                }
                            } else {
                                ProgressEvent::ImageDone {
                                    image: img_path.clone(),
                                    index: idx,
                                    total,
                                    count: result.count,
                                    duration_s: result.duration_s,
                                }
                            };
                            let _ = app_clone.emit("inference_progress", event);

                            results_mutex.lock().unwrap().push((idx, result));
                        });
                    });

                    let mut indexed: Vec<(usize, ImageResult)> = results_mutex.lock().unwrap().drain(..).collect();
                    indexed.sort_by_key(|(i, _)| *i);
                    all_results = indexed.into_iter().map(|(_, r)| r).collect();
                } else {
                    // GPU: single session, sequential
                    for (idx, img_path) in images.iter().enumerate() {
                        if cancel.load(Ordering::Relaxed) { break; }

                        let _ = app_clone.emit("inference_progress", ProgressEvent::ImageStart {
                            image: img_path.clone(),
                            index: idx,
                            total,
                        });

                        let result = process_image_onnx(img_path, &mut session, &config);

                        let event = if result.error.is_some() {
                            ProgressEvent::ImageError {
                                image: img_path.clone(),
                                index: idx,
                                total,
                                error: result.error.clone().unwrap(),
                            }
                        } else {
                            ProgressEvent::ImageDone {
                                image: img_path.clone(),
                                index: idx,
                                total,
                                count: result.count,
                                duration_s: result.duration_s,
                            }
                        };
                        let _ = app_clone.emit("inference_progress", event);
                        all_results.push(result);
                    }
                }
            }

            _ => {
                let _ = app_clone.emit("inference_error", format!("Unknown backend: {}", config.backend));
                return;
            }
        }

        let total_plaques: usize = all_results.iter().map(|r| r.count).sum();
        let duration_s = batch_start.elapsed().as_secs_f64();

        *results_store.lock().unwrap() = all_results.clone();

        if cancel.load(Ordering::Relaxed) {
            let _ = app_clone.emit("inference_progress", ProgressEvent::Cancelled);
        } else {
            let _ = app_clone.emit("inference_progress", ProgressEvent::Done {
                results: all_results,
                total_plaques,
                duration_s,
            });
        }
    });

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// App entry point
// ──────────────────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            run_inference,
            cancel_inference,
            scan_models,
            select_images,
            select_folder,
            select_models_folder,
            get_images_in_folder,
            save_results_json,
            save_results_csv,
            open_in_explorer,
            read_image_preview,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
