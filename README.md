# PlaqSeg Desktop

> **Tiled YOLO plaque segmentation — native Rust desktop application.**  
> Built with Tauri 2 + SvelteKit 5. Runs on Windows and Linux. No Python required at runtime.

[![GitHub Actions CI](https://github.com/your-username/plaqseg-desktop/actions/workflows/release.yml/badge.svg)](https://github.com/your-username/plaqseg-desktop/actions)

---

## Features

| Feature | Details |
|---|---|
| **Native Rust inference** | Zero Python dependency at runtime - ONNX Runtime + TFLite crates |
| **6 model variants** | nano/small × ONNX (GPU/CPU) and float16/float32 TFLite (CPU) |
| **Tiling** | 1280 px tiles with 1024 px stride — handles high-res images |
| **Multi-worker batches** | Rayon thread pool — set workers to match your CPU core count |
| **GPU support** | CUDA and DirectML (Windows) via ONNX Runtime execution providers |
| **Drag & drop** | Drop single images or whole folders onto the queue |
| **Real-time progress** | Per-image status, live plaque count, and elapsed timer |
| **Export** | JSON (full detections) and CSV (one row per detection) |
| **Dark / Light theme** | Persisted across sessions |

---

## Model Variants

| ID | File | Backend | GPU | Description |
|---|---|---|---|---|
| `nano` | `nano.onnx` | ONNX Runtime | ✅ CUDA / DirectML | Fastest — best for large batches |
| `small` | `small.onnx` | ONNX Runtime | ✅ CUDA / DirectML | Higher accuracy, moderate speed |
| `nano-float32` | `nano_float32.tflite` | TFLite | ❌ CPU only | Highest TFLite accuracy, largest file |
| `small-float32` | `small_float32.tflite` | TFLite | ❌ CPU only | Best TFLite accuracy |
| `nano-float16` | `nano_float16.tflite` | TFLite | ❌ CPU only | Fast & lightweight |
| `small-float16` | `small_float16.tflite` | TFLite | ❌ CPU only | Good accuracy–size balance |

---

## Installation

Download the latest release from the GitHub Releases page:

- **Windows**: Run the `.msi` installer or the `.exe` NSIS installer
- **Linux**: Run the `.AppImage` directly or install the `.deb` package

---

## Usage

1. **Select a model** — click any model card in the left panel
2. **Choose device** — CPU, CUDA, or DirectML (GPU options disabled for TFLite)
3. **Set workers** — drag the slider (CPU mode: more workers = faster batches)
4. **Add images** — drag & drop files/folders, or use the *Add Images* / *Add Folder* buttons
5. **Run** — click the **Run Inference** button
6. **Export** — use *Export JSON* or *Export CSV* when done

### Advanced settings

Expand **Advanced — Tiling & Thresholds** to tune:
- **Tile size** (default: 1280 px) — larger tiles cover more area per inference pass
- **Stride** (default: 1024 px) — overlap = tile size − stride
- **Confidence** (default: 0.15) — lower = more detections, higher = fewer false positives
- **NMS IoU** (default: 0.50) — cross-tile non-maximum suppression threshold

### Petri dish crop

Expand **Petri Dish Crop** to auto-detect and crop to the dish boundary using Hough Circle Transform before tiling. Useful for plate images with large blank backgrounds.

---

## Performance Tips

| Scenario | Recommendation |
|---|---|
| Large PC (≥8 cores, no GPU) | `nano` or `small` ONNX, workers = CPU core count |
| PC with NVIDIA GPU | `nano` or `small` ONNX, device = CUDA |
| PC with AMD/Intel GPU | `nano` or `small` ONNX, device = DirectML |
| Laptop (battery saver) | `nano-float16` TFLite, workers = 2–4 |
| Maximum accuracy | `small` ONNX or `small-float32` TFLite |

---

## Development

### Prerequisites

- [Rust](https://rustup.rs/) (stable, 1.76+)
- [Node.js](https://nodejs.org/) (20+)
- [Tauri CLI](https://tauri.app/start/prerequisites/)

```bash
# Install JS dependencies
npm install

# Start Tauri dev mode (hot-reload Svelte + Rust)
npm run tauri dev

# Type-check Svelte
npm run check

# Build release installer
npm run tauri build
```

### Project Structure

```
PlaqSegDesktop/
├── src/                        # Svelte frontend
│   ├── app.html                # HTML shell
│   ├── app.css                 # Design system
│   ├── lib/
│   │   ├── types.ts            # TypeScript types (mirrors Rust structs)
│   │   └── invoke.ts           # Typed Tauri command wrappers
│   └── routes/
│       └── +page.svelte        # Main app page (3-column layout)
├── src-tauri/                  # Rust backend
│   ├── src/
│   │   ├── lib.rs              # Inference engine + Tauri commands
│   │   └── main.rs             # Entry point
│   ├── Cargo.toml              # Rust deps (ort, tflite, rayon, image)
│   └── tauri.conf.json         # Window + bundle config
├── models/                     # Model files (.onnx + .tflite)
├── scripts/
│   └── export_models.py        # PyTorch → ONNX export helper
└── .gitlab-ci.yml              # CI/CD pipeline
```

## License

MIT — see [LICENSE](LICENSE).
