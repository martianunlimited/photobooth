# 🎃 AI Zombie Photobooth

A decoupled, zero-disk-write Generative AI Photobooth system that transforms attendee photos into photorealistic undead zombies using Stable Diffusion, ControlNet InstructPix2Pix, and DPM-Solver++.

---

## 🏗 System Architecture (Photobooth V2)

The system is decoupled into two independent, fault-tolerant tiers:

```
[Attendee Face / USB Foot Pedal (Ctrl+Alt+P)]
                    │
                    ▼
   [Local HTML5 / MediaPipe Kiosk UI]
   - Offline WASM BlazeFace Frontal Tracker
   - 3-Second Cooldown & Auto-Stabilization
                    │
                    ▼  (Multipart in-memory RAM stream)
   [Node.js Express Kiosk Proxy (Port 3000)]
   - multer.memoryStorage(): Pure RAM buffers, ZERO disk writes
   - Native Fetch streaming to GPU backend
                    │
                    ▼  (HTTP POST /zombify)
   [FastAPI GPU Inference Backend (Port 8000)]
   - ControlNet InstructPix2Pix (lllyasviel/control_v11e_sd15_ip2p) + Full VAE
   - DPMSolverMultistepScheduler (15 total steps, 9 actual denoising steps at strength=0.60)
   - Additive Gaussian Noise (σ=15) into ControlNet conditioning input
   - Sub-second inference latency (~995 ms on RTX 3090 / RTX 3070)
```

---

## 🚀 Quickstart Launch Guide

### 1. Start the GPU Inference Backend
On Windows:
```cmd
photobooth_v2\start_gpu_backend.bat
```
On Linux / macOS:
```bash
./photobooth_v2/start_gpu_backend.sh
```
*Listens on `http://0.0.0.0:8000`. Warms up CUDA kernels automatically on launch.*

### 2. Start the Node.js Kiosk Proxy
On Windows:
```cmd
photobooth_v2\start_booth_nodejs.bat
```
On Linux / macOS:
```bash
./photobooth_v2/start_booth_nodejs.sh
```
*Listens on `http://127.0.0.1:3000`.*

### 3. Open the Kiosk
Navigate in your browser to:
👉 **[http://127.0.0.1:3000](http://127.0.0.1:3000)**

---

## 📸 Trigger Modes

1. **Auto-Face Trigger (Default):** Stand in front of the webcam. MediaPipe BlazeFace tracks face proximity ($\ge 14\%$) and frontal gaze centering ($[0.32, 0.68]$). Once stabilized for 1.0 second, a countdown (`3... 2... 1... 📸 Snapping!`) fires automatically, followed by a 3-second cooldown.
2. **USB Foot Pedal (`Ctrl + Alt + P`):** Pressing the USB foot pedal immediately triggers an instant capture and transformation.
3. **Manual Button:** Click the on-screen **"🧟 Zombify Me!"** button.

---

## 📊 Benchmark & Quality Metrics

Verified on an NVIDIA GeForce RTX 3090:
- **Mean Inference Latency:** `995.31 ms` (Target: `< 3500 ms`)
- **Total Proxy Roundtrip:** `1075.45 ms`
- **UNet Denoising Steps:** `9` (15 scheduler steps @ `strength=0.60`)
- **Guidance Scale (CFG):** `6.0`
- **ControlNet Noise Sigma:** `15.0`
- **Disk Writes During Streaming:** `0 bytes` (Pure RAM buffer streaming)

---

## 🧪 Testing & Verification

Run the automated test suite and linters:
```bash
# Run pytest verification suite
pytest photobooth_v2/tests -q

# Run Ruff linter and formatter
ruff check photobooth_v2
ruff format photobooth_v2 --check
```

---

## 📁 Repository Structure

```
photobooth/
├── photobooth_v2/                     # Decoupled V2 Architecture
│   ├── configs/
│   │   └── system_manifest.json      # Deterministic hardware and hyperparameter manifest
│   ├── artifacts/                    # Intermediate persistence and benchmark metrics
│   │   ├── test_input_640x480.jpg
│   │   ├── test_noisy_control_682x512.png
│   │   ├── test_output_zombie.jpg
│   │   └── quality_and_latency_metrics.json
│   ├── logs/                         # Phase execution and audit verification logs
│   │   ├── phase1_env.log
│   │   ├── phase2_backend_bench.log
│   │   ├── phase3_nodejs_frontend.log
│   │   └── phase4_e2e_verification.log
│   ├── backend/                      # FastAPI GPU service
│   │   ├── config.py                 # pydantic-settings configuration
│   │   ├── pipeline.py               # InstructPix2Pix + DPM-Solver pipeline
│   │   ├── server.py                 # FastAPI endpoints (/health, /zombify)
│   │   └── benchmark_quality.py      # Latency & VRAM benchmark runner
│   ├── frontend_node/                # Node.js Express proxy & kiosk UI
│   │   ├── package.json
│   │   ├── server.js                 # In-memory multer proxy (zero disk writes)
│   │   └── public/
│   │       ├── index.html            # Halloween kiosk UI with MediaPipe WASM
│   │       ├── image.gif             # Stable Diffusion explainer diagram
│   │       └── vendor/               # Offline BlazeFace & WASM binaries
│   ├── tests/
│   │   └── test_photobooth.py        # System contract and artifact test suite
│   ├── start_gpu_backend.bat / .sh   # GPU backend startup scripts
│   └── start_booth_nodejs.bat / .sh  # Kiosk proxy startup scripts
│
├── photobooth_api.py                 # Legacy V1 photobooth script
├── footSwitch.py                     # Legacy V1 foot switch listener
├── arduino.ino                       # Legacy V1 Arduino foot pedal firmware
└── requirements.txt                  # Legacy V1 requirements
```
