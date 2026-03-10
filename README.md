# 🎯 Student Attention Score Prediction System

A real-time student attention monitoring system that uses **computer vision** (MediaPipe) and a **deep learning regression model** (TensorFlow/Keras) to predict how attentive a student is during a class session.

The system follows a **hybrid client-server architecture**: heavy CV processing (face/body detection, head pose estimation) runs on the client side, while model inference, gaze penalty logic, and session tracking run on a lightweight FastAPI backend.

---

## 📁 Project Structure

```
FinalProjct/
├── api/
│   ├── __init__.py
│   └── app.py                  # FastAPI backend (model inference + sessions)
├── models/
│   ├── face_landmarker.task    # MediaPipe face model (auto-downloaded)
│   └── pose_landmarker_lite.task  # MediaPipe pose model (auto-downloaded)
├── FinalModel_MultiSubject_Regression.ipynb  # Training notebook
├── final_multisubject_regression_model.keras # Trained Keras model
├── final_scaler.pkl            # Fitted StandardScaler
├── final_production_config.pkl # Feature columns + median values
├── final_memory.pkl            # Training memory pool (7500 samples)
├── streamlit_app.py            # Streamlit frontend (test/demo UI)
├── requirements.txt            # Python dependencies
├── sessions.json               # Saved session summaries (auto-created)
└── README.md
```

---

## 🗂️ About the Dataset

### Source Data

The model was trained on data collected from **50 subjects** in a classroom-like setting. For each subject, video frames were captured and annotated with attention labels.

### Features Extracted (per frame)

From each video frame, **6 base features** are extracted using computer vision:

| Feature | Description |
|---------|-------------|
| `face_area` | Bounding-box area of the detected face (pixels²). Larger = closer to camera. |
| `body_area` | Bounding-box area of the detected body (pixels²). |
| `pitch` | Head pitch angle in degrees. Negative = looking down, Positive = looking up. |
| `yaw` | Head yaw angle in degrees. 0 = facing camera, ±90 = turned sideways. |
| `roll` | Head roll (tilt) angle in degrees. 0 = upright. |
| `pose_vis_mean` | Mean visibility score (0–1) of all body keypoints detected by the pose estimator. |

### Derived Features

From the 6 base features, **5 additional features** are computed during the pipeline:

| Derived Feature | How It's Computed |
|-----------------|-------------------|
| `face_detected` | Binary flag: 1 if a face is present, 0 otherwise. |
| `face_body_ratio` | `face_area / body_area` — proxy for distance/engagement. |
| `head_movement` | `√(pitch² + yaw² + roll²)` — magnitude of head deviation. |
| `abs_yaw` | Absolute value of yaw — direction-agnostic lateral turn. |
| `abs_pitch` | Absolute value of pitch — direction-agnostic vertical tilt. |

This gives a total of **11 features** fed into the model.

### Labels

Each frame was labeled on a **1–5 scale**:

| Score | Meaning |
|-------|---------|
| 1 | Fully attentive |
| 2 | Mostly attentive |
| 3 | Neutral / moderate |
| 4 | Mostly inattentive |
| 5 | Fully inattentive |

The training data distribution was: score 2 (36.1%), score 3 (40.2%), score 4 (18.9%), score 5 (4.7%), score 1 (0.1%).

---

## 🧠 Model Architecture

The model is a **fully-connected regression neural network** built with TensorFlow/Keras:

```
Input (11 features)
  → Dense(128, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(64, ReLU) + BatchNorm + Dropout(0.3)
  → Dense(32, ReLU) + BatchNorm + Dropout(0.2)
  → Dense(1, Linear)   →  output: predicted score (1–5)
```

### Training Strategy

- **Progressive learning**: The model was trained incrementally across all 50 subjects, with a memory retention pool of 150 samples per subject (7500 total) to prevent catastrophic forgetting.
- **StandardScaler**: All 11 features are standardized using a fitted `StandardScaler` saved as `final_scaler.pkl`.
- **Feature medians**: Saved in `final_production_config.pkl` for imputing missing values at inference time (e.g., when no face is detected).

### Post-Processing

The raw model output (1–5) is converted to an attention percentage:

```
attention_% = ((5 - raw_score) / 4) × 100
```

So **1 → 100%** (fully attentive) and **5 → 0%** (fully inattentive).

Two additional penalty mechanisms are applied:

1. **Gaze-away penalty** (per-frame): Subtracts percentage points when the student is looking sideways or up (away from screen). Uses a quadratic ramp with dead zones (|yaw| ≤ 25° and |pitch| ≤ 15° = no penalty). Looking **down** (writing notes) is exempt from this penalty.

2. **Looking-down session penalty** (end-of-session): If the student spent more than **30%** of the session looking down, a proportional penalty is applied to the session average (0.5 pp per % above 30%, max 25 pp).

---

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- Webcam (for live testing)

### 1. Install Dependencies

```bash
# Create and activate a conda environment (recommended)
conda create -n aienv python=3.10 -y
conda activate aienv

# Install all dependencies
pip install -r requirements.txt
```

### 2. Start the API Backend

The API serves the trained model and handles predictions, session tracking, and persistence.

```bash
uvicorn api.app:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`. You can check the interactive docs at `http://127.0.0.1:8000/docs`.

### 3. Start the Streamlit Test UI

In a **separate terminal** (with the same conda env activated):

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## 🖥️ Using the Application

### Streamlit Test UI (`streamlit_app.py`)

The Streamlit app is a **testing and demo interface** with 4 tabs:

| Tab | Purpose |
|-----|---------|
| 📹 **Live Webcam** | Opens your webcam, runs MediaPipe locally for face/body detection, sends extracted features to the API every 1.5s. Starts a tracked session — when you stop, it shows the session summary with average scores and penalties. |
| 📋 **Manual Input** | Enter the 6 features by hand or use quick presets (Attentive, Writing Notes, Looking Away, etc.) to test specific scenarios. |
| 📦 **Batch Prediction** | Paste a JSON array of samples for bulk scoring. |
| 📊 **Session History** | View all past webcam sessions with their average attention scores, looking-down percentages, and penalties. |

### API Endpoints (`api/app.py`)

This is the **main backend** that would be integrated into your production frontend (e.g., Jitsi Meet).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — model loaded, active sessions count |
| `POST` | `/predict` | Stateless prediction for one or more samples |
| `POST` | `/session/start` | Start a new tracked session (returns `session_id`) |
| `POST` | `/session/sample` | Send one frame's features during a session |
| `POST` | `/session/end` | End session — returns summary with averages + penalties |
| `GET` | `/sessions` | List all previously saved session summaries |
| `POST` | `/reload` | Hot-reload model files without restarting the server |

#### Example: Stateless Prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"face_area": 25000, "body_area": 280000, "pitch": -10, "yaw": -2, "roll": 0, "pose_vis_mean": 0.85}]}'
```

Response:
```json
{
  "predictions": [{
    "raw_score": 1.63,
    "percentage": 84.2,
    "gaze_penalty": 0.0,
    "looking_down": false
  }]
}
```

#### Example: Session Workflow

```bash
# 1. Start session
curl -X POST http://127.0.0.1:8000/session/start

# 2. Send samples (use the session_id from step 1)
curl -X POST http://127.0.0.1:8000/session/sample \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID", "features": {"face_area": 25000, "body_area": 280000, "pitch": -10, "yaw": -2, "roll": 0, "pose_vis_mean": 0.85}}'

# 3. End session (get summary)
curl -X POST http://127.0.0.1:8000/session/end \
  -H "Content-Type: application/json" \
  -d '{"session_id": "YOUR_SESSION_ID"}'
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────┐
│          Client (Browser)           │
│                                     │
│  Webcam → MediaPipe Face/Pose       │
│  → Extract 6 features (head pose,   │
│    face area, body area, visibility) │
│  → Send features to API             │
└──────────────┬──────────────────────┘
               │  HTTP POST (6 numbers)
               ▼
┌─────────────────────────────────────┐
│        FastAPI Backend (API)        │
│                                     │
│  → Compute derived features (5)     │
│  → StandardScaler transform         │
│  → Keras model inference            │
│  → Gaze penalty post-processing     │
│  → Session tracking + persistence   │
│  → Return prediction + metadata     │
└─────────────────────────────────────┘
```

**Why hybrid?** MediaPipe (face/body detection) is CPU-intensive but runs efficiently on the client. The model inference and business logic run on the server, keeping the network payload minimal (just 6 numbers per frame) and offloading the ML model from student laptops.

---

## 📄 Key Files Explained

| File | Role |
|------|------|
| `api/app.py` | **Main API** — loads model at startup, serves predictions, manages sessions, applies gaze penalties, saves session history to `sessions.json`. |
| `streamlit_app.py` | **Test UI** — opens webcam, runs MediaPipe, calls the API, displays results. Used for testing and demonstration only. |
| `FinalModel_MultiSubject_Regression.ipynb` | **Training notebook** — full pipeline: data loading, feature engineering, progressive multi-subject training, model evaluation, and artefact export. |
| `final_multisubject_regression_model.keras` | Trained Keras model (128→64→32→1 regression). |
| `final_scaler.pkl` | Fitted `StandardScaler` for the 11 features. |
| `final_production_config.pkl` | Feature column names, base column names, and median values for imputation. |
| `final_memory.pkl` | Training memory pool (7500 samples from 50 subjects) — can be used to retrain the model. |
| `requirements.txt` | All Python package dependencies. |
