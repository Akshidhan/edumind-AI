# Attention Score Prediction – Copilot Instructions

## Project Overview
This is a **Student Attention Score Prediction** system with two runtime components:

- **FastAPI backend** (`api/app.py`) — loads the Keras model and serves REST predictions + session tracking on port **8000**
- **Streamlit frontend** (`streamlit_app.py`) — CV pipeline (MediaPipe face/pose) + UI that calls the backend at `http://127.0.0.1:8000`

---

## How to Run the Application

### 1. Install Dependencies
Run once from the project root:
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
**Always start the backend first.** Run from the project root:
```bash
uvicorn api.app:app --reload --port 8000
```
- API will be live at: `http://127.0.0.1:8000`
- Interactive docs at: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 3. Start the Streamlit Frontend
In a **second terminal**, from the project root:
```bash
streamlit run streamlit_app.py
```
- UI will open at: `http://localhost:8501`

> Both processes must run simultaneously. The Streamlit app will show a warning if the backend is unreachable.

---

## Architecture Notes
- The backend (`api/app.py`) expects these artefact files in the project root at startup:
  - `final_multisubject_regression_model.keras`
  - `final_scaler.pkl`
  - `final_production_config.pkl`
- MediaPipe model files (`models/face_landmarker.task`, `models/pose_landmarker_lite.task`) are auto-downloaded by the Streamlit app if missing.
- Session data is persisted to `sessions.json` (auto-created).

## Key API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | One-shot attention prediction |
| `POST` | `/session/start` | Begin a tracked webcam session |
| `POST` | `/session/sample` | Send a frame sample during a session |
| `POST` | `/session/end` | End session and get summary |
| `GET`  | `/sessions` | List all past sessions |
| `GET`  | `/health` | Health check |
| `POST` | `/reload` | Hot-reload model artefacts |
