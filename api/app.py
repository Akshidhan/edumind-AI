"""
FastAPI backend for Attention Score Prediction.

Loads the trained Keras model, StandardScaler, and production config once at
startup, then serves predictions via a REST endpoint.

Endpoints
---------
  POST /predict         – stateless one-shot prediction
  POST /session/start   – begin a tracked webcam session
  POST /session/sample  – send a sample during a session (returns prediction)
  POST /session/end     – end session, get summary + averages
  GET  /sessions        – list past sessions
  GET  /health          – health check
  POST /reload          – hot-reload model artefacts

Run (from the project root):
    uvicorn api.app:app --reload --port 8000
"""

import json as _json
import os
import math
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# TensorFlow – silence info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras  # noqa: E402

# ---------------------------------------------------------------------------
# Paths  (all artefacts sit in the project root, one level above api/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "final_multisubject_regression_model.keras"
SCALER_PATH = PROJECT_ROOT / "final_scaler.pkl"
CONFIG_PATH = PROJECT_ROOT / "final_production_config.pkl"
SESSIONS_PATH = PROJECT_ROOT / "sessions.json"

# ---------------------------------------------------------------------------
# Feature engineering  (exact replica of the notebook pipeline)
# ---------------------------------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features – must match training pipeline exactly."""
    if "face_detected" not in df.columns:
        df["face_detected"] = (
            (df["face_area"].notna()) & (df["face_area"] > 0)
        ).astype(float)

    df["face_body_ratio"] = np.where(
        (df["face_area"].notna()) & (df["body_area"].notna()) & (df["body_area"] > 0),
        df["face_area"].fillna(0) / df["body_area"],
        0.0,
    )

    pitch_vals = df["pitch"].fillna(0.0)
    yaw_vals = df["yaw"].fillna(0.0)
    roll_vals = df["roll"].fillna(0.0)
    df["head_movement"] = np.sqrt(pitch_vals**2 + yaw_vals**2 + roll_vals**2)
    df.loc[df["face_detected"] == 0.0, "head_movement"] = 0.0

    df["abs_yaw"] = df["yaw"].fillna(0.0).abs()
    df["abs_pitch"] = df["pitch"].fillna(0.0).abs()
    df.loc[df["face_detected"] == 0.0, "abs_yaw"] = 0.0
    df.loc[df["face_detected"] == 0.0, "abs_pitch"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Load artefacts at startup (graceful if not yet available)
# ---------------------------------------------------------------------------
model = None
scaler = None
production_config = None
FEATURE_COLS: List[str] = []
BASE_COLS: List[str] = ["face_area", "body_area", "pitch", "yaw", "roll", "pose_vis_mean"]
FEAT_MEDIANS: dict = {}


def _load_artefacts():
    """Load model, scaler, and production config from disk."""
    global model, scaler, production_config, FEATURE_COLS, BASE_COLS, FEAT_MEDIANS

    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not CONFIG_PATH.exists():
        missing = [p.name for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH] if not p.exists()]
        print(f"⚠️  Missing artefact(s): {missing}")
        print("   Train the model first by running the notebook, then restart this server.")
        return False

    model = keras.models.load_model(str(MODEL_PATH))
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(CONFIG_PATH, "rb") as f:
        production_config = pickle.load(f)

    FEATURE_COLS.clear()
    FEATURE_COLS.extend(production_config["feature_cols"])
    BASE_COLS.clear()
    BASE_COLS.extend(production_config["base_feature_cols"])
    FEAT_MEDIANS.clear()
    FEAT_MEDIANS.update(production_config["feature_medians"])

    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"✅ Scaler loaded from {SCALER_PATH}")
    print(f"✅ Config loaded – {len(FEATURE_COLS)} features")
    return True


artefacts_loaded = _load_artefacts()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class FeatureInput(BaseModel):
    """A single observation's raw features (6 base features)."""
    face_area: Optional[float] = Field(None, description="Detected face bounding-box area (use null if no face)")
    body_area: Optional[float] = Field(None, description="Detected body bounding-box area")
    pitch: Optional[float] = Field(None, description="Head pitch angle in degrees")
    yaw: Optional[float] = Field(None, description="Head yaw angle in degrees")
    roll: Optional[float] = Field(None, description="Head roll angle in degrees")
    pose_vis_mean: Optional[float] = Field(None, description="Mean body-keypoint visibility score")


class PredictionRequest(BaseModel):
    """One or more observations to score."""
    samples: List[FeatureInput]


class SampleResult(BaseModel):
    raw_score: float = Field(..., description="Attention score on 1-5 scale (1=attentive, 5=inattentive)")
    percentage: float = Field(..., description="Attention percentage 0-100% (100%=fully attentive), after gaze penalty")
    gaze_penalty: float = Field(0.0, description="Percentage-points subtracted for looking away (0 = looking at screen)")
    looking_down: bool = Field(False, description="True when student is looking down (writing notes)")


class PredictionResponse(BaseModel):
    predictions: List[SampleResult]


# ---------------------------------------------------------------------------
# Post-process: gaze-away penalty  (v2 – wider dead-zone, no penalty for
# looking DOWN because students write notes)
# ---------------------------------------------------------------------------
LOOKING_DOWN_PITCH_THRESHOLD = -18.0  # pitch below this = "looking down at notes"

def _gaze_penalty(signed_yaw: float, signed_pitch: float,
                  face_detected: float) -> tuple:
    """
    Compute an attention-percentage penalty based on how far the user's
    gaze deviates from the screen.

    Returns (penalty_pp, is_looking_down).

    Rules
    -----
    * No face detected                    → 40 pp penalty
    * Looking DOWN (pitch < -18°)         → 0 pp  (writing notes – allowed)
    * |yaw| ≤ 25° AND upward_pitch ≤ 15° → 0 pp  (looking at screen)
    * Beyond the dead-zone               → quadratic ramp
    """
    if face_detected < 0.5:
        return 40.0, False

    abs_yaw = abs(signed_yaw)

    # Looking down at notes – no real-time penalty
    is_looking_down = signed_pitch < LOOKING_DOWN_PITCH_THRESHOLD
    if is_looking_down:
        # Slight yaw penalty may still apply if they're turned sideways AND
        # looking down (not at their own desk)
        YAW_DEAD = 30.0
        if abs_yaw <= YAW_DEAD:
            return 0.0, True
        yaw_ratio = min((abs_yaw - YAW_DEAD) / (90.0 - YAW_DEAD), 1.0)
        return min((yaw_ratio ** 2) * 40.0, 40.0), True

    # --- Normal gaze penalty (only yaw + upward pitch) ---
    YAW_DEAD    = 25.0   # degrees – generous for natural head position
    PITCH_DEAD  = 15.0   # only for UPWARD pitch (positive)
    MAX_YAW_P   = 50.0
    MAX_PITCH_P = 30.0
    MAX_TOTAL   = 70.0

    yaw_excess   = max(0.0, abs_yaw - YAW_DEAD)
    # Only penalise upward pitch (looking up / away from screen), not downward
    upward_pitch = max(0.0, signed_pitch)
    pitch_excess = max(0.0, upward_pitch - PITCH_DEAD)

    yaw_ratio   = min(yaw_excess   / (90.0 - YAW_DEAD),   1.0)
    pitch_ratio = min(pitch_excess / (90.0 - PITCH_DEAD),  1.0)

    yaw_pen   = (yaw_ratio   ** 2) * MAX_YAW_P
    pitch_pen = (pitch_ratio ** 2) * MAX_PITCH_P

    return min(yaw_pen + pitch_pen, MAX_TOTAL), False


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------
def predict(samples: List[FeatureInput]) -> List[SampleResult]:
    """Run the full inference pipeline on a batch of samples."""
    if model is None or scaler is None:
        raise RuntimeError(
            "Model artefacts not loaded. Train the model first by running the notebook, "
            "then restart this server."
        )

    # Build DataFrame from raw inputs
    rows = []
    for s in samples:
        rows.append({
            "face_area": s.face_area if s.face_area is not None else np.nan,
            "body_area": s.body_area if s.body_area is not None else np.nan,
            "pitch": s.pitch if s.pitch is not None else np.nan,
            "yaw": s.yaw if s.yaw is not None else np.nan,
            "roll": s.roll if s.roll is not None else np.nan,
            "pose_vis_mean": s.pose_vis_mean if s.pose_vis_mean is not None else np.nan,
        })
    df = pd.DataFrame(rows, columns=BASE_COLS)

    # Compute face_detected
    df["face_detected"] = ((df["face_area"].notna()) & (df["face_area"] > 0)).astype(float)

    # Derived features (same pipeline as training)
    df = add_derived_features(df)

    # Impute NaNs with saved training medians
    for col in FEATURE_COLS:
        median_val = FEAT_MEDIANS.get(col)
        if median_val is not None and not (isinstance(median_val, float) and math.isnan(median_val)):
            df[col] = df[col].fillna(median_val)
    df = df.fillna(0.0)

    # Select & order columns, scale, predict
    X = df[FEATURE_COLS].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)

    raw_scores = model.predict(X_scaled, verbose=0).flatten()
    raw_scores = np.clip(raw_scores, 1.0, 5.0)

    # In the training data: 1 = fully attentive, 5 = not attentive
    # So attention % is INVERTED:  1 → 100%,  5 → 0%
    base_percentages = ((5.0 - raw_scores) / 4) * 100

    # ── Gaze-away penalty (post-process) ──────────────────────
    results = []
    for i, (raw, base_pct) in enumerate(zip(raw_scores, base_percentages)):
        signed_yaw   = float(df.iloc[i].get("yaw",   0.0))
        signed_pitch = float(df.iloc[i].get("pitch", 0.0))
        face_det     = float(df.iloc[i].get("face_detected", 0.0))

        penalty, looking_down = _gaze_penalty(signed_yaw, signed_pitch, face_det)
        adjusted = max(0.0, float(base_pct) - penalty)

        results.append(SampleResult(
            raw_score=round(float(raw), 4),
            percentage=round(adjusted, 2),
            gaze_penalty=round(penalty, 2),
            looking_down=looking_down,
        ))
    return results


# ---------------------------------------------------------------------------
# Session management  (in-memory store + JSON persistence)
# ---------------------------------------------------------------------------
_sessions: Dict[str, dict] = {}   # session_id → session data


def _load_sessions_file() -> list:
    """Load past sessions from JSON file."""
    if SESSIONS_PATH.exists():
        try:
            return _json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_session_to_file(summary: dict):
    """Append a completed session summary to sessions.json."""
    history = _load_sessions_file()
    history.append(summary)
    # Ensure the directory exists before writing
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_PATH.write_text(
        _json.dumps(history, indent=2, default=str), encoding="utf-8"
    )


class SessionStartResponse(BaseModel):
    session_id: str
    started_at: str


class SessionSampleRequest(BaseModel):
    session_id: str
    features: FeatureInput


class SessionSampleResponse(BaseModel):
    prediction: SampleResult
    samples_count: int
    session_duration_sec: float
    looking_down_pct: float = Field(
        ..., description="Percentage of session samples where student was looking down"
    )


class SessionEndRequest(BaseModel):
    session_id: str


class SessionSummary(BaseModel):
    session_id: str
    started_at: str
    ended_at: str
    duration_sec: float
    total_samples: int
    avg_raw_score: float
    avg_attention_pct: float
    looking_down_pct: float
    looking_down_penalty: float = Field(
        0.0,
        description="Extra penalty applied because student looked down > 30% of session",
    )
    final_avg_attention_pct: float = Field(
        ..., description="Average attention after looking-down session penalty"
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Attention Score Prediction API",
    description="Predict student attention levels (1-5 / 0-100%) from face & pose features.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health-check endpoint."""
    return {
        "status": "ok" if artefacts_loaded else "model_not_loaded",
        "model_loaded": model is not None,
        "features": FEATURE_COLS if FEATURE_COLS else None,
        "model_path": str(MODEL_PATH),
        "artefacts_exist": {
            "model": MODEL_PATH.exists(),
            "scaler": SCALER_PATH.exists(),
            "config": CONFIG_PATH.exists(),
        },
        "active_sessions": len(_sessions),
    }


# ── Stateless prediction ──────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """
    Predict attention scores for one or more samples (stateless).

    Send 6 base features per sample:
    `face_area`, `body_area`, `pitch`, `yaw`, `roll`, `pose_vis_mean`.
    Use `null` for any feature that is unavailable (e.g. no face detected).
    """
    if not request.samples:
        raise HTTPException(status_code=400, detail="No samples provided")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model by running the notebook first, then restart the server.",
        )
    try:
        results = predict(request.samples)
        return PredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Session-based prediction ──────────────────────────────────────────
@app.post("/session/start", response_model=SessionStartResponse)
def session_start():
    """Start a new tracked webcam session."""
    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _sessions[sid] = {
        "started_at": now,
        "samples": [],          # list of SampleResult dicts
        "looking_down_count": 0,
        "total_count": 0,
    }
    return SessionStartResponse(session_id=sid, started_at=now)


@app.post("/session/sample", response_model=SessionSampleResponse)
def session_sample(req: SessionSampleRequest):
    """
    Send one sample during an active session.
    Returns the prediction AND running session stats.
    """
    sess = _sessions.get(req.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start first.")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = predict([req.features])
    pred = results[0]

    # Track
    sess["samples"].append(pred.model_dump())
    sess["total_count"] += 1
    if pred.looking_down:
        sess["looking_down_count"] += 1

    started = datetime.fromisoformat(sess["started_at"])
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    ld_pct = (sess["looking_down_count"] / sess["total_count"]) * 100

    return SessionSampleResponse(
        prediction=pred,
        samples_count=sess["total_count"],
        session_duration_sec=round(duration, 1),
        looking_down_pct=round(ld_pct, 1),
    )


@app.post("/session/end", response_model=SessionSummary)
def session_end(req: SessionEndRequest):
    """
    End a session.  Returns the full summary including the
    looking-down penalty (if the student looked down > 30% of the time).
    """
    sess = _sessions.pop(req.session_id, None)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found or already ended.")

    now = datetime.now(timezone.utc).isoformat()
    started = datetime.fromisoformat(sess["started_at"])
    duration = (datetime.now(timezone.utc) - started).total_seconds()

    samples = sess["samples"]
    n = len(samples)
    if n == 0:
        summary = SessionSummary(
            session_id=req.session_id,
            started_at=sess["started_at"],
            ended_at=now,
            duration_sec=round(duration, 1),
            total_samples=0,
            avg_raw_score=0.0,
            avg_attention_pct=0.0,
            looking_down_pct=0.0,
            looking_down_penalty=0.0,
            final_avg_attention_pct=0.0,
        )
        _save_session_to_file(summary.model_dump())
        return summary

    avg_raw  = sum(s["raw_score"]  for s in samples) / n
    avg_pct  = sum(s["percentage"] for s in samples) / n
    ld_pct   = (sess["looking_down_count"] / n) * 100

    # ── Looking-down session penalty ──────────────────────────
    # If the student spent > 30 % of samples looking down, apply a
    # proportional penalty:  each % above 30 costs 0.5 pp  (max 25 pp)
    ld_penalty = 0.0
    if ld_pct > 30.0:
        excess = ld_pct - 30.0            # how much above the 30 % threshold
        ld_penalty = min(excess * 0.5, 25.0)

    final_avg = max(0.0, avg_pct - ld_penalty)

    summary = SessionSummary(
        session_id=req.session_id,
        started_at=sess["started_at"],
        ended_at=now,
        duration_sec=round(duration, 1),
        total_samples=n,
        avg_raw_score=round(avg_raw, 2),
        avg_attention_pct=round(avg_pct, 2),
        looking_down_pct=round(ld_pct, 1),
        looking_down_penalty=round(ld_penalty, 2),
        final_avg_attention_pct=round(final_avg, 2),
    )
    _save_session_to_file(summary.model_dump())
    return summary


@app.get("/sessions")
def list_sessions():
    """Return all previously saved session summaries."""
    return _load_sessions_file()


@app.post("/reload")
def reload_artefacts():
    """Hot-reload model artefacts without restarting the server."""
    global artefacts_loaded
    artefacts_loaded = _load_artefacts()
    if artefacts_loaded:
        return {"status": "ok", "message": "Artefacts reloaded successfully."}
    else:
        raise HTTPException(status_code=503, detail="Failed to load artefacts. Check that model files exist.")
