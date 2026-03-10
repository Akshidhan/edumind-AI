"""
Streamlit front-end for Attention Score Prediction.

Tabs
----
  📹 Live Webcam    – real-time face / body detection → API prediction (session-tracked)
  📋 Manual Input   – enter features by hand (quick presets included)
  📦 Batch          – paste a JSON array for batch scoring
  📊 Session History – view past session averages

Run:
    streamlit run streamlit_app.py
"""

import json
import math
import os
import time
import urllib.request
from pathlib import Path

# Suppress noisy TensorFlow / MediaPipe warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # hide TF info & warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # silence oneDNN messages
os.environ["GLOG_minloglevel"] = "3"                 # silence MediaPipe C++ logs

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ────────────────────────────── Config ──────────────────────────────────
API_URL = "http://127.0.0.1:8000"
MODELS_DIR = Path(__file__).resolve().parent / "models"
FACE_MODEL = MODELS_DIR / "face_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"

# Webcam processing settings (hybrid optimisation)
WEBCAM_WIDTH  = 640          # resize frames before MediaPipe
WEBCAM_HEIGHT = 480
API_INTERVAL  = 1.5          # seconds between API calls (reduces server load)
PROCESS_EVERY = 2            # run MediaPipe on every Nth frame (skip others)
FPS_CAP_SLEEP = 0.033        # ~30 fps cap

st.set_page_config(page_title="Attention Predictor", page_icon="🎯", layout="wide")

# ────────────────── Download model bundles if missing ───────────────────
_MODEL_URLS = {
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _ensure_models():
    MODELS_DIR.mkdir(exist_ok=True)
    for path, url in _MODEL_URLS.items():
        if not path.exists():
            with st.spinner(f"Downloading {path.name} …"):
                urllib.request.urlretrieve(url, str(path))


_ensure_models()

# ────────────────────── MediaPipe Tasks aliases ─────────────────────────
BaseOptions       = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarksConns    = mp.tasks.vision.FaceLandmarksConnections

PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarksConns    = mp.tasks.vision.PoseLandmarksConnections

draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec    = mp.tasks.vision.drawing_utils.DrawingSpec

# 3-D reference face points for solvePnP head-pose estimation
_FACE_3D = np.array([
    [ 0.0,    0.0,    0.0  ],   # Nose tip           (landmark 1)
    [ 0.0, -330.0,  -65.0 ],   # Chin                (landmark 152)
    [-225.0, 170.0, -135.0],   # Left eye outer      (landmark 33)
    [ 225.0, 170.0, -135.0],   # Right eye outer     (landmark 263)
    [-150.0,-150.0, -125.0],   # Left mouth corner   (landmark 61)
    [ 150.0,-150.0, -125.0],   # Right mouth corner  (landmark 291)
], dtype=np.float64)
_FACE_IDX = [1, 152, 33, 263, 61, 291]


# ═══════════════════════════ Helpers ════════════════════════════════════

def _head_pose_solvepnp(face_landmarks, w, h):
    """Return (pitch, yaw, roll) in **degrees** via solvePnP, or three Nones."""
    try:
        pts = np.array(
            [[face_landmarks[i].x * w,
              face_landmarks[i].y * h] for i in _FACE_IDX],
            dtype=np.float64,
        )
        f = float(w)
        cam = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D, pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None, None, None
        rmat, _ = cv2.Rodrigues(rvec)
        ang, *_ = cv2.RQDecomp3x3(rmat)
        return float(ang[0]), float(ang[1]), float(ang[2])
    except Exception:
        return None, None, None


def _process_frame(frame, face_lm, pose_lm):
    """
    Run MediaPipe FaceLandmarker + PoseLandmarker on *frame*.

    Returns
    -------
    features  : dict      - the 6 base features (values may be None)
    annotated : ndarray   - BGR frame with overlays
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ann = frame.copy()

    face_area = pitch = yaw = roll = None
    body_area = vis_mean = None

    # ── Face ──────────────────────────────────────────────────
    face_result = face_lm.detect(mp_image)
    if face_result.face_landmarks:
        fl = face_result.face_landmarks[0]

        draw_landmarks(
            ann, fl,
            FaceLandmarksConns.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(color=(0, 200, 0), thickness=1),
        )

        xs = [lm.x * w for lm in fl]
        ys = [lm.y * h for lm in fl]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        face_area = float((x1 - x0) * (y1 - y0))
        cv2.rectangle(ann, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(ann, f"face {face_area:.0f}px",
                    (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        pitch, yaw, roll = _head_pose_solvepnp(fl, w, h)
        if pitch is not None:
            cv2.putText(ann,
                        f"P:{pitch:.1f}  Y:{yaw:.1f}  R:{roll:.1f}",
                        (x0, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    else:
        cv2.putText(ann, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ── Body ──────────────────────────────────────────────────
    pose_result = pose_lm.detect(mp_image)
    if pose_result.pose_landmarks:
        pl = pose_result.pose_landmarks[0]

        draw_landmarks(
            ann, pl,
            PoseLandmarksConns.POSE_LANDMARKS,
            landmark_drawing_spec=DrawingSpec(color=(255, 128, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=DrawingSpec(color=(200, 200, 200), thickness=1),
        )

        bx = [lm.x * w for lm in pl]
        by_ = [lm.y * h for lm in pl]
        bx0, bx1 = int(min(bx)), int(max(bx))
        by0, by1 = int(min(by_)), int(max(by_))
        body_area = float((bx1 - bx0) * (by1 - by0))
        vis_mean = float(np.mean([lm.visibility for lm in pl]))
        cv2.rectangle(ann, (bx0, by0), (bx1, by1), (255, 128, 0), 1)

    return {
        "face_area":     face_area,
        "body_area":     body_area,
        "pitch":         pitch,
        "yaw":           yaw,
        "roll":          roll,
        "pose_vis_mean": round(vis_mean, 4) if vis_mean is not None else None,
    }, ann


# ── API helpers ───────────────────────────────────────────────────────

def _call_api(features):
    """POST one sample to /predict (stateless).  Returns dict or None."""
    try:
        r = requests.post(
            f"{API_URL}/predict", json={"samples": [features]}, timeout=5
        )
        if r.status_code == 200:
            return r.json()["predictions"][0]
    except Exception:
        pass
    return None


def _session_start():
    """Start a tracked session via API.  Returns response dict or None."""
    try:
        r = requests.post(f"{API_URL}/session/start", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _session_sample(session_id, features):
    """Send one sample in a session.  Returns response dict or None."""
    try:
        r = requests.post(
            f"{API_URL}/session/sample",
            json={"session_id": session_id, "features": features},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _session_end(session_id):
    """End session.  Returns summary dict or None."""
    try:
        r = requests.post(
            f"{API_URL}/session/end",
            json={"session_id": session_id},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _get_sessions():
    """Fetch list of past sessions from API."""
    try:
        r = requests.get(f"{API_URL}/sessions", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def _badge(pct):
    """Colour-coded attention level."""
    if pct >= 75:
        st.success(f"✅ High Attention ({pct:.1f} %)")
    elif pct >= 50:
        st.info(f"ℹ️ Moderate Attention ({pct:.1f} %)")
    elif pct >= 25:
        st.warning(f"⚠️ Low Attention ({pct:.1f} %)")
    else:
        st.error(f"🚨 Very Low Attention ({pct:.1f} %)")


def _fmt_duration(sec):
    """Format seconds as MM:SS or HH:MM:SS."""
    sec = int(sec)
    if sec < 3600:
        return f"{sec // 60}:{sec % 60:02d}"
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# ═════════════════════════ Sidebar ══════════════════════════════════════
with st.sidebar:
    st.header("⚙️ API Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            info = resp.json()
            if info.get("model_loaded"):
                st.success("API is **online** ✅")
                st.caption(f"Features: {len(info.get('features') or [])}")
                active = info.get("active_sessions", 0)
                if active:
                    st.info(f"Active sessions: {active}")
            else:
                st.warning("API online – **model not loaded**.")
        else:
            st.error(f"API status {resp.status_code}")
    except requests.ConnectionError:
        st.error("Cannot reach API at `127.0.0.1:8000`.")

    st.markdown("---")
    st.markdown(
        "**Tabs**\n"
        "- 📹 **Live Webcam** – session-tracked real-time prediction\n"
        "- 📋 **Manual Input** – enter features by hand\n"
        "- 📦 **Batch** – JSON array prediction\n"
        "- 📊 **Session History** – past session averages"
    )
    st.markdown("---")
    st.caption(
        "💡 **Hybrid architecture**: MediaPipe (face/pose detection) "
        "runs client-side, model inference + session tracking runs on the API server."
    )


# ═════════════════════════ Main ═════════════════════════════════════════
st.title("🎯 Attention Score Predictor")

tab_cam, tab_manual, tab_batch, tab_history = st.tabs(
    ["📹 Live Webcam", "📋 Manual Input", "📦 Batch Prediction", "📊 Session History"]
)


# ─────────────────────── TAB 1 · Live Webcam ───────────────────────────
with tab_cam:
    st.subheader("Real-time Attention Monitoring")
    st.markdown(
        "Start the webcam to begin a **tracked session**.  "
        "MediaPipe detects your face & body locally, extracts features, and the API "
        "predicts your attention score.  When you stop, you'll see the **session summary** "
        "with the average score."
    )

    # Session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "session_summary" not in st.session_state:
        st.session_state.session_summary = None
    if "stop_session_flag" not in st.session_state:
        st.session_state.stop_session_flag = False

    col_start, col_stop = st.columns([3, 1])
    with col_start:
        run_cam = st.checkbox("🟢 Start Webcam Session", value=False, key="run_cam")
    with col_stop:
        if run_cam and st.button("⏹️ Stop", width="stretch"):
            st.session_state.stop_session_flag = True
            st.rerun()

    # ── Show previous session summary (after stopping) ──────────
    if st.session_state.session_summary and not run_cam:
        s = st.session_state.session_summary
        st.markdown("---")
        st.subheader("📊 Last Session Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", _fmt_duration(s["duration_sec"]))
        c2.metric("Samples", s["total_samples"])
        c3.metric("Avg Attention", f"{s['avg_attention_pct']:.1f} %")
        c4.metric("Final Score", f"{s['final_avg_attention_pct']:.1f} %")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Avg Raw Score", f"{s['avg_raw_score']:.2f} / 5.0")
            st.metric("Looking Down %", f"{s['looking_down_pct']:.1f} %")
        with col_b:
            ld_pen = s.get("looking_down_penalty", 0.0)
            if ld_pen > 0:
                st.warning(
                    f"⚠️ Looking-down penalty: **-{ld_pen:.1f} pp** "
                    f"(looked down {s['looking_down_pct']:.0f}% of session, threshold is 30%)"
                )
            else:
                st.success("✅ No looking-down penalty")
        _badge(s["final_avg_attention_pct"])
        st.markdown("---")

    if run_cam:
        col_vid, col_info = st.columns([3, 2])
        with col_vid:
            ph_frame = st.empty()
        with col_info:
            ph_score = st.empty()
            ph_feats = st.empty()
        ph_session_info = st.empty()
        ph_chart  = st.empty()
        ph_status = st.empty()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error(
                "❌ Cannot open webcam – make sure no other application is using it."
            )
        else:
            # Reduce webcam resolution for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

            # Create MediaPipe landmarkers (IMAGE mode)
            face_lm = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_facial_transformation_matrixes=True,
                )
            )
            pose_lm = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
                    running_mode=VisionRunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )

            # Start session via API
            sess_info = _session_start()
            if sess_info:
                st.session_state.session_id = sess_info["session_id"]
                st.session_state.session_summary = None
            else:
                ph_status.warning("⚠️ Could not start session – predictions will be stateless.")

            history = []
            last_api = 0.0
            frame_count = 0
            cached_feats = {}
            cached_ann = None

            try:
                while True:
                    # Check if stop button was pressed
                    if st.session_state.stop_session_flag:
                        st.session_state.stop_session_flag = False
                        break

                    ret, frame = cap.read()
                    if not ret:
                        ph_status.warning("⚠️ Webcam read failed.")
                        break

                    frame = cv2.flip(frame, 1)
                    frame_count += 1

                    # ── Process only every Nth frame (performance) ────
                    if frame_count % PROCESS_EVERY == 0 or cached_ann is None:
                        feats, ann = _process_frame(frame, face_lm, pose_lm)
                        cached_feats = feats
                        cached_ann = ann
                    else:
                        ann = frame
                        feats = cached_feats

                    ph_frame.image(
                        cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        width="stretch",
                    )

                    # ── Call API at interval ──────────────────────────
                    now = time.time()
                    if now - last_api >= API_INTERVAL:
                        last_api = now

                        sid = st.session_state.session_id
                        if sid:
                            res_full = _session_sample(sid, feats)
                            if res_full:
                                res = res_full["prediction"]
                                pct = res["percentage"]
                                raw = res["raw_score"]
                                penalty = res.get("gaze_penalty", 0.0)
                                looking_down = res.get("looking_down", False)

                                with ph_score.container():
                                    sc1, sc2 = st.columns(2)
                                    sc1.metric("Score", f"{raw:.2f} / 5")
                                    sc2.metric("Attention", f"{pct:.1f} %")
                                    st.progress(pct / 100)
                                    if looking_down:
                                        st.caption("📝 Looking down (writing notes)")
                                    if penalty > 0:
                                        st.caption(f"👁️ Gaze penalty: -{penalty:.1f} pp")
                                    _badge(pct)

                                with ph_session_info.container():
                                    si1, si2, si3 = st.columns(3)
                                    si1.metric("Session Samples", res_full["samples_count"])
                                    si2.metric("Duration", _fmt_duration(res_full["session_duration_sec"]))
                                    ld_pct = res_full["looking_down_pct"]
                                    ld_color = "🟢" if ld_pct <= 30 else "🟡" if ld_pct <= 50 else "🔴"
                                    si3.metric("Looking Down", f"{ld_color} {ld_pct:.0f}%")

                                with ph_feats.container():
                                    st.caption("Extracted features:")
                                    for k, v in feats.items():
                                        txt = f"{v:.2f}" if v is not None else "N/A"
                                        st.text(f"  {k}: {txt}")

                                history.append({
                                    "time": time.strftime("%H:%M:%S"),
                                    "Attention %": pct,
                                })
                                if len(history) > 1:
                                    ph_chart.line_chart(
                                        pd.DataFrame(history).set_index("time"),
                                        width="stretch",
                                    )
                        else:
                            # Fallback: stateless prediction
                            res = _call_api(feats)
                            if res:
                                pct = res["percentage"]
                                raw = res["raw_score"]
                                with ph_score.container():
                                    sc1, sc2 = st.columns(2)
                                    sc1.metric("Score", f"{raw:.2f} / 5")
                                    sc2.metric("Attention", f"{pct:.1f} %")
                                    st.progress(pct / 100)
                                    _badge(pct)

                    time.sleep(FPS_CAP_SLEEP)

            except Exception as e:
                ph_status.error(f"Webcam error: {e}")
            finally:
                face_lm.close()
                pose_lm.close()
                cap.release()

                # End session and store summary
                sid = st.session_state.session_id
                if sid:
                    summary = _session_end(sid)
                    if summary:
                        st.session_state.session_summary = summary
                    st.session_state.session_id = None

                ph_status.info("📷 Webcam stopped – session ended. Uncheck to see summary.")


# ─────────────────────── TAB 2 · Manual Input ──────────────────────────
with tab_manual:
    st.subheader("Manual Feature Input")

    _DEFAULTS = {
        "m_face":  25000.0,
        "m_body":  280000.0,
        "m_pitch": -10.0,
        "m_yaw":   -2.0,
        "m_roll":  0.0,
        "m_vis":   0.85,
    }
    for _k, _v in _DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    st.markdown("**⚡ Quick Presets**  *(angles in degrees, matching training data)*")
    _pcols = st.columns(5)
    _PRESETS = {
        "👀 Attentive": dict(
            m_face=25000.0, m_body=280000.0,
            m_pitch=-10.0, m_yaw=-2.0, m_roll=0.0, m_vis=0.85,
        ),
        "😐 Slightly Off": dict(
            m_face=20000.0, m_body=250000.0,
            m_pitch=-5.0, m_yaw=15.0, m_roll=-3.0, m_vis=0.70,
        ),
        "📝 Writing": dict(
            m_face=22000.0, m_body=260000.0,
            m_pitch=-25.0, m_yaw=-5.0, m_roll=-2.0, m_vis=0.75,
        ),
        "🙄 Looking Away": dict(
            m_face=12000.0, m_body=200000.0,
            m_pitch=5.0, m_yaw=35.0, m_roll=5.0, m_vis=0.50,
        ),
        "🚫 No Face": dict(
            m_face=0.0, m_body=250000.0,
            m_pitch=0.0, m_yaw=0.0, m_roll=0.0, m_vis=0.30,
        ),
    }
    for (lbl, vals), col in zip(_PRESETS.items(), _pcols):
        if col.button(lbl, width="stretch"):
            for k2, v2 in vals.items():
                st.session_state[k2] = v2
            st.rerun()

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        face_area = st.number_input(
            "Face Area (px²)", min_value=0.0, max_value=500000.0,
            step=500.0, key="m_face",
            help="Face bounding-box area in pixels.  0 = no face detected.",
        )
        body_area = st.number_input(
            "Body Area (px²)", min_value=0.0, max_value=500000.0,
            step=500.0, key="m_body",
            help="Body bounding-box area in pixels.",
        )
        pitch = st.number_input(
            "Pitch (°)", min_value=-90.0, max_value=90.0,
            step=1.0, format="%.1f", key="m_pitch",
            help="Head pitch in degrees.  Negative = looking down at desk/notes.",
        )
    with c2:
        yaw = st.number_input(
            "Yaw (°)", min_value=-90.0, max_value=90.0,
            step=1.0, format="%.1f", key="m_yaw",
            help="Head yaw in degrees.  0 = facing camera.",
        )
        roll = st.number_input(
            "Roll (°)", min_value=-90.0, max_value=90.0,
            step=1.0, format="%.1f", key="m_roll",
            help="Head roll (tilt) in degrees.  0 = upright.",
        )
        vis = st.number_input(
            "Pose Visibility Mean", min_value=0.0, max_value=1.0,
            step=0.05, format="%.2f", key="m_vis",
            help="Mean body-keypoint visibility (0-1).",
        )

    if st.button("🔮 Predict Attention Score", type="primary",
                 width="stretch"):
        sample = {
            "face_area":     face_area if face_area > 0 else None,
            "body_area":     body_area if body_area > 0 else None,
            "pitch":         pitch if face_area > 0 else None,
            "yaw":           yaw if face_area > 0 else None,
            "roll":          roll if face_area > 0 else None,
            "pose_vis_mean": vis,
        }
        result = _call_api(sample)
        if result:
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Raw Score", f"{result['raw_score']:.2f} / 5.0")
            rc2.metric("Attention %", f"{result['percentage']:.1f} %")
            penalty = result.get("gaze_penalty", 0.0)
            rc3.metric("Gaze Penalty", f"-{penalty:.1f} pp")
            st.progress(result["percentage"] / 100)
            if result.get("looking_down"):
                st.caption("📝 Student is looking down (writing notes) – no gaze penalty applied.")
            elif penalty > 0:
                st.caption(f"ℹ️ A {penalty:.1f} pp penalty was applied because gaze is away from screen.")
            _badge(result["percentage"])
            with st.expander("🔍 Raw API Response"):
                st.json({"predictions": [result]})
        else:
            st.error("Prediction failed – check API connection.")


# ─────────────────────── TAB 3 · Batch Prediction ──────────────────────
with tab_batch:
    st.subheader("Batch Prediction")
    st.markdown(
        "Paste a JSON array of samples.  Use `null` for missing features.\n\n"
        "```json\n"
        "[\n"
        '  {"face_area":25000,"body_area":280000,"pitch":-10,"yaw":-2,'
        '"roll":0,"pose_vis_mean":0.85},\n'
        '  {"face_area":null,"body_area":250000,"pitch":null,"yaw":null,'
        '"roll":null,"pose_vis_mean":0.3}\n'
        "]\n"
        "```"
    )
    batch_json = st.text_area("JSON array:", height=150)

    if st.button("🚀 Run Batch Prediction"):
        if batch_json.strip():
            try:
                samples = json.loads(batch_json)
                r = requests.post(
                    f"{API_URL}/predict",
                    json={"samples": samples},
                    timeout=30,
                )
                if r.status_code == 200:
                    preds = r.json()["predictions"]
                    st.success(f"{len(preds)} prediction(s) returned")
                    st.dataframe(pd.DataFrame(preds), width="stretch")
                else:
                    st.error(f"API Error {r.status_code}: {r.text}")
            except json.JSONDecodeError:
                st.error("Invalid JSON format.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Paste some JSON first.")


# ─────────────────────── TAB 4 · Session History ───────────────────────
with tab_history:
    st.subheader("📊 Session History")
    st.markdown("All completed webcam sessions and their average attention scores.")

    if st.button("🔄 Refresh"):
        st.rerun()

    sessions = _get_sessions()
    if not sessions:
        st.info("No sessions recorded yet.  Start a webcam session to begin tracking.")
    else:
        # Most recent first
        sessions = list(reversed(sessions))

        # Summary table
        rows = []
        for s in sessions:
            rows.append({
                "Date": s.get("started_at", "")[:19].replace("T", " "),
                "Duration": _fmt_duration(s.get("duration_sec", 0)),
                "Samples": s.get("total_samples", 0),
                "Avg Attention %": round(s.get("avg_attention_pct", 0), 1),
                "Looking Down %": round(s.get("looking_down_pct", 0), 1),
                "LD Penalty": round(s.get("looking_down_penalty", 0), 1),
                "Final Score %": round(s.get("final_avg_attention_pct", 0), 1),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        # Expandable details
        for i, s in enumerate(sessions):
            with st.expander(
                f"Session {i+1}  –  "
                f"{s.get('started_at', '')[:19].replace('T', ' ')}  –  "
                f"Final: {s.get('final_avg_attention_pct', 0):.1f}%"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Raw Score", f"{s.get('avg_raw_score', 0):.2f}")
                c2.metric("Avg Attention", f"{s.get('avg_attention_pct', 0):.1f}%")
                c3.metric("Final (after penalties)", f"{s.get('final_avg_attention_pct', 0):.1f}%")

                ld_pct = s.get("looking_down_pct", 0)
                ld_pen = s.get("looking_down_penalty", 0)
                if ld_pen > 0:
                    st.warning(
                        f"⚠️ Student looked down {ld_pct:.0f}% of the session "
                        f"(threshold 30%) → {ld_pen:.1f} pp penalty deducted."
                    )
                else:
                    st.success(f"✅ Looking down at {ld_pct:.0f}% – within acceptable range.")
                _badge(s.get("final_avg_attention_pct", 0))
