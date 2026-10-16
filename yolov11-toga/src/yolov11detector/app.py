"""
📟 Makers - YOLOv11 TFLite Detector
YOLOv11 Detector - Toga (BeeWare)

Single-file app: detector + helpers + Toga UI.
Modes : Image | Video | Webcam
Targets: Windows / Linux / macOS (live webcam via OpenCV) and
         Android / iOS (snapshot mode via Toga's camera API, see README).
"""
from __future__ import annotations

import asyncio
import mimetypes
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import toga
from toga.platform import current_platform
from toga.style import Pack
from toga.style.pack import BOLD, COLUMN, ROW, CENTER

try:  # desktop
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # Android (Chaquopy) / fallback
    from tflite_runtime.interpreter import Interpreter


# ════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════
APP_TITLE = "YOLOv11 Detector - Toga (BeeWare)"
HEADLINE = "📟 Makers - YOLOv11 TFLite Detector"
MODEL_PATH = Path(__file__).parent / "resources" / "yolo11n_float32.tflite"

PLATFORM = str(current_platform)

IS_ANDROID = PLATFORM == "android"
IS_MOBILE = PLATFORM in ("android", "iOS")
IS_WINDOWS = PLATFORM == "windows"
IS_SNAPSHOT = IS_MOBILE and not IS_ANDROID  # iOS: no frame-stream API in Toga -> photo mode


PLAY_TXT, PAUSE_TXT = ("▶ Play", "⏸ Pause") if IS_MOBILE else ("▶  Play", "⏸  Pause")


def T(desktop: str, mobile: str) -> str:
    """Pick the short label on phones so buttons fit the screen."""
    return mobile if IS_MOBILE else desktop


DISPLAY_MAX_SIDE = 900 if IS_MOBILE else 1280

# Your Flet build needed this on Android (VideoCapture returned swapped channels).
ANDROID_VIDEO_SWAP_RB = True

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mkv", "mov", "webm", "m4v"]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# UI palette
C_BG, C_PANEL, C_CARD, C_VIEW = "#0b1020", "#141b34", "#1b2444", "#05070f"
C_TEXT, C_MUTED = "#e8ecff", "#8b97c4"
C_PINK, C_ORANGE, C_YELLOW = "#ff4d8d", "#ff8a3d", "#f5b800"
C_GREEN, C_CYAN, C_BLUE = "#16a34a", "#0891b2", "#3b82f6"
C_VIOLET, C_RED, C_SLATE = "#8b5cf6", "#ef4444", "#475569"
RAINBOW = [C_PINK, C_ORANGE, C_YELLOW, C_GREEN, C_CYAN, C_BLUE, C_VIOLET]
LEVEL_COLOR = {"info": C_TEXT, "ok": "#4ade80", "warn": "#fbbf24", "error": "#f87171", "busy": "#67e8f9"}


# ════════════════════════════════════════════════════════════════════════════
#  Vision helpers
# ════════════════════════════════════════════════════════════════════════════
def _build_palette(n: int = 80) -> list[tuple[int, int, int]]:
    """One distinct BGR colour per class (golden-ratio hue spacing)."""
    palette = []
    for i in range(n):
        hue = int(((i * 0.61803398875) % 1.0) * 179)
        sat = (255, 200, 230)[i % 3]
        val = (255, 235, 210)[(i // 3) % 3]
        bgr = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


CLASS_COLORS = _build_palette(len(COCO_CLASSES))
FONT = cv2.FONT_HERSHEY_SIMPLEX


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou < iou_thr]
    return keep


class YoloDetector:
    """YOLOv11 TFLite wrapper (vectorised post-processing, class-aware NMS)."""

    def __init__(self, model_path: Path, confidence: float = 0.25, iou: float = 0.55):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.input_w = self.input_h = 320
        self._interp = None
        self._in_idx = self._out_idx = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._interp is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        threads = max(1, (os.cpu_count() or 4) // 2)
        try:
            interp = Interpreter(model_path=str(self.model_path), num_threads=threads)
        except TypeError:
            interp = Interpreter(model_path=str(self.model_path))
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
        self._in_idx, self._out_idx = inp["index"], out["index"]
        if len(inp["shape"]) == 4:
            self.input_h, self.input_w = int(inp["shape"][1]), int(inp["shape"][2])
        self._interp = interp

    def detect(self, frame: np.ndarray):
        """Returns (boxes_xyxy[N,4], scores[N], class_ids[N]) in frame pixels."""
        frame = ensure_bgr(frame)
        oh, ow = frame.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame, (self.input_w, self.input_h)), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0)[None]

        with self._lock:
            self._load()
            self._interp.set_tensor(self._in_idx, tensor)
            self._interp.invoke()
            out = np.squeeze(self._interp.get_tensor(self._out_idx))

        if out.ndim != 2:
            raise ValueError(f"Unexpected YOLO output shape: {out.shape}")
        if out.shape[0] < out.shape[1]:  # (84, N) -> (N, 84)
            out = out.T

        cls_scores = out[:, 4:]
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(out)), class_ids]
        keep = scores >= self.confidence
        if not keep.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)

        raw, scores, class_ids = out[keep, :4].astype(np.float32), scores[keep], class_ids[keep]
        if raw.max() <= 2.0:  # normalised 0..1
            raw *= np.array([ow, oh, ow, oh], np.float32)
        else:  # model-input pixels
            raw *= np.array([ow / self.input_w, oh / self.input_h] * 2, np.float32)
        cx, cy, bw, bh = raw.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh - 1)

        offset = class_ids[:, None].astype(np.float32) * (max(ow, oh) + 1)  # class-aware NMS
        kept = nms(boxes + offset, scores, self.iou)[:300]
        return boxes[kept], scores[kept], class_ids[kept].astype(np.int32)

    def annotate(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        frame = ensure_bgr(frame)
        boxes, scores, class_ids = self.detect(frame)
        out = frame.copy()
        h, w = out.shape[:2]
        thick = max(2, round(max(h, w) / 400))
        scale = max(0.5, max(h, w) / 1300)
        for (x1, y1, x2, y2), score, cid in zip(boxes.astype(int), scores, class_ids):
            color = CLASS_COLORS[cid % len(CLASS_COLORS)]
            name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class {cid}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            text = f"{name} {score:.2f}"
            (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
            top = max(y1 - th - bl - 4, 0)
            cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + bl + 4), color, -1)
            lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            cv2.putText(out, text, (x1 + 3, top + th + 1), FONT, scale,
                        (0, 0, 0) if lum > 140 else (255, 255, 255), 1, cv2.LINE_AA)
        return out, len(boxes)


class FpsMeter:
    def __init__(self, alpha: float = 0.15):
        self.alpha, self.fps, self._last = alpha, 0.0, None

    def reset(self) -> None:
        self._last = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else self.fps + self.alpha * (inst - self.fps)
        self._last = now
        return self.fps


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """FPS badge, top-left. Green >= 15, amber >= 8, red below."""
    h, w = frame.shape[:2]
    scale = max(0.6, max(h, w) / 1100)
    text = f"FPS {fps:5.1f}"
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 2)
    pad = int(8 * scale) + 4
    y2, x2 = min(th + bl + 2 * pad, h), min(tw + 2 * pad, w)
    roi = frame[0:y2, 0:x2]
    roi[:] = (roi * 0.30).astype(np.uint8)
    color = (80, 230, 90) if fps >= 15 else (40, 190, 250) if fps >= 8 else (70, 70, 255)
    cv2.putText(frame, text, (pad, pad + th), FONT, scale, color, 2, cv2.LINE_AA)


def to_jpeg(frame: np.ndarray, max_side: int = DISPLAY_MAX_SIDE) -> bytes:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode frame.")
    return buf.tobytes()


def make_placeholder(title: str, hint: str, accent: tuple[int, int, int]) -> bytes:
    img = np.zeros((540, 960, 3), np.uint8)
    img[:] = (30, 18, 12)
    cv2.rectangle(img, (30, 30), (930, 510), accent, 2, cv2.LINE_AA)
    for text, y, sc, th, col in ((title, 260, 1.6, 3, accent), (hint, 320, 0.8, 1, (190, 175, 160))):
        (tw, _), _ = cv2.getTextSize(text, FONT, sc, th)
        cv2.putText(img, text, ((960 - tw) // 2, y), FONT, sc, col, th, cv2.LINE_AA)
    return to_jpeg(img)


def read_image(path: Path) -> np.ndarray | None:
    """Unicode-path-safe imread."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def write_image(path: Path, frame: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix.lower() or ".jpg", frame)
    if not ok:
        raise ValueError(f"Cannot encode as {path.suffix}")
    buf.tofile(str(path))


def _backends() -> list[int]:
    """Windows: try DirectShow, then Media Foundation, then OpenCV's default."""
    return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if IS_WINDOWS else [cv2.CAP_ANY]


def scan_cameras(max_index: int = 6) -> list[tuple[int, int]]:
    """Returns [(camera_index, working_backend), ...]."""
    found: list[tuple[int, int]] = []
    misses = 0
    for i in range(max_index):
        hit = None
        for backend in _backends():
            cap = cv2.VideoCapture(i, backend)
            try:
                ok = cap.isOpened() and (IS_WINDOWS or cap.read()[0])
            finally:
                cap.release()
            if ok:
                hit = backend
                break
        if hit is None:
            misses += 1
            if found and misses >= 2:  # stop after a gap once something was found
                break
        else:
            found.append((i, hit))
            misses = 0
    return found


def open_camera(key):
    index, backend = key if isinstance(key, tuple) else (key, cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ════════════════════════════════════════════════════════════════════════════
#  Android helpers (Chaquopy `java` API) - only called when running on Android
# ════════════════════════════════════════════════════════════════════════════
def android_list_cameras() -> list[tuple[str, int]]:
    from java import jclass

    Camera = jclass("android.hardware.Camera")
    CameraInfo = jclass("android.hardware.Camera$CameraInfo")
    cams = []
    for i in range(Camera.getNumberOfCameras()):
        info = CameraInfo()
        Camera.getCameraInfo(i, info)
        side = "Front" if info.facing == CameraInfo.CAMERA_FACING_FRONT else "Back"
        cams.append((f"{side} camera ({i})", i))
    return cams


def _make_preview_callback(handler):
    from java import dynamic_proxy, jclass

    PreviewCallback = jclass("android.hardware.Camera$PreviewCallback")

    class _Callback(dynamic_proxy(PreviewCallback)):
        def onPreviewFrame(self, data, camera):
            try:
                handler(data)
            except Exception as e:  # never let an exception escape into Java
                print(f"Preview callback error: {e}")

    return _Callback()


def java_list(jlist) -> list:
    """java.util.List -> Python list (Chaquopy's List proxies aren't always iterable)."""
    if jlist is None:
        return []
    return [jlist.get(i) for i in range(jlist.size())]


def java_bytes_to_np(data) -> np.ndarray:
    try:
        return np.frombuffer(data, dtype=np.uint8).copy()
    except Exception:  # slow but always works
        return np.array(data, dtype=np.int8).view(np.uint8)


_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


# ════════════════════════════════════════════════════════════════════════════
#  Toga style helpers
# ════════════════════════════════════════════════════════════════════════════
def pack(**kw) -> Pack:
    """Pack() that also works on Toga builds that still call 'margin' -> 'padding'."""
    try:
        return Pack(**kw)
    except Exception:
        if "margin" in kw:
            kw["padding"] = kw.pop("margin")
        return Pack(**kw)


def make_button(text: str, handler, color: str) -> toga.Button:
    return toga.Button(
        text, on_press=handler,
        style=pack(flex=1, margin=2 if IS_MOBILE else 3, background_color=color, color="#ffffff",
                   font_weight=BOLD, font_size=9 if IS_MOBILE else 11),
    )


def row(*children) -> toga.Box:
    return toga.Box(children=list(children), style=pack(direction=ROW, margin=(0, 6, 0, 6)))


# ════════════════════════════════════════════════════════════════════════════
#  The app
# ════════════════════════════════════════════════════════════════════════════
class YoloApp(toga.App):
    # ───────────────────────── startup / UI ─────────────────────────
    def startup(self):
        self.detector = YoloDetector(MODEL_PATH, confidence=0.25)

        # state
        self._run_id = 0
        self._task: asyncio.Task | None = None
        self.img_orig: np.ndarray | None = None
        self.img_result: np.ndarray | None = None
        self.img_name = "image"
        self._img_busy = self._img_dirty = False
        self.video_path: Path | None = None
        self.video_state = "idle"  # idle | playing | paused
        self.cam_state = "idle"    # idle | live
        self.cameras: list[tuple[str, object]] = []
        self.cam_index = 0
        self.cam_last: np.ndarray | None = None
        self._cams_scanned = False
        self._frame_evt = asyncio.Event()      # Android live camera
        self._loop = None
        self._latest = None
        self._cam_busy = False
        self._acam: dict | None = None

        self.ph_image = make_placeholder("IMAGE MODE", "Select an image to run detection", (96, 77, 255))
        self.ph_video = make_placeholder("VIDEO MODE", "Select a video to start", (61, 138, 255))
        self.ph_cam = make_placeholder("WEBCAM MODE", "Start the webcam or take a photo", (250, 165, 59))

        self.tabs = toga.OptionContainer(
            content=[
                toga.OptionItem("🖼 Image", self._build_image_tab()),
                toga.OptionItem("🎞 Video", self._build_video_tab()),
                toga.OptionItem("📷 Webcam", self._build_webcam_tab()),
            ],
            on_select=self.on_tab_change,
            style=pack(flex=1),
        )

        rainbow = toga.Box(
            children=[toga.Box(style=pack(flex=1, height=5, background_color=c)) for c in RAINBOW],
            style=pack(direction=ROW),
        )
        header = toga.Box(
            children=[
                toga.Label(HEADLINE, style=pack(font_size=T(18, 12), font_weight=BOLD, color=C_TEXT,
                                                margin=T((10, 12, 0, 12), (8, 8, 0, 8)))),
                toga.Label(APP_TITLE, style=pack(font_size=T(10, 8), color=C_PINK,
                                                 margin=T((0, 12, 8, 12), (0, 8, 6, 8)))),
            ],
            style=pack(direction=COLUMN, background_color=C_PANEL),
        )

        self.conf_lbl = toga.Label(
            T("🎚 Confidence 0.25", "🎚 Conf 0.25"),
            style=pack(color=C_TEXT, font_weight=BOLD, font_size=T(11, 9),
                       margin=T((8, 8, 8, 12), (8, 4, 8, 8)), width=T(190, 105)),
        )
        self.conf_slider = toga.Slider(
            min=0.05, max=0.95, value=0.25, on_change=self.on_conf_change,
            style=pack(flex=1, margin=(8, 12, 8, 0)),
        )
        conf_row = toga.Box(children=[self.conf_lbl, self.conf_slider],
                            style=pack(direction=ROW, background_color=C_CARD))

        self.status_lbl = toga.Label("Ready.", style=pack(flex=1, color=C_TEXT, font_size=T(10, 9),
                                                          margin=T((6, 6, 6, 12), (6, 4, 6, 8))))
        self.info_lbl = toga.Label(
            T(f"🧠 YOLO11n · ⚙ {PLATFORM}", "🧠 YOLO11n"),
            style=pack(color=C_MUTED, font_size=T(10, 8), margin=T((6, 12, 6, 6), (6, 8, 6, 2)))
        )
        status_bar = toga.Box(children=[self.status_lbl, self.info_lbl],
                              style=pack(direction=ROW, background_color=C_PANEL))

        root = toga.Box(
            children=[rainbow, header, self.tabs, conf_row, status_bar],
            style=pack(direction=COLUMN, background_color=C_BG),
        )

        self.main_window = toga.MainWindow(title=APP_TITLE, size=(1280, 820))
        self.main_window.content = root
        self.on_exit = self._on_exit
        self.main_window.show()
        if not IS_MOBILE:
            # Maximize after the window is really on screen (setting it inside startup() is ignored)
            self.add_background_task(self._maximize)
        self._sync()
        self._set_status("Ready. Pick a mode and a source.", "ok")

    async def _maximize(self, app=None, **kwargs):
        await asyncio.sleep(0.3)
        try:
            self.main_window.state = toga.WindowState.MAXIMIZED
        except Exception:
            pass
        try:  # native fallbacks if Toga's own state switch had no effect
            native = self.main_window._impl.native
            if PLATFORM == "windows":
                from System.Windows.Forms import FormWindowState
                if native.WindowState != FormWindowState.Maximized:
                    native.WindowState = FormWindowState.Maximized
            elif PLATFORM == "linux":
                native.maximize()
        except Exception:
            pass

    def _on_exit(self, app, **kwargs):
        self._run_id += 1  # tell loops to stop and release devices
        return True

    def _view(self, placeholder: bytes) -> toga.ImageView:
        return toga.ImageView(image=toga.Image(placeholder),
                              style=pack(flex=1, margin=6, background_color=C_VIEW))

    def _tab_box(self, view, *rows) -> toga.Box:
        return toga.Box(children=[view, *rows],
                        style=pack(direction=COLUMN, flex=1, background_color=C_BG))

    def _build_image_tab(self):
        self.img_view = self._view(self.ph_image)
        self.btn_img_select = make_button(T("🖼  Select Image", "🖼 Select"), self.on_select_image, C_PINK)
        self.btn_img_save = make_button(T("💾  Save Result", "💾 Save"), self.on_save_image, C_GREEN)
        self.btn_img_clear = make_button(T("🧹  Clear", "🧹 Clear"), self.on_clear_image, C_SLATE)
        return self._tab_box(self.img_view, row(self.btn_img_select, self.btn_img_save, self.btn_img_clear))

    def _build_video_tab(self):
        self.vid_view = self._view(self.ph_video)
        self.btn_vid_select = make_button(T("🎞  Select Video", "🎞 Select"), self.on_select_video, C_BLUE)
        self.btn_vid_play = make_button(PLAY_TXT, self.on_play_pause, C_GREEN)
        self.btn_vid_stop = make_button(T("⏹  Stop", "⏹ Stop"), self.on_stop_video, C_RED)
        self.btn_vid_clear = make_button(T("🧹  Clear", "🧹 Clear"), self.on_clear_video, C_SLATE)
        if IS_MOBILE:  # two rows of two so nothing overflows
            return self._tab_box(self.vid_view, row(self.btn_vid_select, self.btn_vid_play),
                                 row(self.btn_vid_stop, self.btn_vid_clear))
        return self._tab_box(self.vid_view, row(self.btn_vid_select, self.btn_vid_play,
                                                self.btn_vid_stop, self.btn_vid_clear))

    def _build_webcam_tab(self):
        self.cam_view = self._view(self.ph_cam)
        self.cam_select = toga.Selection(
            items=["Scanning…"], on_change=self.on_cam_select, style=pack(flex=1, margin=(4, 6, 4, 6))
        )
        self.btn_cam_start = make_button(
            "📷 Photo" if IS_SNAPSHOT else T("▶  Start Webcam", "▶ Start"), self.on_cam_start, C_ORANGE)
        self.btn_cam_swap = make_button(T("🔄  Swap Cam", "🔄 Swap"), self.on_cam_swap, C_VIOLET)
        self.btn_cam_stop = make_button(T("⏹  Stop", "⏹ Stop"), self.on_cam_stop, C_RED)
        self.btn_cam_capture = make_button(T("📸  Capture", "📸 Capture"), self.on_cam_capture, C_GREEN)
        self.btn_cam_rescan = make_button(T("🔍  Rescan", "🔍 Rescan"), self.on_cam_rescan, C_CYAN)
        self.btn_cam_clear = make_button(T("🧹  Clear", "🧹 Clear"), self.on_cam_clear, C_SLATE)
        return self._tab_box(
            self.cam_view,
            row(toga.Label("Camera:", style=pack(color=C_MUTED, margin=(8, 4, 0, 6))), self.cam_select),
            row(self.btn_cam_start, self.btn_cam_swap, self.btn_cam_stop),
            row(self.btn_cam_capture, self.btn_cam_rescan, self.btn_cam_clear),
        )

    # ───────────────────────── small utilities ─────────────────────────
    def _set_status(self, msg: str, level: str = "info") -> None:
        self.status_lbl.text = msg
        self.status_lbl.style.color = LEVEL_COLOR.get(level, C_TEXT)

    def _set_info(self, ms: float) -> None:
        self.info_lbl.text = T(f"⚡ {ms:.0f} ms · {PLATFORM}", f"⚡ {ms:.0f} ms")

    def _tab_index(self) -> int:
        try:
            return self.tabs.current_tab.index
        except Exception:
            return 0

    def _sync(self) -> None:
        """Enable/disable buttons from state."""
        self.btn_img_save.enabled = self.img_result is not None
        self.btn_img_clear.enabled = self.img_orig is not None

        has_vid = self.video_path is not None
        self.btn_vid_play.text = PAUSE_TXT if self.video_state == "playing" else PLAY_TXT
        self.btn_vid_play.enabled = has_vid
        self.btn_vid_stop.enabled = self.video_state != "idle"
        self.btn_vid_clear.enabled = has_vid

        live = self.cam_state == "live"
        self.btn_cam_start.enabled = IS_SNAPSHOT or (bool(self.cameras) and not live)
        self.btn_cam_stop.enabled = live
        self.btn_cam_swap.enabled = len(self.cameras) > 1
        self.btn_cam_capture.enabled = self.cam_last is not None
        self.btn_cam_clear.enabled = live or self.cam_last is not None

    async def _halt(self) -> None:
        """Stop whatever video/webcam loop is running and wait for it to release its device."""
        self._run_id += 1
        self._frame_evt.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                pass
        self.video_state = "idle"
        self.cam_state = "idle"

    def _process(self, frame: np.ndarray, meter: FpsMeter | None = None):
        """Thread-safe: detect + (optional FPS overlay) + JPEG. Returns clean frame too."""
        t0 = time.perf_counter()
        annotated, count = self.detector.annotate(frame)
        ms = (time.perf_counter() - t0) * 1000
        clean, fps = annotated, None
        if meter is not None:
            fps = meter.tick()
            clean = annotated.copy()
            draw_fps(annotated, fps)
        return clean, to_jpeg(annotated), count, ms, fps

    def _captures_dir(self) -> Path:
        if IS_MOBILE:
            base = Path(self.paths.data) / "captures"
        else:
            pics = Path.home() / "Pictures"
            base = (pics if pics.exists() else Path(self.paths.data)) / "YOLOv11 Detector"
        base.mkdir(parents=True, exist_ok=True)
        return base

    async def _save_auto(self, frame: np.ndarray, prefix: str):
        if IS_ANDROID:  # try the public gallery first (Pictures/YOLOv11 Detector)
            try:
                return await asyncio.to_thread(self._android_save_gallery, frame, prefix)
            except Exception as e:
                print(f"Gallery save failed, using app folder: {e}")
        dest = self._captures_dir() / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        await asyncio.to_thread(write_image, dest, frame)
        return dest

    # ───────────────────────── file picking / saving ─────────────────────────
    async def _pick_file(self, title: str, extensions: list[str], mime: str, default_ext: str):
        if IS_ANDROID:
            return await self._android_pick(mime, default_ext)
        return await self.main_window.dialog(toga.OpenFileDialog(title, file_types=extensions))

    async def _android_pick(self, mime: str, default_ext: str) -> Path | None:
        """System file chooser via an Intent; the chosen content:// URI is copied into the cache."""
        from java import jarray, jbyte, jclass

        Intent = jclass("android.content.Intent")
        intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType(mime)
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        result = await self._impl.intent_result(Intent.createChooser(intent, "Select file"))
        if not result or result["resultCode"] != -1:  # RESULT_OK
            return None
        data = result["resultData"]
        uri = data.getData() if data is not None else None
        if uri is None:
            return None

        resolver = self._impl.native.getContentResolver()
        ext = mimetypes.guess_extension(str(resolver.getType(uri) or "")) or default_ext
        cache = Path(self.paths.cache)
        cache.mkdir(parents=True, exist_ok=True)
        for old in cache.glob("picked_*"):
            try:
                old.unlink()
            except OSError:
                pass
        dest = cache / f"picked_{int(time.time())}{ext}"

        def copy() -> None:
            stream = resolver.openInputStream(uri)
            out = jclass("java.io.FileOutputStream")(str(dest))
            buf = jarray(jbyte)(65536)
            try:
                while True:
                    n = stream.read(buf)
                    if n < 0:
                        break
                    out.write(buf, 0, n)
            finally:
                out.close()
                stream.close()

        await asyncio.to_thread(copy)
        return dest

    def _android_save_gallery(self, frame: np.ndarray, prefix: str) -> str:
        from java import jarray, jbyte, jclass

        name = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("Could not encode image.")
        raw = buf.tobytes()
        values = jclass("android.content.ContentValues")()
        values.put("_display_name", name)
        values.put("mime_type", "image/jpeg")
        values.put("relative_path", "Pictures/YOLOv11 Detector")
        resolver = self._impl.native.getContentResolver()
        media = jclass("android.provider.MediaStore$Images$Media")
        uri = resolver.insert(media.EXTERNAL_CONTENT_URI, values)
        if uri is None:
            raise RuntimeError("MediaStore insert failed")
        out = resolver.openOutputStream(uri)
        try:
            try:
                arr = jarray(jbyte)(raw)
            except Exception:
                arr = jarray(jbyte)([b - 256 if b > 127 else b for b in raw])
            out.write(arr)
        finally:
            out.close()
        return f"Pictures/YOLOv11 Detector/{name}"

    # ───────────────────────── Android live camera (Camera API + preview callback) ─────────────────────────
    def _display_rotation(self) -> int:
        try:
            return int(self._impl.native.getWindowManager().getDefaultDisplay().getRotation()) * 90
        except Exception:
            return 0

    def _on_android_preview(self, data) -> None:
        """Called by Android for every preview frame. Keep it tiny; drop frames while busy."""
        if self._cam_busy or self._latest is not None or self._acam is None or self._loop is None:
            return
        self._latest = (java_bytes_to_np(data), self._acam)
        self._loop.call_soon_threadsafe(self._frame_evt.set)

    def _android_step(self, item, meter: FpsMeter):
        arr, meta = item
        w, h = meta["w"], meta["h"]
        yuv = arr[: w * h * 3 // 2].reshape(h * 3 // 2, w)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        if meta["rot"] in _ROTATIONS:
            frame = cv2.rotate(frame, _ROTATIONS[meta["rot"]])
        if meta["mirror"]:
            frame = cv2.flip(frame, 1)
        clean, jpeg, count, ms, fps = self._process(frame, meter)
        self.cam_last = clean
        return jpeg, count, ms, fps

    async def _android_cam_loop(self, rid: int, label: str, cam_id: int):
        from java import jclass

        cam = None
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()
            if rid != self._run_id:
                return
            Camera = jclass("android.hardware.Camera")
            CameraInfo = jclass("android.hardware.Camera$CameraInfo")
            info = CameraInfo()
            Camera.getCameraInfo(cam_id, info)
            cam = Camera.open(cam_id)

            params = cam.getParameters()
            sizes = [(sz.width, sz.height) for sz in java_list(params.getSupportedPreviewSizes())]
            w, h = min(sizes, key=lambda wh: abs(wh[0] * wh[1] - 640 * 480))
            params.setPreviewSize(w, h)
            try:
                if "continuous-video" in [str(m) for m in java_list(params.getSupportedFocusModes())]:
                    params.setFocusMode("continuous-video")
            except Exception:
                pass
            cam.setParameters(params)

            degrees = self._display_rotation()
            if info.facing == CameraInfo.CAMERA_FACING_FRONT:
                rot, mirror = (info.orientation + degrees) % 360, True
            else:
                rot, mirror = (info.orientation - degrees + 360) % 360, False

            texture = jclass("android.graphics.SurfaceTexture")(0)  # dummy surface, we draw ourselves
            cam.setPreviewTexture(texture)
            self._latest, self._cam_busy = None, False
            self._acam = {"w": int(w), "h": int(h), "rot": rot, "mirror": mirror}
            cam.setPreviewCallback(_make_preview_callback(self._on_android_preview))
            cam.startPreview()
            self._frame_evt.clear()
            meter = FpsMeter()

            while rid == self._run_id:
                await self._frame_evt.wait()  # set by the preview callback, or by _halt()
                self._frame_evt.clear()
                if rid != self._run_id:
                    break
                item, self._latest = self._latest, None
                if item is None:
                    continue
                self._cam_busy = True
                try:
                    jpeg, count, ms, fps = await asyncio.to_thread(self._android_step, item, meter)
                finally:
                    self._cam_busy = False
                if rid != self._run_id:
                    break
                self.cam_view.image = toga.Image(jpeg)
                self._set_status(f"📷 {label} · {count} object(s) · {fps:.1f} FPS", "ok")
                self._set_info(ms)
                if self.btn_cam_capture.enabled is False:
                    self._sync()
        except Exception as e:
            if rid == self._run_id:
                self.cam_state = "idle"
                self._set_status(f"❌ Camera failed: {e}", "error")
                self._sync()
        finally:
            self._acam, self._latest = None, None
            if cam is not None:
                try:
                    cam.setPreviewCallback(None)
                    cam.stopPreview()
                except Exception:
                    pass
                try:
                    cam.release()
                except Exception:
                    pass

    # ───────────────────────── tabs / confidence ─────────────────────────
    async def on_tab_change(self, widget, **kwargs):
        await self._halt()
        idx = self._tab_index()
        if idx == 2 and not self._cams_scanned:
            await self._rescan()
        self._sync()
        self._set_status(["🖼 Image mode", "🎞 Video mode", "📷 Webcam mode"][idx], "info")

    async def on_conf_change(self, widget, **kwargs):
        value = round(float(widget.value), 2)
        self.detector.confidence = value
        self.conf_lbl.text = T(f"🎚 Confidence {value:.2f}", f"🎚 Conf {value:.2f}")
        if self._tab_index() == 0 and self.img_orig is not None:
            await self._refresh_image()

    # ───────────────────────── IMAGE ─────────────────────────
    async def on_select_image(self, widget, **kwargs):
        try:
            path = await self._pick_file("Select an image", IMAGE_TYPES, "image/*", ".jpg")
        except Exception as e:
            self._set_status(f"File dialog unavailable on {PLATFORM}: {e}", "error")
            return
        if not path:
            return
        path = Path(path)
        frame = await asyncio.to_thread(read_image, path)
        if frame is None:
            self._set_status(f"❌ Could not decode {path.name}.", "error")
            return
        self.img_orig, self.img_name = frame, path.name
        await self._refresh_image()

    async def _refresh_image(self):
        if self._img_busy:  # slider spam: just mark dirty, running loop re-runs
            self._img_dirty = True
            return
        self._img_busy = True
        try:
            while True:
                self._img_dirty = False
                self._set_status(f"⏳ Running inference on {self.img_name}…", "busy")
                clean, jpeg, count, ms, _ = await asyncio.to_thread(self._process, self.img_orig)
                self.img_result = clean
                self.img_view.image = toga.Image(jpeg)
                self._set_status(f"✅ {count} object(s) in {self.img_name} · {ms:.0f} ms", "ok")
                self._set_info(ms)
                if not self._img_dirty:
                    break
        except Exception as e:
            self._set_status(f"❌ Inference failed: {e}", "error")
        finally:
            self._img_busy = False
            self._sync()

    async def on_save_image(self, widget, **kwargs):
        if self.img_result is None:
            return
        try:
            dest = await self.main_window.dialog(
                toga.SaveFileDialog("Save result", suggested_filename=f"{Path(self.img_name).stem}_yolo.jpg",
                                    file_types=["jpg", "png"])
            )
        except Exception:  # backend has no save dialog (e.g. mobile) -> auto-save
            dest = await self._save_auto(self.img_result, "result")
            self._set_status(f"💾 Saved to {dest}", "ok")
            return
        if not dest:
            return
        try:
            await asyncio.to_thread(write_image, Path(dest), self.img_result)
            self._set_status(f"💾 Saved to {dest}", "ok")
        except Exception as e:
            self._set_status(f"❌ Save failed: {e}", "error")

    async def on_clear_image(self, widget, **kwargs):
        self.img_orig = self.img_result = None
        self.img_view.image = toga.Image(self.ph_image)
        self._set_status("🧹 Image cleared.")
        self._sync()

    # ───────────────────────── VIDEO ─────────────────────────
    async def on_select_video(self, widget, **kwargs):
        await self._halt()
        try:
            path = await self._pick_file("Select a video", VIDEO_TYPES, "video/*", ".mp4")
        except Exception as e:
            self._set_status(f"File dialog unavailable on {PLATFORM}: {e}", "error")
            self._sync()
            return
        if not path:
            self._sync()
            return
        self.video_path = Path(path)
        await self._play_video()

    async def _play_video(self):
        await self._halt()
        self.video_state = "playing"
        rid = self._run_id
        self._task = asyncio.create_task(self._video_loop(rid, self.video_path))
        self._sync()

    async def on_play_pause(self, widget, **kwargs):
        if self.video_state == "playing":
            self.video_state = "paused"
            self._set_status("⏸ Paused.", "warn")
        elif self.video_state == "paused":
            self.video_state = "playing"
        elif self.video_path is not None:
            await self._play_video()
            return
        self._sync()

    async def on_stop_video(self, widget, **kwargs):
        await self._halt()
        self._set_status("⏹ Video stopped.", "warn")
        self._sync()

    async def on_clear_video(self, widget, **kwargs):
        await self._halt()
        self.video_path = None
        self.vid_view.image = toga.Image(self.ph_video)
        self._set_status("🧹 Video cleared.")
        self._sync()

    def _video_step(self, cap, meter: FpsMeter):
        ok, frame = cap.read()
        if not ok:
            return None
        if IS_ANDROID and ANDROID_VIDEO_SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, jpeg, count, ms, fps = self._process(frame, meter)
        return jpeg, count, ms, fps

    async def _video_loop(self, rid: int, path: Path):
        cap = None
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, str(path))
            if not cap.isOpened():
                if rid == self._run_id:
                    self.video_state = "idle"
                    self._set_status(f"❌ Could not open {path.name} (check codec).", "error")
                    self._sync()
                return
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_time = 1.0 / src_fps if 1 <= src_fps <= 120 else 1 / 30
            meter = FpsMeter()
            while rid == self._run_id:
                if self.video_state == "paused":
                    meter.reset()
                    await asyncio.sleep(0.05)
                    continue
                t0 = time.perf_counter()
                res = await asyncio.to_thread(self._video_step, cap, meter)
                if rid != self._run_id:
                    break
                if res is None:
                    self.video_state = "idle"
                    self._set_status(f"✅ Finished {path.name}.", "ok")
                    self._sync()
                    break
                jpeg, count, ms, fps = res
                self.vid_view.image = toga.Image(jpeg)
                self._set_status(f"▶ {path.name} · {count} object(s) · {fps:.1f} FPS", "ok")
                self._set_info(ms)
                await asyncio.sleep(max(frame_time - (time.perf_counter() - t0), 0.001))
        except Exception as e:
            if rid == self._run_id:
                self.video_state = "idle"
                self._set_status(f"❌ Video inference failed: {e}", "error")
                self._sync()
        finally:
            if cap is not None:
                await asyncio.to_thread(cap.release)

    # ───────────────────────── WEBCAM ─────────────────────────
    def _fill_cam_select(self) -> None:
        labels = [c[0] for c in self.cameras] or ["No camera found"]
        self.cam_select.items = labels
        self.cam_select.enabled = bool(self.cameras)
        self._select_cam_ui()

    def _select_cam_ui(self) -> None:
        if self.cameras:
            self.cam_select.value = self.cameras[self.cam_index][0]

    async def _rescan(self):
        self._set_status("🔍 Scanning for cameras…", "busy")
        if IS_ANDROID:
            self.cameras = android_list_cameras()
        elif IS_MOBILE:
            devices = []
            try:
                devices = list(self.camera.devices)
            except Exception:
                pass
            self.cameras = [(getattr(d, "name", None) or f"Camera {i}", d) for i, d in enumerate(devices)]
            if not self.cameras:
                self.cameras = [("Default camera", None)]
        else:
            found = await asyncio.to_thread(scan_cameras)
            self.cameras = [(f"Camera {i}", (i, backend)) for i, backend in found]
        self._cams_scanned = True
        self.cam_index = 0
        self._fill_cam_select()
        n = len(self.cameras)
        self._set_status(
            f"✅ Found {n} camera(s)." if n else
            f"⚠️ No camera found (OpenCV {cv2.__version__}, {PLATFORM}). Check privacy settings / other apps, then Rescan.",
            "ok" if n else "warn")
        self._sync()

    async def on_cam_rescan(self, widget, **kwargs):
        await self._halt()
        await self._rescan()

    async def on_cam_select(self, widget, **kwargs):
        labels = [c[0] for c in self.cameras]
        try:
            idx = labels.index(widget.value)
        except ValueError:
            return
        if idx == self.cam_index:
            return
        self.cam_index = idx
        if self.cam_state == "live":
            await self._start_live()

    async def on_cam_start(self, widget, **kwargs):
        if not self._cams_scanned:
            await self._rescan()
        if IS_SNAPSHOT:
            await self._mobile_snap()
        elif self.cameras:
            await self._start_live()
        else:
            self._set_status("⚠️ No camera found.", "warn")

    async def _start_live(self):
        await self._halt()
        self.cam_state = "live"
        rid = self._run_id
        label, key = self.cameras[self.cam_index]
        self._set_status(f"⏳ Opening {label}…", "busy")
        self._loop = asyncio.get_running_loop()
        if IS_ANDROID:
            self._task = asyncio.create_task(self._android_cam_loop(rid, label, key))
        else:
            self._task = asyncio.create_task(self._webcam_loop(rid, label, key))
        self._sync()

    async def on_cam_swap(self, widget, **kwargs):
        if len(self.cameras) < 2:
            self._set_status("⚠️ Only one camera available.", "warn")
            return
        self.cam_index = (self.cam_index + 1) % len(self.cameras)
        self._select_cam_ui()
        if self.cam_state == "live":
            await self._start_live()
        else:
            self._set_status(f"🔄 Selected {self.cameras[self.cam_index][0]}.", "info")

    async def on_cam_stop(self, widget, **kwargs):
        await self._halt()
        self._set_status("⏹ Webcam stopped.", "warn")
        self._sync()

    async def on_cam_clear(self, widget, **kwargs):
        await self._halt()
        self.cam_last = None
        self.cam_view.image = toga.Image(self.ph_cam)
        self._set_status("🧹 Webcam view cleared.")
        self._sync()

    async def on_cam_capture(self, widget, **kwargs):
        if self.cam_last is None:
            return
        try:
            dest = await self._save_auto(self.cam_last, "capture")
            self._set_status(f"📸 Captured → {dest}", "ok")
        except Exception as e:
            self._set_status(f"❌ Capture failed: {e}", "error")

    def _webcam_step(self, cap, meter: FpsMeter):
        ok, frame = cap.read()
        if not ok:
            return None
        clean, jpeg, count, ms, fps = self._process(frame, meter)
        self.cam_last = clean
        return jpeg, count, ms, fps

    async def _webcam_loop(self, rid: int, label: str, key):
        cap = None
        try:
            cap = await asyncio.to_thread(open_camera, key)
            if rid != self._run_id:
                return
            if cap is None:
                self.cam_state = "idle"
                self._set_status(f"❌ Could not open {label}.", "error")
                self._sync()
                return
            meter = FpsMeter()
            while rid == self._run_id:
                res = await asyncio.to_thread(self._webcam_step, cap, meter)
                if rid != self._run_id:
                    break
                if res is None:
                    self.cam_state = "idle"
                    self._set_status(f"❌ {label}: frame could not be read.", "error")
                    self._sync()
                    break
                jpeg, count, ms, fps = res
                self.cam_view.image = toga.Image(jpeg)
                self._set_status(f"📷 {label} · {count} object(s) · {fps:.1f} FPS", "ok")
                self._set_info(ms)
                if self.btn_cam_capture.enabled is False:
                    self._sync()
                await asyncio.sleep(0)
        except Exception as e:
            if rid == self._run_id:
                self.cam_state = "idle"
                self._set_status(f"❌ Webcam inference failed: {e}", "error")
                self._sync()
        finally:
            if cap is not None:
                await asyncio.to_thread(cap.release)

    async def _mobile_snap(self):
        """Android/iOS: Toga has no frame-stream API, so detect on a photo per press."""
        try:
            cam = self.camera
            if not cam.has_permission:
                await cam.request_permission()
            device = self.cameras[self.cam_index][1] if self.cameras else None
            self._set_status("📷 Opening camera…", "busy")
            photo = await cam.take_photo(device=device)
            if photo is None:
                self._set_status("Photo cancelled.", "warn")
                return
            frame = await asyncio.to_thread(
                cv2.imdecode, np.frombuffer(photo.data, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError("Could not decode the photo.")
            self._set_status("⏳ Running inference…", "busy")
            clean, jpeg, count, ms, _ = await asyncio.to_thread(self._process, frame)
            self.cam_last = clean
            self.cam_view.image = toga.Image(jpeg)
            self._set_info(ms)
            self._set_status(f"✅ {count} object(s) · {ms:.0f} ms (snapshot mode)", "ok")
        except Exception as e:
            self._set_status(f"❌ Camera failed: {e}", "error")
        self._sync()


def main() -> toga.App:
    return YoloApp()