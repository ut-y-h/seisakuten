# live_faceswap_m1_fast.py
# 品質維持のまま高速化：検出は間引き＋LK光学フローでkps追跡、
# 認識(embedding)はターゲット画像で1回だけ、ブレンドは全体αで背景不変。

import time
import math
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
from threading import Thread
from collections import deque

# ========= ユーザー設定 =========
TARGET_IMAGE_PATH = "target.png"
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
DETECTION_SIZE = (640, 640)     # (512,512)でも可（品質変わらず軽くなりやすい）
MORPH_DURATION_SEC = 20.0
MIRROR_VIEW = True
# 検出間引き（Nフレームごとに1回検出）。12〜15あたりから試す
DETECTION_INTERVAL = 12
# 検出をROIに限定するパディング率（前回bboxからの拡張）
ROI_PAD = 0.35
# 光学フロー（LK）の設定
LK_WINSZ = 21
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
# 流れが悪化したら再検出する閾値（有効点率）
TRACK_GOOD_MIN_RATIO = 0.8
# bbox拡張倍率（kpsからbboxを作るときに少し広げる）
BBOX_EXPAND = 1.7
# =================================

def ease_in_out_smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)

def largest_face(faces):
    if not faces:
        return None
    def area(face):
        x1, y1, x2, y2 = face.bbox.astype(float)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)

def expand_bbox(bbox, pad, w, h):
    x1, y1, x2, y2 = bbox.astype(float)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1), (y2 - y1)
    bw2, bh2 = bw * (1.0 + pad), bh * (1.0 + pad)
    nx1, ny1 = int(max(0, cx - bw2 / 2.0)), int(max(0, cy - bh2 / 2.0))
    nx2, ny2 = int(min(w - 1, cx + bw2 / 2.0)), int(min(h - 1, cy + bh2 / 2.0))
    return nx1, ny1, nx2, ny2

def detect_with_roi(app, frame, prev_bbox=None):
    h, w = frame.shape[:2]
    # 直近の位置があればROI検出→失敗時フル画像
    if prev_bbox is not None:
        x1, y1, x2, y2 = expand_bbox(prev_bbox, ROI_PAD, w, h)
        roi = frame[y1:y2, x1:x2]
        faces = app.get(roi, max_num=1)
        if faces:
            f = largest_face(faces)
            # オフセットを戻す
            f.bbox = f.bbox.astype(np.float32)
            f.bbox[0::2] += x1  # x1, x2
            f.bbox[1::2] += y1  # y1, y2
            f.kps = f.kps.astype(np.float32)
            f.kps[:, 0] += x1
            f.kps[:, 1] += y1
            return f
    faces = app.get(frame, max_num=1)
    return largest_face(faces) if faces else None

def kps_to_bbox(kps, w, h, expand=1.0):
    x_min = float(np.min(kps[:, 0]))
    y_min = float(np.min(kps[:, 1]))
    x_max = float(np.max(kps[:, 0]))
    y_max = float(np.max(kps[:, 1]))
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    bw, bh = (x_max - x_min), (y_max - y_min)
    bw *= expand
    bh *= expand
    x1 = int(max(0, cx - bw / 2.0))
    y1 = int(max(0, cy - bh / 2.0))
    x2 = int(min(w - 1, cx + bw / 2.0))
    y2 = int(min(h - 1, cy + bh / 2.0))
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def track_kps(prev_gray, gray, prev_kps):
    # prev_kps: (5,2) float32
    p0 = prev_kps.reshape(-1, 1, 2).astype(np.float32)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(LK_WINSZ, LK_WINSZ),
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA
    )
    st = st.reshape(-1)
    good = st == 1
    if np.count_nonzero(good) < 4:
        return None, 0.0
    kps = p1.reshape(-1, 2)
    good_ratio = float(np.count_nonzero(good)) / float(len(st))
    return kps, good_ratio

class CameraThread:
    def __init__(self, index=0, w=1280, h=720, mirror=True):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.mirror = mirror
        self.q = deque(maxlen=1)
        self.stopped = False
        self.t = Thread(target=self._loop, daemon=True)

    def start(self):
        self.t.start()
        return self

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            self.q.append(frame)

    def read(self):
        if not self.q:
            return False, None
        return True, self.q[-1].copy()

    def stop(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)  # 過剰スレッドを抑制しスケジューリングを安定化

    # --- Provider（CoreML優先） ---
    available = ort.get_available_providers()
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] \
        if "CoreMLExecutionProvider" in available else ["CPUExecutionProvider"]
    print("ONNX Runtime providers:", available)
    print("Using providers:", providers)

    # --- FaceAnalysis：ライブ検出専用（認識は切る） ---
    try:
        app_det = FaceAnalysis(name="buffalo_l",
                               providers=providers,
                               allowed_modules=["detection"])
    except TypeError:
        app_det = FaceAnalysis(name="buffalo_l", providers=providers)
    app_det.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    # 念のため不要モジュールを無効化（存在すれば）
    for k in ("recognition", "genderage", "landmark_3d_68", "landmark_2d_106"):
        try:
            if hasattr(app_det, "models") and k in app_det.models:
                app_det.models.pop(k, None)
        except Exception:
            pass

    # --- FaceAnalysis：ターゲット埋め込み用（1回だけ使用） ---
    app_tgt = FaceAnalysis(name="buffalo_l", providers=providers)
    app_tgt.prepare(ctx_id=0, det_size=DETECTION_SIZE)

    # --- INSwapper ---
    model_path = Path.home() / ".insightface/models/inswapper_128.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    swapper = insightface.model_zoo.get_model(
        str(model_path), providers=providers, download=False
    )

    # --- ターゲット顔の準備（埋め込みをここで確定） ---
    target_img = cv2.imread(TARGET_IMAGE_PATH)
    if target_img is None:
        raise FileNotFoundError(f"target.png が見つかりません: {TARGET_IMAGE_PATH}")
    tfaces = app_tgt.get(target_img, max_num=1)
    if not tfaces:
        raise RuntimeError("target.png から顔を検出できませんでした。")
    target_face = largest_face(tfaces)
    # app_tgtは以降不要（解放してもOK）
    del app_tgt

    # --- カメラ開始 ---
    cam = CameraThread(CAM_INDEX, FRAME_W, FRAME_H, MIRROR_VIEW).start()

    start_t = time.time()
    prev_gray = None
    prev_kps = None
    last_face = None
    frame_idx = 0
    last_detect_frame = -999
    fps_t0 = time.time()
    fps_counter = 0
    fps = 0.0

    while True:
        ok, frame = cam.read()
        if not ok:
            cv2.waitKey(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        need_detect = (last_face is None) or \
                      (frame_idx - last_detect_frame >= DETECTION_INTERVAL)

        if not need_detect and prev_gray is not None and prev_kps is not None:
            # LKで5点kps追跡
            tracked_kps, good_ratio = track_kps(prev_gray, gray, prev_kps)
            if tracked_kps is None or good_ratio < TRACK_GOOD_MIN_RATIO:
                need_detect = True
            else:
                # 追跡結果でfaceを更新
                last_face.kps = tracked_kps.astype(np.float32)
                last_face.bbox = kps_to_bbox(
                    last_face.kps, w, h, expand=BBOX_EXPAND
                ).astype(np.float32)

        if need_detect:
            f = detect_with_roi(app_det, frame, last_face.bbox if last_face is not None else None)
            if f is not None:
                last_face = f
                prev_kps = f.kps.astype(np.float32)
                last_detect_frame = frame_idx
            # fがNoneでも前フレの追跡で乗り切れることがあるので即スキップしない

        prev_gray = gray

        if last_face is None:
            # 顔が無い間はそのまま表示
            cv2.imshow("Boundary Morph (Fast) - Press q to quit, r to reset", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_idx += 1
            continue

        # スワップ（背景は変わらない）
        swapped = swapper.get(frame.copy(), last_face, target_face, paste_back=True)

        # 20秒モーフ（全体αブレンド：背景は同一のため顔だけが変わる）
        elapsed = time.time() - start_t
        alpha = ease_in_out_smoothstep(elapsed / MORPH_DURATION_SEC)
        out = cv2.addWeighted(frame, 1.0 - alpha, swapped, alpha, 0.0)

        # FPS表示
        fps_counter += 1
        if (time.time() - fps_t0) >= 0.5:
            fps = fps_counter / (time.time() - fps_t0)
            fps_counter = 0
            fps_t0 = time.time()

        cv2.putText(out, f"alpha={alpha:.2f}  fps={fps:.1f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Boundary Morph (Fast) - Press q to quit, r to reset", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            start_t = time.time()

        frame_idx += 1

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
