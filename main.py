import ctypes
import time
from pathlib import Path

import cv2 as cv
import mediapipe as mp
import numpy as np
import scipy.io as sio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_DEF = Path("models/itracker_deploy.prototxt")
MODEL_WEIGHTS = Path("models/snapshots/itracker25x_iter_92000.caffemodel")
MEAN_FACE = Path("models/mean_images/mean_face_224.mat")
MEAN_LEFT = Path("models/mean_images/mean_left_224.mat")
MEAN_RIGHT = Path("models/mean_images/mean_right_224.mat")

INPUT_SIZE = (224, 224)
GRID_SIZE = 25
SMOOTHING = 0.85

# Tight eye contours from MediaPipe Face Mesh.
RIGHT_EYE_INDICES = [33, 133, 159, 145, 160, 144, 158, 153, 173]
LEFT_EYE_INDICES = [362, 263, 386, 374, 387, 373, 385, 380, 398]


def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def load_mean_image(path):
    mat = sio.loadmat(str(path))
    return mat["image_mean"].astype(np.float32) / 255.0


def clamp_rect(x, y, w, h, frame_w, frame_h):
    x = max(0, min(int(x), frame_w - 1))
    y = max(0, min(int(y), frame_h - 1))
    w = max(1, int(w))
    h = max(1, int(h))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return x, y, w, h


def crop_repeating_edge(image, rect):
    x, y, w, h = [int(v) for v in rect]
    frame_h, frame_w = image.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((h, w, 3), dtype=image.dtype)

    top = y0 - y
    left = x0 - x
    bottom = (y + h) - y1
    right = (x + w) - x1

    if top > 0 or left > 0 or bottom > 0 or right > 0:
        crop = cv.copyMakeBorder(crop, top, bottom, left, right, borderType=cv.BORDER_REPLICATE)

    return crop


def rect_from_points(points_xy, frame_w, frame_h, margin=0.15):
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    w = x_max - x_min
    h = y_max - y_min

    x = x_min - margin * w
    y = y_min - margin * h
    w = (1.0 + 2.0 * margin) * w
    h = (1.0 + 2.0 * margin) * h
    return clamp_rect(x, y, w, h, frame_w, frame_h)


def preprocess_crop(crop_bgr, mean_rgb):
    crop_rgb = cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB)
    crop_rgb = cv.resize(crop_rgb, INPUT_SIZE, interpolation=cv.INTER_LINEAR)
    crop_rgb = crop_rgb.astype(np.float32) / 255.0
    crop_rgb = crop_rgb - mean_rgb
    return np.transpose(crop_rgb, (2, 0, 1))


def make_face_grid(face_rect, frame_w, frame_h, grid_size=GRID_SIZE):
    x, y, w, h = face_rect
    scale_x = grid_size / float(frame_w)
    scale_y = grid_size / float(frame_h)

    x_lo = int(round(x * scale_x))
    y_lo = int(round(y * scale_y))
    gw = int(round(w * scale_x))
    gh = int(round(h * scale_y))

    x_hi = x_lo + gw
    y_hi = y_lo + gh

    x_lo = max(0, min(grid_size, x_lo))
    y_lo = max(0, min(grid_size, y_lo))
    x_hi = max(0, min(grid_size, x_hi))
    y_hi = max(0, min(grid_size, y_hi))

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    if x_hi > x_lo and y_hi > y_lo:
        grid[y_lo:y_hi, x_lo:x_hi] = 1.0

    # Match Matlab flatten convention used by the original code.
    return grid.T.reshape(-1)


def landmarks_to_pixels(face_landmarks, frame_w, frame_h):
    pts = np.array([(lm.x * frame_w, lm.y * frame_h) for lm in face_landmarks], dtype=np.float32)
    return pts


class AffineCalibrator:
    def __init__(self, screen_w, screen_h):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.targets = [
            (0.10 * screen_w, 0.10 * screen_h),
            (0.50 * screen_w, 0.10 * screen_h),
            (0.90 * screen_w, 0.10 * screen_h),
            (0.10 * screen_w, 0.50 * screen_h),
            (0.50 * screen_w, 0.50 * screen_h),
            (0.90 * screen_w, 0.50 * screen_h),
            (0.10 * screen_w, 0.90 * screen_h),
            (0.50 * screen_w, 0.90 * screen_h),
            (0.90 * screen_w, 0.90 * screen_h),
        ]
        self.samples_cam = []
        self.samples_screen = []
        self.target_idx = 0
        self.matrix = None
        self.active = False

    def start(self):
        self.samples_cam = []
        self.samples_screen = []
        self.target_idx = 0
        self.matrix = None
        self.active = True

    def add_sample(self, cam_xy):
        if not self.active or self.target_idx >= len(self.targets):
            return
        self.samples_cam.append([cam_xy[0], cam_xy[1], 1.0])
        self.samples_screen.append(self.targets[self.target_idx])
        self.target_idx += 1

        if self.target_idx >= len(self.targets):
            self._fit()
            self.active = False

    def _fit(self):
        a = np.asarray(self.samples_cam, dtype=np.float32)
        b = np.asarray(self.samples_screen, dtype=np.float32)
        coeff, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        self.matrix = coeff

    def predict(self, cam_xy):
        if self.matrix is None:
            return None
        v = np.array([cam_xy[0], cam_xy[1], 1.0], dtype=np.float32)
        out = v @ self.matrix
        x = float(np.clip(out[0], 0, self.screen_w - 1))
        y = float(np.clip(out[1], 0, self.screen_h - 1))
        return x, y

    def status_text(self):
        if self.active and self.target_idx < len(self.targets):
            tx, ty = self.targets[self.target_idx]
            return f"Calibrating {self.target_idx + 1}/{len(self.targets)} target=({int(tx)}, {int(ty)})"
        if self.matrix is not None:
            return "Calibrated"
        return "Uncalibrated"


def build_landmarker_options():
    return vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def load_itracker_net(model_def, model_weights):
    if hasattr(cv.dnn, "readNetFromCaffe"):
        return cv.dnn.readNetFromCaffe(str(model_def), str(model_weights))

    if hasattr(cv.dnn, "readNet"):
        try:
            return cv.dnn.readNet(str(model_weights), str(model_def), "caffe")
        except TypeError:
            return cv.dnn.readNet(str(model_weights), str(model_def))

    raise RuntimeError("OpenCV DNN Caffe loader is unavailable in this build.")


def main():
    if not MODEL_DEF.exists() or not MODEL_WEIGHTS.exists():
        print("Error: Caffe model files were not found.")
        print(f"Expected: {MODEL_DEF} and {MODEL_WEIGHTS}")
        return

    for mean_path in (MEAN_FACE, MEAN_LEFT, MEAN_RIGHT):
        if not mean_path.exists():
            print(f"Error: Missing mean image file: {mean_path}")
            return

    mean_face = load_mean_image(MEAN_FACE)
    mean_left = load_mean_image(MEAN_LEFT)
    mean_right = load_mean_image(MEAN_RIGHT)

    try:
        net = load_itracker_net(MODEL_DEF, MODEL_WEIGHTS)
    except Exception as exc:
        print("Error: Failed to load the Caffe iTracker model via OpenCV DNN.")
        print(f"Details: {exc}")
        print("Install a compatible OpenCV build if this persists (for example: opencv-python<5).")
        return

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    screen_w, screen_h = get_screen_size()
    calibrator = AffineCalibrator(screen_w, screen_h)
    ema_cam = None

    with vision.FaceLandmarker.create_from_options(build_landmarker_options()) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Exiting.")
                break

            frame_h, frame_w = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            cam_xy = None
            if result.face_landmarks:
                face_lm = result.face_landmarks[0]
                all_pts = landmarks_to_pixels(face_lm, frame_w, frame_h)

                face_rect = rect_from_points(all_pts, frame_w, frame_h, margin=0.10)
                right_eye_rect = rect_from_points(all_pts[RIGHT_EYE_INDICES], frame_w, frame_h, margin=0.60)
                left_eye_rect = rect_from_points(all_pts[LEFT_EYE_INDICES], frame_w, frame_h, margin=0.60)

                x, y, w, h = face_rect
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)

                face_crop = crop_repeating_edge(frame, face_rect)
                left_crop = crop_repeating_edge(frame, left_eye_rect)
                right_crop = crop_repeating_edge(frame, right_eye_rect)

                face_blob = preprocess_crop(face_crop, mean_face)[np.newaxis, ...]
                left_blob = preprocess_crop(left_crop, mean_left)[np.newaxis, ...]
                right_blob = preprocess_crop(right_crop, mean_right)[np.newaxis, ...]
                grid = make_face_grid(face_rect, frame_w, frame_h).reshape(1, 625, 1, 1)

                net.setInput(left_blob, "image_left")
                net.setInput(right_blob, "image_right")
                net.setInput(face_blob, "image_face")
                net.setInput(grid, "facegrid")

                pred = net.forward("fc3").reshape(-1)
                cam_xy = (float(pred[0]), float(pred[1]))

                if ema_cam is None:
                    ema_cam = np.array(cam_xy, dtype=np.float32)
                else:
                    ema_cam = SMOOTHING * ema_cam + (1.0 - SMOOTHING) * np.array(cam_xy, dtype=np.float32)
                cam_xy = (float(ema_cam[0]), float(ema_cam[1]))

                cv.putText(
                    frame,
                    f"Cam(cm): x={cam_xy[0]:+.2f} y={cam_xy[1]:+.2f}",
                    (20, 35),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                screen_xy = calibrator.predict(cam_xy)
                if screen_xy is not None:
                    sx, sy = screen_xy
                    cv.putText(
                        frame,
                        f"Screen(px): x={int(sx)} y={int(sy)}",
                        (20, 70),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    marker_x = int((sx / screen_w) * frame_w)
                    marker_y = int((sy / screen_h) * frame_h)
                    cv.circle(frame, (marker_x, marker_y), 8, (0, 255, 0), -1)

            cv.putText(
                frame,
                calibrator.status_text(),
                (20, frame_h - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv.imshow("Gaze iTracker (Caffe 25x)", frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrator.start()
                print("Calibration started. Look at each target and press SPACE.")
            if key == 32 and cam_xy is not None:
                calibrator.add_sample(cam_xy)
                if calibrator.active:
                    idx = calibrator.target_idx
                    total = len(calibrator.targets)
                    print(f"Captured calibration sample {idx}/{total}.")
                elif calibrator.matrix is not None:
                    print("Calibration complete.")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
