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

    def add_arbitrary_sample(self, cam_xy, screen_xy):
        if cam_xy is None:
            return False
        self.samples_cam.append([cam_xy[0], cam_xy[1], 1.0])
        self.samples_screen.append(screen_xy)
        if len(self.samples_cam) >= 3:
            self._fit()
            self.active = False
            return True
        return False

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


def draw_gaze_screen(calibrator, screen_w, screen_h, screen_xy=None,
                    raw_cam=None, ema_cam=None, manual_offset=(0.0, 0.0),
                    arbitrary_count=0):
    screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if calibrator.active:
        for idx, (x, y) in enumerate(calibrator.targets):
            point = (int(round(x)), int(round(y)))
            color = (0, 255, 255) if idx == calibrator.target_idx else (120, 120, 120)
            cv.circle(screen, point, 26, color, 3, cv.LINE_AA)
            cv.circle(screen, point, 5, color, -1, cv.LINE_AA)

        cv.putText(
            screen,
            "Look at the highlighted circle and press SPACE",
            (30, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
    else:
        if calibrator.matrix is None:
            status = "Uncalibrated - approximate gaze; press C to calibrate"
        else:
            status = "Calibrated gaze tracking"
        cv.putText(
            screen,
            status,
            (30, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        if screen_xy is not None:
            point = (int(round(screen_xy[0])), int(round(screen_xy[1])))
            cv.circle(screen, point, 60, (0, 255, 0), -1, cv.LINE_AA)

        y = 85
        if raw_cam is not None:
            cv.putText(screen, f"Raw cam: x={raw_cam[0]:+.2f} y={raw_cam[1]:+.2f}",
                       (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (180, 180, 255), 2, cv.LINE_AA)
        y += 35
        if ema_cam is not None:
            cv.putText(screen, f"EMA cam: x={ema_cam[0]:+.2f} y={ema_cam[1]:+.2f}",
                       (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (180, 255, 180), 2, cv.LINE_AA)
        y += 35
        if screen_xy is not None:
            cv.putText(screen, f"Screen px: x={int(screen_xy[0])} y={int(screen_xy[1])}",
                       (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2, cv.LINE_AA)
        y += 35
        cv.putText(screen, f"Manual offset: x={int(manual_offset[0])} y={int(manual_offset[1])}",
                   (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 180, 180), 2, cv.LINE_AA)
        y += 35
        cv.putText(screen, f"Click-calibration samples: {arbitrary_count}",
                   (30, y), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(screen, "Arrows: nudge | Left click: calibrate | C: 9-point routine",
                   (30, y + 35), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2, cv.LINE_AA)

    return screen


def approximate_screen_position(cam_xy, screen_w, screen_h):
    range_x_cm = 15.0
    range_y_cm = 10.0
    x = screen_w / 2 + cam_xy[0] * screen_w / (2 * range_x_cm)
    y = screen_h / 2 + cam_xy[1] * screen_h / (2 * range_y_cm)
    return (
        float(np.clip(x, 0, screen_w - 1)),
        float(np.clip(y, 0, screen_h - 1)),
    )


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


def make_crop_preview(face_crop, left_crop, right_crop, screen_w, screen_h):
    crops = []
    for label, crop in (
        ("Face", face_crop),
        ("Left eye", left_crop),
        ("Right eye", right_crop),
    ):
        if crop is None:
            panel = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            panel = cv.resize(crop, (224, 224), interpolation=cv.INTER_NEAREST)
        cv.putText(panel, label, (12, 28), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 255), 2, cv.LINE_AA)
        crops.append(panel)
    preview = np.hstack(crops)
    preview = cv.resize(preview, (screen_w, screen_h), interpolation=cv.INTER_NEAREST)
    cv.putText(preview, "Press F to return to gaze fullscreen",
               (30, screen_h - 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    return preview


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
    raw_cam = None
    manual_offset = np.zeros(2, dtype=np.float32)
    arbitrary_calibration_count = 0
    clicked_position = {"value": None}
    crop_preview_mode = False
    latest_crops = {"face": None, "left": None, "right": None}
    gaze_window = "Gaze Screen"
    gaze_window_open = False

    def mouse_calibration(event, mouse_x, mouse_y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            clicked_position["value"] = (float(mouse_x), float(mouse_y))

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
            screen_xy = None
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

                latest_crops["face"] = face_crop
                latest_crops["left"] = left_crop
                latest_crops["right"] = right_crop

                face_blob = preprocess_crop(face_crop, mean_face)[np.newaxis, ...]
                left_blob = preprocess_crop(left_crop, mean_left)[np.newaxis, ...]
                right_blob = preprocess_crop(right_crop, mean_right)[np.newaxis, ...]
                grid = make_face_grid(face_rect, frame_w, frame_h).reshape(1, 625, 1, 1)

                net.setInput(left_blob, "image_left")
                net.setInput(right_blob, "image_right")
                net.setInput(face_blob, "image_face")
                net.setInput(grid, "facegrid")

                pred = net.forward("fc3").reshape(-1)
                raw_cam = (float(pred[0]), float(pred[1]))
                cam_xy = raw_cam

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
                if screen_xy is None:
                    screen_xy = approximate_screen_position(cam_xy, screen_w, screen_h)
                screen_xy = (
                    float(np.clip(screen_xy[0] + manual_offset[0], 0, screen_w - 1)),
                    float(np.clip(screen_xy[1] + manual_offset[1], 0, screen_h - 1)),
                )
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

            nudge = 10
            if key == 81:
                manual_offset[0] -= nudge
            elif key == 83:
                manual_offset[0] += nudge
            elif key == 82:
                manual_offset[1] -= nudge
            elif key == 84:
                manual_offset[1] += nudge

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

            if key == ord("f"):
                crop_preview_mode = not crop_preview_mode
                if crop_preview_mode:
                    cv.destroyWindow(gaze_window)
                    gaze_window_open = False
                    cv.namedWindow("Crop Preview", cv.WINDOW_NORMAL)
                    cv.setWindowProperty("Crop Preview", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                    print("Crop preview enabled.")
                else:
                    cv.destroyWindow("Crop Preview")
                    print("Crop preview disabled.")

            if clicked_position["value"] is not None and cam_xy is not None:
                screen_sample = clicked_position["value"]
                clicked_position["value"] = None
                if calibrator.add_arbitrary_sample(cam_xy, screen_sample):
                    print("Click calibration fitted.")
                arbitrary_calibration_count += 1
                print(f"Captured click-calibration sample {arbitrary_calibration_count} at {screen_sample}")

            if crop_preview_mode:
                preview_frame = make_crop_preview(
                    latest_crops["face"], latest_crops["left"], latest_crops["right"],
                    screen_w, screen_h,
                )
                cv.imshow("Crop Preview", preview_frame)
            else:
                if not gaze_window_open:
                    cv.namedWindow(gaze_window, cv.WINDOW_NORMAL)
                    cv.setWindowProperty(gaze_window, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                    cv.setMouseCallback(gaze_window, mouse_calibration)
                    gaze_window_open = True

                cv.imshow(
                    gaze_window,
                    draw_gaze_screen(
                        calibrator,
                        screen_w,
                        screen_h,
                        screen_xy,
                        raw_cam,
                        ema_cam,
                        manual_offset,
                        arbitrary_calibration_count,
                    ),
                )


if __name__ == "__main__":
    main()
