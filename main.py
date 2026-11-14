import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RIGHT_EYE = {"left": 33, "right": 133, "top": 159, "bottom": 145, "iris": 468}
LEFT_EYE = {"left": 362, "right": 263, "top": 386, "bottom": 374, "iris": 473}
COLORS = {
    "left": (0, 0, 255),
    "right": (255, 0, 0),
    "top": (0, 255, 255),
    "bottom": (255, 0, 255),
    "iris": (0, 255, 0),
}
HORIZONTAL_WEIGHT = 1.7
UP_THRESHOLD = 4
DOWN_THRESHOLD = 5


def viewing_direction(pts):
    """Return the direction of gaze for one eye from its landmark pixels."""
    left = np.array(pts["left"], dtype=float)
    right = np.array(pts["right"], dtype=float)
    top = np.array(pts["top"], dtype=float)
    bottom = np.array(pts["bottom"], dtype=float)
    iris = np.array(pts["iris"], dtype=float)

    if abs(iris[0] - right[0]) > HORIZONTAL_WEIGHT * abs(iris[0] - left[0]):
        horizontal_direction = "right"
    elif abs(iris[0] - left[0]) > HORIZONTAL_WEIGHT * abs(iris[0] - right[0]):
        horizontal_direction = "left"
    else:
        horizontal_direction = "center"

    # if abs(iris[1] - bottom[1]) > DIRECTION_WEIGHT * abs(iris[1] - top[1]):
    #     vertical_direction = "top"
    # elif abs(iris[1] - top[1]) > DIRECTION_WEIGHT * abs(iris[1] - bottom[1]):
    #     vertical_direction = "bottom"
    # else:
    #     vertical_direction = "center"

    midline = (left[1] + right[1]) / 2
    if abs(top[1] - midline) < DOWN_THRESHOLD:
        vertical_direction = "down"
    elif abs(iris[1] - midline) > UP_THRESHOLD:
        vertical_direction = "up"
    else:
        vertical_direction = "center"

    # Project the iris onto the eye's horizontal and vertical axes.  Using
    # projections makes this work for either eye and for a slightly rotated
    # face/camera view.
    # horizontal = right - left
    # vertical = bottom - top
    # horizontal_ratio = np.dot(iris - left, horizontal) / np.dot(horizontal, horizontal)
    # vertical_ratio = np.dot(iris - top, vertical) / np.dot(vertical, vertical)

    # if horizontal_ratio < 0.35:
    #     horizontal_direction = "left"
    # elif horizontal_ratio > 0.65:
    #     horizontal_direction = "right"
    # else:
    #     horizontal_direction = "center"

    # if vertical_ratio < 0.35:
    #     vertical_direction = "up"
    # elif vertical_ratio > 0.65:
    #     vertical_direction = "down"
    # else:
    #     vertical_direction = "center"

    return horizontal_direction, vertical_direction


def px(face, idx, w, h):
    lm = face[idx]
    return (lm.x * w, lm.y * h)


options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def main():

    # Open the camera
    cap = cv.VideoCapture(0)

    # Check if opened
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Begin camera processing loop
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # Capture frame by frame
            ret, frame = cap.read()

            # Check if can read frame
            if not ret:
                print("Error: Could not read frame. Exiting.")
                break

            # Perform processing here
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamps must be strictly increasing
            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            directions = []

            h, w = frame.shape[:2]
            for face in result.face_landmarks:
                for eye in (RIGHT_EYE, LEFT_EYE):
                    pts = {name: px(face, idx, w, h) for name, idx in eye.items()}

                    directions.append(viewing_direction(pts))

                    for name, (x, y) in pts.items():
                        cv.circle(frame, (int(x), int(y)), 3, COLORS[name], -1)

            if directions:
                horizontal = max(
                    (direction[0] for direction in directions),
                    key=lambda direction: direction != "center",
                )
                vertical = max(
                    (direction[1] for direction in directions),
                    key=lambda direction: direction != "center",
                )
                if horizontal != "center" and vertical != "center":
                    gaze = f"{vertical}-{horizontal}"
                else:
                    gaze = vertical if vertical != "center" else horizontal
                cv.putText(
                    frame,
                    f"Viewing: {gaze}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Display the processed frame
            cv.imshow("Camera Feed", frame)

            # Exit on 'q' key
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    # End
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
