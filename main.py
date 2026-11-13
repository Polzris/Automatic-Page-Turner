import cv2 as cv
import numpy as np


def main():

    # Open the camera
    cap = cv.VideoCapture(0)

    # Check if opened
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create sliders for adjusting the Canny thresholds in real time.
    cv.namedWindow("Camera Feed")
    cv.createTrackbar("Lower threshold", "Camera Feed", 50, 255, lambda _: None)
    cv.createTrackbar("Upper threshold", "Camera Feed", 200, 255, lambda _: None)

    # Begin camera processing loop
    while True:
        # Capture frame by frame
        ret, frame = cap.read()

        # Check if can read frame
        if not ret:
            print("Error: Could not read frame. Exiting.")
            break

        # Perform processing here
        frame = cv.cvtColor(
            frame, cv.COLOR_BGR2GRAY
        )  # Example processing: convert to grayscale
        frame = cv.GaussianBlur(
            frame, (5, 5), 0
        )  # Example processing: apply Gaussian blur

        lower_threshold = cv.getTrackbarPos("Lower threshold", "Camera Feed")
        upper_threshold = cv.getTrackbarPos("Upper threshold", "Camera Feed")
        lower_threshold, upper_threshold = sorted((lower_threshold, upper_threshold))

        frame = cv.Canny(
            frame, lower_threshold, upper_threshold
        )  # Apply Canny edge detection
        contours, _ = cv.findContours(
            frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )  # Find contours
        cv.drawContours(
            frame, contours, 1, (255, 0, 0), 3
        )  # Draw contours on the frame

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
