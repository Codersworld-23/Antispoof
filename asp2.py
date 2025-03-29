import cv2
import numpy as np

def is_live_face_opencv(frame, previous_frame=None):
    """
    Basic liveness check using OpenCV (motion detection).

    Args:
        frame: Current camera frame.
        previous_frame: Previous camera frame.

    Returns:
        True if motion is detected, False otherwise.
    """
    if previous_frame is None:
        return True  # Assume live on first frame

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(gray_frame, gray_prev)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    motion = np.sum(thresh)

    return motion > 1000000  # Adjust threshold as needed

def live_camera_antispoof_opencv():
    cap = cv2.VideoCapture(1)
    previous_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        live = is_live_face_opencv(frame, previous_frame)
        if live:
            cv2.putText(frame, "Live", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Spoof?", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame = frame.copy() #important copy, or frames will be the same.

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_camera_antispoof_opencv()