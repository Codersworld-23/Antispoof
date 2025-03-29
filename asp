import cv2
import cvzone
import os
import pickle
from PIL import Image
from deepface import DeepFace
import tensorflow
import keras
# import tf_keras from tensorflow.keras
# from tensorflow.keras import tf_keras


def is_live_face(image):    
    try:
        result = DeepFace.analyze(image, actions=['antispoofing'])
        # DeepFace returns a list of dictionaries, even for a single face.
        # We'll assume the first detected face.
        spoof = result[0]['antispoofing']['spoof']
        #Deepface returns a boolean, true = spoof. Therefore we must invert it.
        return not spoof

    except ValueError as e:
        print(f"Error: {e}") #Prints if no face found.
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

# Example usage with live camera:
def live_camera_antispoof():
    cap = cv2.VideoCapture(1)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        live = is_live_face(frame)
        if live:
            print("Live face detected.")
            cv2.putText(frame, "ok", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:
            print("Spoof detected or no face found.")
            cv2.putText(frame, "x", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_camera_antispoof()