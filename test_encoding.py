import face_recognition
import cv2

img = cv2.imread("data/unknown/001_9cd1160a.jpg")
if img is None:
    raise ValueError("Image not loaded correctly")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 1: detect face locations
boxes = face_recognition.face_locations(rgb, model="hog")

# Step 2: compute encodings only if faces are found
if boxes:
    encodings = face_recognition.face_encodings(rgb, boxes)
    print("Encodings:", encodings)
else:
    print("No faces found in this image")
