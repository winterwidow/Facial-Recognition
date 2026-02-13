# recognize_image.py
import cv2
import pickle
import face_recognition
import numpy as np
import os

ENC_FILE = os.path.join("encodings", "encodings.pickle")
# IMAGE_PATH = r"C:\Users\rohan\Python\facial_recog\data\unknown\test.jpg"  # change to your image path
IMAGE_PATH = (
    r"C:\Users\rohan\Python\facial_recog\test\family.jpg"  # change to your image path
)
TOLERANCE = 0.5  # lower = stricter match

# Load encodings
if not os.path.exists(ENC_FILE):
    raise FileNotFoundError(
        f"Encodings file not found: {ENC_FILE}. Run encode_faces.py first."
    )

with open(ENC_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data.get("encodings", [])
known_names = data.get("names", [])
known_files = data.get("filenames", [])

# Load target image
img = cv2.imread(IMAGE_PATH)
# cv2.imshow("Recognition Result0", img)
#'''
if img is None:
    raise ValueError(f"Could not read image at {IMAGE_PATH}")

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces
boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)
# print(f"boxes {boxes} encoding {encodings}")

# Match each face
for (top, right, bottom, left), face_enc in zip(boxes, encodings):
    matches = face_recognition.compare_faces(
        known_encodings, face_enc, tolerance=TOLERANCE
    )
    print(f"printing matches {matches}")
    face_distances = face_recognition.face_distance(known_encodings, face_enc)
    # print(f"found matches# of {len(matches)} face distance is {len(face_distances)} ")
    name = "Unknown"
    # filename = "unknown"
    if len(face_distances) > 0:
        best_idx = np.argmin(face_distances)
        if matches[best_idx]:
            name = known_names[best_idx]
            filename = known_files[best_idx]
            print(
                f"found matches# of {len(matches)} face distance is {len(face_distances)} best index is {best_idx} name is {name} and filename is {filename}"
            )
        else:
            print("no match found")
            filename = None
    else:
        print("no encodings to compare")
        filename = None
    # print(f"found matches# of {len(matches)} face distance is {len(face_distances)} best index is {best_idx} name is {name} and filename is {filename}")
    # print(f"found matches# of {len(matches)} face distance is {len(face_distances)} best index is {best_idx} name is {name}")
    # Draw results
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(
        img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

# Show output
cv2.imshow("Recognition Result", img)
#'''
cv2.waitKey(0)
cv2.destroyAllWindows()
