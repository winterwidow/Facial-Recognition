# encode_faces.py
import os
import cv2
import face_recognition
import pickle

DATA_DIR = "data"
OUT_DIR = "encodings"
OUT_FILE = os.path.join(OUT_DIR, "encodings.pickle")
FILENAME = os.path.join(OUT_DIR, "filenames.txt")  # to save the filename
MODEL = "cnn"  # "hog" for CPU, "cnn" if you have GPU and dlib compiled with CUDA

os.makedirs(OUT_DIR, exist_ok=True)

encodings = []
names = []
filenames = []


def is_image_file(filename):
    lower = filename.lower()
    return lower.endswith((".jpg", ".jpeg", ".png", ".bmp"))


for person_name in sorted(os.listdir(DATA_DIR)):
    person_dir = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for fname in sorted(os.listdir(person_dir)):
        if not is_image_file(fname):
            continue
        path = os.path.join(person_dir, fname)
        image_bgr = cv2.imread(path)

        if image_bgr is None:
            print(f"Warning: could not read {path}")
            continue
        # image = image_bgr[:, :, ::-1]  # BGR -> RGB
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (2 * 224, 2 * 224))

        try:
            boxes = face_recognition.face_locations(image, model=MODEL)
            print(boxes)

            feats = face_recognition.face_encodings(image, boxes)

            if len(feats) == 0:
                print(f"No face found in {path}")
            for f in feats:
                encodings.append(f)
                names.append(person_name)
                filenames.append(fname)
                """print(
                    f"Encoded {path} as {person_name} with features as {f} with lenght of {len(f)}"
                )"""
        except Exception as e:
            print(f"Error processing {path}: {e}")

data = {"encodings": encodings, "names": names, "filenames": filenames}
# data2 = {"filename": filenames}
with open(OUT_FILE, "wb") as f:
    pickle.dump(data, f)
"""
#save filenames
with open(FILENAME, "w") as f1:
    for fname in filenames:
        f1.write(fname + "\n")
"""


print(f"Saved {len(encodings)} face encodings to {OUT_FILE}")
print(f"saved {len(filenames)} filenames to {FILENAME}")
