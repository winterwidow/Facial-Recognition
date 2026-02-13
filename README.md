# Facial Recognition System

A Python-based facial recognition system that uses deep learning to detect, encode, and identify faces in images. The system employs a two-stage pipeline: encoding known faces and recognizing faces in new images.

## Overview

This facial recognition system uses the `face_recognition` library (built on dlib's state-of-the-art face recognition) to create 128-dimensional face encodings and match them against a database of known faces.

## How It Works

### Architecture

The system consists of two main components:

#### 1. **Face Encoding (`encode_faces.py`)**  
This script processes a dataset of known faces and creates a database of face encodings.

**Process:**
1. **Image Loading**: Scans the `data/` directory organized by person name (each subdirectory represents a person)
2. **Preprocessing**:  
   - Converts images from BGR to RGB color space  
   - Resizes images to 448x448 pixels (2 × 224) for optimal detection
3. **Face Detection**: Uses either CNN or HOG model to locate faces in images  
   - **CNN model**: More accurate, requires GPU with CUDA support  
   - **HOG model**: Faster, works on CPU
4. **Feature Extraction**: Generates 128-dimensional face encodings for each detected face
5. **Serialization**: Saves all encodings, names, and filenames to `encodings/encodings.pickle`

**Key Features:**
- Supports multiple image formats (JPG, JPEG, PNG, BMP)
- Handles multiple faces per image
- Error handling for unreadable images or missing faces
- Configurable detection model (CNN/HOG)

#### 2. **Face Recognition (`recognize.py`)**  
This script identifies faces in new images by comparing them against the encoded database.

**Process:**
1. **Load Encodings**: Retrieves the pre-computed face encodings from the pickle file
2. **Image Processing**: Loads and converts the target image to RGB
3. **Face Detection**: Locates all faces in the image using HOG model
4. **Face Encoding**: Generates 128-dimensional encodings for detected faces
5. **Matching:**  
   - Compares each face encoding against all known encodings  
   - Calculates Euclidean distance between encodings  
   - Uses configurable tolerance threshold (default: 0.5)
6. **Identification:**  
   - Selects the best match with minimum distance  
   - Labels face as "Unknown" if no match meets tolerance threshold
7. **Visualization:**  
   - Draws bounding boxes around detected faces  
   - Labels each face with the recognized name

**Key Features:**
- Adjustable tolerance for match strictness (lower = stricter)
- Distance-based matching for better accuracy
- Real-time visual feedback with bounding boxes
- Handles multiple faces in a single image

### The Face Recognition Algorithm

The system uses **dlib's ResNet-based deep learning model** which:
- Generates a unique 128-dimensional vector (encoding) for each face
- These encodings capture distinctive facial features
- Faces of the same person produce similar encodings
- Face matching uses Euclidean distance between encodings:
  - Distance < 0.5: Strong match (same person)
  - Distance 0.5-0.6: Possible match
  - Distance > 0.6: Different person

## Features

✅ **High Accuracy**: Uses state-of-the-art deep learning models  
✅ **Flexible Detection**: Support for both CPU (HOG) and GPU (CNN) processing  
✅ **Multi-Face Processing**: Detects and recognizes multiple faces in a single image  
✅ **Tolerance Control**: Adjustable matching threshold for precision tuning  
✅ **Visual Output**: Displays results with labeled bounding boxes  
✅ **Scalable**: Easily add new people by organizing images in folders  
✅ **Error Handling**: Robust handling of corrupted images and edge cases  

## Installation

### Prerequisites
- Python 3.6+
- pip package manager
- (Optional) CUDA-enabled GPU for CNN model

### Setup

```bash
# Clone the repository
git clone https://github.com/winterwidow/Facial-Recognition.git
cd Facial-Recognition

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `opencv-python`: Image processing and visualization
- `face_recognition`: Face detection and encoding (wraps dlib)
- `numpy`: Numerical operations
- `imutils`: Image manipulation utilities

## Usage

### Step 1: Prepare Your Dataset

Organize your training images in the `data/` directory:

```
data/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.png
├── person2/
│   ├── img1.jpg
│   └── img2.jpg
└── person3/
    └── portrait.jpg
```

Each subdirectory name becomes the label for that person.

### Step 2: Encode Faces

Run the encoding script to process your dataset:

```bash
python encode_faces.py
```

**Configuration Options (in `encode_faces.py`):**
- `DATA_DIR`: Directory containing person subdirectories (default: "data")
- `OUT_DIR`: Output directory for encodings (default: "encodings")
- `MODEL`: Detection model - "cnn" for GPU or "hog" for CPU

**Output:**
- Creates `encodings/encodings.pickle` containing all face data
- Prints the number of faces successfully encoded

### Step 3: Recognize Faces

Modify the `IMAGE_PATH` in `recognize.py` to point to your test image, then run:

```bash
python recognize.py
```

**Configuration Options (in `recognize.py`):**
- `IMAGE_PATH`: Path to the image to analyze
- `TOLERANCE`: Matching threshold (default: `0.5`)
  - Lower values = stricter matching
  - Recommended range: 0.4-0.6

**Output:**
- Opens a window displaying the image with:
  - Green bounding boxes around detected faces
  - Labels showing recognized names or "Unknown"
- Press any key to close the window

## Performance Considerations

### Detection Models

| Model | Speed | Accuracy | Hardware |
|-------|-------|----------|----------|
| HOG | Fast | Good | CPU |
| CNN | Slower | Excellent | GPU (CUDA) |

**Recommendation:**
- Use **HOG** for quick processing or CPU-only systems
- Use **CNN** for maximum accuracy with GPU support

### Optimization Tips

1. **Image Quality**: Higher resolution images improve detection accuracy
2. **Lighting**: Well-lit, front-facing photos work best
3. **Training Data**: 3-5 varied images per person recommended
4. **Tolerance**: Fine-tune based on your use case:
   - Security applications: 0.4-0.45 (strict)
   - General recognition: 0.5-0.55 (balanced)
   - Lenient matching: 0.6+ (may increase false positives)

## Troubleshooting

**"No face found in [image]"**
- Ensure the face is clearly visible and well-lit
- Try increasing image resolution
- Face may be at extreme angle

**"Encodings file not found"**
- Run `encode_faces.py` first to create the encodings database

**Poor recognition accuracy**
- Add more training images per person
- Adjust tolerance threshold
- Ensure training images have similar lighting/angles to test images

## Project Structure

```
Facial-Recognition/
├── encode_faces.py      # Creates face encoding database
├── recognize.py         # Recognizes faces in images
├── requirements.txt     # Python dependencies
├── data/               # Training images (organized by person)
├── encodings/          # Generated face encodings (pickle files)
└── README.md           # This file
```

## Future Enhancements

- [ ] Real-time video stream recognition
- [ ] Web interface for easier interaction
- [ ] Support for video file processing
- [ ] Database integration for larger datasets
- [ ] Confidence scores in recognition output
- [ ] Face detection API endpoint

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- Uses dlib's state-of-the-art face recognition model
- OpenCV for image processing