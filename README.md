# Person Finder AI

A powerful facial recognition system that can detect and track specific individuals in images and videos using deep learning.

## Features

- Face detection and recognition using DeepFace and FaceNet
- Support for both image and video processing
- Multiple reference photos for improved accuracy
- Real-time confidence scoring
- Progress tracking with detailed statistics
- Support for multiple target photos to improve recognition accuracy
- Visual bounding boxes with confidence percentages
- Processing history for improved detection stability

## Requirements

```
opencv-python
numpy
deepface
tqdm
pathlib
```

## Project Structure

```
Person_Finder_AI/
├── input/              # Input images and videos
│   └── Adam/          # Example input directory
├── output/            # Processed outputs
│   └── Adam/         # Example output directory
├── photo_target/      # Reference photos
│   └── AdamSandler/  # Example target photos
├── main.py           # Main application code
└── README.md         # This file
```

## Usage

1. Place your target person's reference photos in the `photo_target/[PersonName]` directory
2. Place the input image or video in the `input/[PersonName]` directory
3. Update the paths in `main.py` if needed
4. Run the script:

```bash
python main.py
```

## Configuration

You can adjust the following parameters in the code:

- `face_threshold`: Controls the strictness of face matching (default: 0.5)
- `history_size`: Number of frames to keep in detection history (default: 15)

## Output

The system will generate:
- Annotated images/videos with bounding boxes
- Confidence scores for each detection
- Processing statistics including:
  - Total detections
  - Processing time
  - Average FPS (for videos)
  - Final detection count

## Example Output Format

- Images: `output/[PersonName]/detection_output[N].jpg`
- Videos: `output/[PersonName]/detections_output[N].mp4`

## Notes

- Higher quality reference photos will yield better results
- Multiple reference photos from different angles improve detection accuracy
- The system uses RetinaFace for face detection and FaceNet for face recognition
- Processing speed depends on hardware capabilities and input resolution 
