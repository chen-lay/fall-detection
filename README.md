# Fall Detection System

A deep learning-based fall detection system using YOLO11-pose for human pose estimation and fall event detection.

## Features

- Real-time human pose estimation using YOLO11-pose
- Fall detection based on posture analysis
- Graphical user interface for easy operation
- Training and evaluation tools included
- Support for custom dataset training

## Technology Stack

- **Python** - Main programming language
- **YOLO11-pose** - Pose estimation model
- **PyTorch** - Deep learning framework
- **OpenCV** - Image processing
- **PyQt** - Graphical user interface

## Project Structure

```
fall-detection/
├── core/                  # Core algorithm modules
│   ├── detector.py        # Fall detection logic
│   └── pose_analyzer.py   # Pose analysis utilities
├── datasets/              # Dataset directory
│   ├── images/            # Image data (train/val/test)
│   └── labels/            # Annotation data (YOLO format)
├── models/                # Model-related files
├── ui/                    # User interface
│   └── main_window.py     # Main window UI
├── main.py                # Main program entry
├── pull_main.py           # Main program utilities
├── test_image_processing.py  # Image processing test
├── visualize_loss.py      # Loss visualization tool
└── yolo11n-pose.pt       # Pre-trained YOLO11-pose model
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Ultralytics (YOLO)
- OpenCV
- PyQt5

### Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install PyQt5
```

## Usage

### Run the Application

```bash
python main.py
```

### Run Detection on Images

```bash
python pull_main.py
```

### Visualize Training Loss

```bash
python visualize_loss.py
```

## Dataset

The dataset includes fall and non-fall samples with the following structure:

- **Training set**: Fall and normal posture images with YOLO format annotations
- **Validation set**: Images for model evaluation
- **Test set**: Images for testing

### Dataset Format

Images are annotated in YOLO format with keypoint annotations for human pose estimation.

## Model

The system uses **YOLO11n-pose** (nano version) for lightweight and fast pose estimation:

- Model size: ~6MB
- Optimized for real-time applications
- Supports 17 keypoints human pose estimation

## How It Works

1. **Pose Estimation**: YOLO11-pose detects human keypoints from input images/video
2. **Posture Analysis**: Analyzes body posture based on keypoint positions and angles
3. **Fall Detection**: Classifies whether a fall event has occurred based on posture features
4. **Alert/Output**: Provides visual feedback and detection results

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests.
