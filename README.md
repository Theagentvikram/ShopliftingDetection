# 🛡️ Intelligent Shoplifting Detection

## 🖊️ Overview
A cutting-edge deep learning system designed to enhance retail security by detecting and preventing shoplifting in real time. This project leverages YOLO for object detection, DeepSORT for multi-object tracking, and LSTM for activity recognition, ensuring efficient and accurate surveillance.


## 🚀 Features
- **Real-Time Detection**: Identifies  objects such as people and products instantly.
- **Behavioral Analysis**: Recognizes suspicious actions like concealing items.
- **Anomaly Detection**: Flags unusual patterns using autoencoders.
- **Multi-Object Tracking**: Tracks individuals consistently across frames.

---

## 🔧 Prerequisites
Before starting, ensure the following software is installed:

1. **Python 3.8+**
2. **pip** (Python package manager)
3. **CUDA Toolkit** (if using GPU acceleration)
4. Supported deep learning frameworks like TensorFlow or PyTorch.

---

## 🔍 Files to Download
### 1. **Pre-trained YOLO Weights**
   - Download from the official YOLO website or GitHub:
     [YOLOv8 Pre-trained Weights](https://github.com/ultralytics/ultralytics)
   - Save the file in the `src/models/` directory.

### 2. **Sample Datasets**
   - Download annotated datasets for training and testing:
     - [COCO Dataset](https://cocodataset.org/#download)
     - Custom dataset (if available): Place in the `data/` folder.

### 3. **DeepSORT Model Files**
   - Download the appearance feature extractor:
     [DeepSORT Weights](https://github.com/nwojke/deep_sort)
     - Save the `.pb` or `.pth` file in the `src/models/` directory.

### 4. **Pre-trained LSTM Model**
   - If using pre-trained LSTM for activity recognition:
     - Place the file in the `src/models/` directory.

---

## 🚪 Folder Structure
```
Shoplifting-Detection/
│
├── README.md          # Project documentation
├── LICENSE            # License information
├── .gitignore         # Ignored files
├── requirements.txt   # Python dependencies
├── data/              # Dataset folder
│   ├── train/         # Training dataset
│   └── test/          # Testing dataset
├── src/               # Source code
│   ├── models/        # Pre-trained models and weights
│   ├── utils/         # Utility functions
│   └── main.py        # Main script
├── notebooks/         # Jupyter notebooks for experiments
├── results/           # Output logs, results, and images
└── docs/              # Documentation files
```

---

## 🚜 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/username/Shoplifting-Detection.git
cd Shoplifting-Detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add Pre-trained Files
Place the required files (YOLO weights, DeepSORT model, etc.) in their respective folders as described above.

### Step 4: Run the System
Execute the main script:
```bash
python src/main.py
```

---

## 🌐 Usage
- **Real-Time Monitoring**:
   Connect to a live surveillance camera and start detection.
   ```bash
   python src/main.py --mode live
   ```
- **Analyze Recorded Video**:
   ```bash
   python src/main.py --mode video --path data/test/video.mp4
   ```

---

## 🔒 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Contributions
We welcome contributions! Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`feature-name`).
3. Submit a pull request.

---

## 🔎 Acknowledgments
- [YOLO](https://github.com/ultralytics/yolov8) for object detection.
- [DeepSORT](https://github.com/nwojke/deep_sort) for multi-object tracking.
- [COCO Dataset](https://cocodataset.org) for training data.
