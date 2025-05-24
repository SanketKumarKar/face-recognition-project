# Face Recognition Project

A real-time face recognition system built with Flask, OpenCV, and PyTorch, capable of detecting and recognizing faces in images and video streams.
The Above is Python Based project. To test live demo of FACE RECOGNITION visit client side based project.
Link for live demo:- https://cyberface-scan.netlify.app/
Link of GitHub Repository:- https://github.com/SanketKumarKar/cyberpunk-face-recognition

## Features

- Real-time face detection and recognition via webcam
- Video file processing with face detection
- Face alignment using facial landmarks
- Snapshot capture functionality
- Web interface for easy interaction

## System Requirements

- Python 3.8 or higher (Python 3.13 compatible)
- Webcam (for real-time recognition)
- Operating System: Windows, macOS, or Linux

## Installation Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/SanketKumarKar/face-recognition-project
cd face-recognition-project
```

### Step 2: Set Up Virtual Environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

#### Special Instructions for Different Operating Systems:

##### Windows:
- If you encounter issues with dlib installation, install CMake first:
  ```bash
  pip install cmake
  pip install dlib
  ```
- Make sure Visual C++ build tools are installed

##### macOS:
- You might need to install XCode Command Line Tools:
  ```bash
  xcode-select --install
  ```

##### Linux:
- Install the following packages before installing requirements:
  ```bash
  sudo apt-get update
  sudo apt-get install -y cmake build-essential libopenblas-dev
  ```

### Step 4: Download the Face Recognition Model (Optional)

The application will work without a pre-trained model, but will only detect faces without recognizing them.

To include face recognition:
1. Place your trained model file in the project root directory
2. For PyTorch models, name it `face_recognition_model.pt`

## Usage Instructions

### Running the Web Application

1. Activate your virtual environment if it's not already activated
2. Start the Flask application:

```bash
python app.py
```

3. Open your web browser and navigate to: `http://127.0.0.1:5000`

### Using the Web Interface

1. **Live Face Recognition**:
   - The home page shows your webcam feed with real-time face detection
   - Detected faces will be highlighted with a blue rectangle
   - If a trained model is loaded, recognized faces will be labeled

2. **Take Snapshots**:
   - Click the "Take Snapshot" button to capture the current frame
   - Snapshots will be saved in the `static/snapshots` directory

3. **Process Video Files**:
   - Click "Choose File" to select a video for processing
   - Click "Upload" to process the video
   - Results will be displayed after processing completes

## Training Your Own Model

To train the system to recognize specific faces:

1. Collect images of faces you want to recognize
2. Organize them into folders, one folder per person
3. Run the training script:

```bash
python train_model.py --data_dir path/to/face/images
```

## Troubleshooting

### Camera Access Issues
- Make sure your browser has permission to access the camera
- Close other applications that might be using your camera

### Model Loading Issues
- Verify that your model file is in the correct format for PyTorch
- Check console logs for specific error messages

### Installation Problems
- For dlib issues, make sure you have the proper build tools installed
- Use `pip install -v` for verbose output to identify specific errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
