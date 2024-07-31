
# Face Recognition

This repository contains a Python-based face recognition system that detects faces from a webcam feed. The system uses dlib and OpenCV libraries for face detection and recognition. At this code it plays specific audio files when certain faces are recognized. The action taken when a face is detected is limitless and can be customized to your needs.

## Demo
https://github.com/user-attachments/assets/cdb8212b-40d0-4cd2-99bd-82d5ba6f4e83

## Overview

The project includes two main scripts:
1. `data_process.py`: Processes images of known faces, encodes them, and saves the encodings to a `.pkl` file.
2. `webcam_face_detect.py`: Loads the encoded face data, detects faces in a webcam feed, and plays specific audio files based on the recognized faces.

## Installation

### Requirements

- Python 3.x
- OpenCV
- dlib
- numpy 
- pygame

### Installing Dependencies

To install the necessary dependencies, you can use `pip`:

```bash
pip install opencv-python dlib numpy pygame
```

### Downloading Models

Ensure you have the required dlib models downloaded and placed in a `models` directory:

- `shape_predictor_68_face_landmarks.dat`: [Download link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- `dlib_face_recognition_resnet_model_v1.dat`: [Download link](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

Unzip these files and place them in a `models` directory at the root of the repository.

## Directory Structure

```
face-recognition/
│
├── models/
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── known_faces/
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── person2/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── other/
│   └── other_files
│
├── scripts/
│   ├── data_process.py
│   └──webcam_face_detect.py
│   
└── processed_data.pkl
```

## Usage

### 1. Processing Known Faces

The `data_process.py` script processes the images in the `known_faces` directory and saves the face encodings to a `.pkl` file.
#### dataset for this projcet 
<p align="left">
  <img src="https://github.com/user-attachments/assets/7097c716-3ee1-4bfa-9a76-ea8e10623486" width="100" height="100">
  <img src="https://github.com/user-attachments/assets/13f2b247-c422-48ae-9056-e140d9f3789e" width="100" height="100">
  <img src="https://github.com/user-attachments/assets/39cacbf4-c4ed-4fa9-b2af-337c5a117c40" width="100" height="100">
  <img src="https://github.com/user-attachments/assets/fee296cc-1022-4505-a35e-716841a342c2" width="100" height="100">
  <img src="https://github.com/user-attachments/assets/bfd4f284-9537-40ad-90bb-da458ff61e12" width="100" height="100">
  <img src="https://github.com/user-attachments/assets/b81787b5-de05-4945-9e2a-a07f87e82c33" width="100" height="100">
</p>

Run the script:

```bash
python data_process.py
```


This will generate a `processed_data.pkl` file containing the encoded faces and their corresponding names. It would take longer time for large number of images. 

### 2. Running the Face Recognition with Webcam

The `webcam_face_detect.py` script loads the face encodings from the `.pkl` file, starts the webcam, and performs face recognition. When a known face is detected, it plays the corresponding audio file.

Run the script:

```bash
python webcam_face_detect.py
```

### Important Details
- Adjust the threshold for face recognition (`0.5` in the code) as needed for your use case.



