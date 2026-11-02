# AR-Face-Filter
A webcam-based Augmented Reality (AR) application that detects facial landmarks and overlays 3D/2D filters in real-time.
# AR Face Filter System 🎭

A real-time Augmented Reality (AR) face filter application built with Python, OpenCV, and MediaPipe. It tracks 468 facial landmarks through a webcam and overlays virtual accessories.

## ✨ Features
- Real-time face tracking (468 landmarks)
- 4 different filters: Sunglasses, Mustache, Hat, and Dog Ears
- Smooth scaling and rotation based on face angle
- Take snapshots using the 'S' key
- Easy keyboard controls to switch filters

## 🛠️ Technologies Used
- Python 3.12
- OpenCV (`cv2`)
- MediaPipe (Face Mesh)
- NumPy

## 🚀 How to Run
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt` *(You need to create this file)*
5. Run the app: `python main.py`

## 🕹️ Controls
- **1** : Sunglasses
- **2** : Mustache
- **3** : Hat
- **4** : Dog Ears
- **S** : Save Snapshot
- **Q** : Quit
