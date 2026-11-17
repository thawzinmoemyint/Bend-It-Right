# Bend it right - Workout and Yoga Pose Monitoring Application

A desktop application for real-time **workout repetition counting** and **yoga pose monitoring** using YOLOv11 Pose and MediaPipe Pose, built with Python and Tkinter. This is the project for CSC-340 Artifical Intelligence Course. 

Team Members: Akari Kyaw Thein, Ant Bone Kyaw, Han Win Aung, Nay Chi Lin Lei, Thaw Zin Moe Myint.

## Features

- **Workout Tracker**
  - Live camera or video upload
  - Real-time repetition counting for:
    - Push Up
    - Curl Up
    - Jumping Jack
    - Squat
  - Visual feedback and rep counters for each exercise

- **Yoga Detection**
  - Live camera or video upload
  - Real-time yoga pose recognition using YOLOv11
  - Supported poses: Goddess, Plank, Tree, Warrior II
  - Timer for each pose (tracks how long each pose is held)

- **User Interface**
  - Modern Tkinter GUI with tabbed navigation
  - Visual pose/exercise buttons and counters
  - Easy reset and upload options

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/thompson129/Bend-It-Right.git
    cd Bend-It-Right
    ```

2. **Install the dependencies** (Python 3.10–3.12; MediaPipe's `solutions` API is not available on 3.13)
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**
    ```bash
    python main_GUI.py
    ```

### Web version (runs in the browser)

The `web/` folder is a browser version of the app. Both models run on the visitor's device
(the yoga model via ONNX Runtime Web, workouts via MediaPipe), so it can be hosted for free as static files
and camera video never leaves the device. `web/model/yoga.onnx` is `models/best.pt` exported with
`YOLO("models/best.pt").export(format="onnx", imgsz=640, nms=True)`.

Try it locally (webcam access needs `localhost` or HTTPS):

```bash
python3 -m http.server 8000 --directory web
```

Then open http://localhost:8000. Adding `?mode=yoga&video=<url>` opens a tab and plays a video from a URL.

Publish it as a free static Hugging Face Space:

```bash
pip install huggingface_hub
hf auth login
python deploy_to_hf.py <your-hf-username>/bend-it-right
```

---

## Usage

- **Workout Tracker Tab**
  - Click "Live Detection" to use your webcam, or "Upload Videos" to analyze a video file.
  - Click on an exercise button (Push Up, Curl Up, Jumping Jack, Squat) to start counting reps.
  - The counter updates in real time as you perform the exercise.
  - Click "Reset" to reset all counters.

- **Yoga Detection Tab**
  - Click "Live Detection" or "Upload Videos" to start.
  - The app will recognize supported yoga poses and display a timer for each pose held.
  - Click "Reset" to reset all timers.

---

## Camera Setup Tips

- For **workout counting** (especially squats), position the camera **side-on** at waist height for best results.
- Ensure your full body is visible and the background is uncluttered.
- Good lighting improves detection accuracy.

---

## Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for pose and object detection
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI

---


**Enjoy tracking your workouts and yoga poses!**