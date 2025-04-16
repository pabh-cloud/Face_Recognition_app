
# 🔐 Smart Security Camera System with Face Recognition and YOLO

A real-time intelligent surveillance system using OpenCV, YOLOv8, and FaceNet to identify and differentiate **insiders** (authorized individuals) from **outsiders** (unregistered or unknown individuals). Designed for secure areas such as labs, homes, or offices.

## 📌 Features

- 🎥 Real-time webcam/video stream processing
- 👤 Face recognition using **FaceNet (InceptionResnetV1)**
- 🧠 Person detection with **YOLOv8 (Ultralytics)**
- 🏷️ Insider vs Outsider tagging
- 🗂️ Face registration and dataset management
- 🖼️ GUI built with **Tkinter**
- ⚠️ Alert or highlight unknown individuals

---

## 🛠️ Tech Stack

| Tool/Library       | Purpose |
|--------------------|---------|
| `OpenCV`           | Video processing & face detection |
| `FaceNet` (facenet-pytorch) | Face embeddings |
| `YOLOv8` (Ultralytics) | Person/object detection |
| `Tkinter`          | GUI interface |
| `Pandas`/`NumPy`   | Data handling |
| `PyTorch`          | Neural network backend |
| `Scikit-learn`     | Cosine similarity calculation |
| `PIL` (Pillow)     | Image display |

---

🧑‍💻 How It Works
1. Register Faces: Use the GUI to register insiders by name and image.

2. Run Surveillance: The webcam starts detecting people using YOLO.

3. Face Recognition: If a detected person matches a registered face using FaceNet + cosine similarity → tagged as Insider.

4. Alert on Outsiders: Unknown faces are tagged as Outsiders and automatically saved for review.

🚀 Getting Started

1️⃣ Create Virtual Environment & Activate It

On Windows:

python -m venv tf

tf\Scripts\activate

On Linux / macOS:

python3 -m venv tf

source tf/bin/activate


2️⃣ Download YOLOv8 Face Detection Model

Download the pretrained YOLOv8 model for face detection from the following GitHub repository:

🔗 YOLOv8 Face Detection – lindevs/yolov8-face

Clone or download the model weights and place them in your project directory or an accessible location within your environment.

3️⃣ Install Required Libraries

Once your virtual environment is active, install all the required Python packages:

pip install -r requirements.txt

This will install OpenCV, PyTorch, YOLO, FaceNet, and other essential libraries used in the project.
