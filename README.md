"# Face_Recognition_app" 
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

