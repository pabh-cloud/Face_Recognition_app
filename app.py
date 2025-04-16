import cv2
import numpy as np
import pandas as pd
import os
import time
import torch
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO  # Import YOLO from ultralytics

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8-FaceNet Insider Detection System")
        self.root.geometry("1000x600")
        self.root.resizable(True, True)
        
        # Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/registered', exist_ok=True)
        os.makedirs('data/detected', exist_ok=True)
        os.makedirs('data/outsiders', exist_ok=True)
        
        # Initialize models
        self.init_models()
        
        # Variables
        self.camera_active = False
        self.detection_active = False
        self.current_frame = None
        self.detection_interval = 0.5
        self.last_detection_time = 0
        self.outsider_capture_cooldown = 10
        self.last_outsider_time = 0
        self.gallery_index = 0
        self.gallery_images = []
        
        # Constants
        self.EMBEDDINGS_FILE = 'data/known_embeddings.csv'
        
        # Create GUI
        self.create_widgets()
        
    def init_models(self):
        """Initialize AI models"""
        # Status label for loading
        loading_label = tk.Label(self.root, text="Loading AI models... Please wait.", font=("Arial", 14))
        loading_label.pack(pady=20)
        self.root.update()
        
        # Load FaceNet model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        try:
            # Load YOLOv8 model
            self.yolo_model = YOLO('yolov8l-face-lindevs.pt')
            loading_label.config(text=f"Models loaded successfully on {self.device}!")
        except Exception as e:
            loading_label.config(text=f"Error loading YOLOv8: {str(e)}\nPlease install required models.")
            messagebox.showerror("Model Error", f"Error loading YOLOv8 model: {str(e)}\n"
                                "You may need to run: pip install ultralytics")
        
        # Remove loading label after 2 seconds
        self.root.after(2000, loading_label.destroy)
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel for video/image display
        self.left_panel = ttk.Frame(main_frame, relief="ridge", borderwidth=2)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Video canvas
        self.canvas = tk.Canvas(self.left_panel, bg="black")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right panel for controls
        right_panel = ttk.Frame(main_frame, relief="ridge", borderwidth=2, width=200)
        right_panel.pack(side="right", fill="y", padx=5, pady=5)
        
        # Status indicator
        self.status_frame = ttk.Frame(right_panel)
        self.status_frame.pack(fill="x", padx=5, pady=5)
        
        self.status_label = ttk.Label(self.status_frame, text="Status: Ready")
        self.status_label.pack(side="left", padx=5)
        
        self.status_indicator = tk.Canvas(self.status_frame, width=20, height=20, bg="gray")
        self.status_indicator.pack(side="right", padx=5)
        self.status_indicator.create_oval(2, 2, 18, 18, fill="gray", outline="black")
        
        # Control buttons
        ttk.Separator(right_panel).pack(fill="x", padx=5, pady=5)
        
        btn_frame = ttk.Frame(right_panel)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        self.register_btn = ttk.Button(btn_frame, text="Register Person", command=self.register_person)
        self.register_btn.pack(fill="x", pady=2)
        
        self.detection_btn = ttk.Button(btn_frame, text="Start Detection", command=self.toggle_detection)
        self.detection_btn.pack(fill="x", pady=2)
        
        self.view_outsiders_btn = ttk.Button(btn_frame, text="View Outsiders", command=self.view_outsider_gallery)
        self.view_outsiders_btn.pack(fill="x", pady=2)
        
        self.exit_btn = ttk.Button(btn_frame, text="Exit", command=self.exit_app)
        self.exit_btn.pack(fill="x", pady=2)
        
        # Gallery navigation frame (initially hidden)
        self.gallery_frame = ttk.Frame(right_panel)
        
        ttk.Label(self.gallery_frame, text="Outsider Gallery").pack(pady=5)
        
        nav_frame = ttk.Frame(self.gallery_frame)
        nav_frame.pack(fill="x", pady=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="Previous", command=self.prev_image)
        self.prev_btn.pack(side="left", padx=5)
        
        self.next_btn = ttk.Button(nav_frame, text="Next", command=self.next_image)
        self.next_btn.pack(side="right", padx=5)
        
        self.register_outsider_btn = ttk.Button(self.gallery_frame, text="Register This Outsider", 
                                              command=self.register_outsider)
        self.register_outsider_btn.pack(fill="x", pady=5)
        
        self.close_gallery_btn = ttk.Button(self.gallery_frame, text="Close Gallery", 
                                          command=self.close_gallery)
        self.close_gallery_btn.pack(fill="x", pady=5)
        
        # Info panel
        ttk.Separator(right_panel).pack(fill="x", padx=5, pady=5)
        
        info_frame = ttk.Frame(right_panel)
        info_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Detection Info:").pack(anchor="w")
        
        self.info_text = tk.Text(info_frame, height=10, width=25, wrap="word", state="disabled")
        self.info_text.pack(fill="both", expand=True, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(right_panel, orient="horizontal", 
                                      length=200, mode="determinate", 
                                      variable=self.progress_var)
        self.progress.pack(fill="x", padx=5, pady=5)
        
    def update_info(self, message):
        """Update the info text widget"""
        self.info_text.config(state="normal")
        self.info_text.insert("end", f"{time.strftime('%H:%M:%S')}: {message}\n")
        self.info_text.see("end")
        self.info_text.config(state="disabled")
        
    def update_status(self, status, color="gray"):
        """Update the status indicator"""
        self.status_label.config(text=f"Status: {status}")
        self.status_indicator.create_oval(2, 2, 18, 18, fill=color, outline="black")
        
    def register_person(self):
        """Register a new person from camera capture"""
        if self.detection_active:
            messagebox.showwarning("Warning", "Please stop detection mode first.")
            return
            
        # Start camera if not active
        if not self.camera_active:
            self.start_camera()
        
        # Get person ID
        person_id = simpledialog.askstring("Register Person", "Enter person ID (e.g., student_001):")
        if not person_id:
            return
            
        self.update_status("Registering...", "yellow")
        self.update_info(f"Registering new person: {person_id}")
        
        # Countdown
        for i in range(3, 0, -1):
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                cv2.putText(frame, f"Capturing in {i}...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.display_frame(frame)
                time.sleep(1)
        
        # Capture the image
        if self.current_frame is not None:
            frame = self.current_frame.copy()
            
            # Detect face
            face, face_rect = self.detect_face_yolo(frame)
            
            if face is None:
                messagebox.showerror("Error", "No face detected in the captured image.")
                self.update_status("Ready", "gray")
                self.update_info("Registration failed: No face detected")
                return
                
            # Save the face image
            face_filename = f"data/registered/{person_id}.jpg"
            cv2.imwrite(face_filename, face)
            
            # Extract features
            features = self.extract_features(face)
            
            # Save to CSV
            if os.path.exists(self.EMBEDDINGS_FILE):
                df = pd.read_csv(self.EMBEDDINGS_FILE)
                # Check if the person_id already exists
                if person_id in df['person_id'].values:
                    df = df[df['person_id'] != person_id]  # Remove existing entry
            else:
                df = pd.DataFrame(columns=['person_id'] + [f'feature_{i}' for i in range(len(features))])
                
            # Add new row
            new_row = [person_id] + list(features)
            df.loc[len(df)] = new_row
            
            # Save the updated DataFrame
            df.to_csv(self.EMBEDDINGS_FILE, index=False)
            
            self.update_status("Ready", "green")
            self.update_info(f"✅ Successfully registered: {person_id}")
            messagebox.showinfo("Success", f"Successfully registered: {person_id}")
            
    def start_camera(self):
        """Start the camera feed"""
        if self.camera_active:
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return
            
        self.camera_active = True
        self.update_status("Camera On", "green")
        
        # Start the video thread
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_active = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.update_status("Camera Off", "gray")
        
    def update_frame(self):
        """Update video frames"""
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                self.update_info("Error: Could not read from camera")
                break
                
            self.current_frame = frame.copy()
            
            # Process the frame if detection is active
            if self.detection_active:
                display_frame = frame.copy()
                current_time = time.time()
                
                if current_time - self.last_detection_time > self.detection_interval:
                    # Detect face
                    face, face_rect = self.detect_face_yolo(frame)
                    
                    if face is not None:
                        x, y, w, h = face_rect
                        
                        try:
                            # Extract features
                            features = self.extract_features(face)
                            
                            # Classify
                            person_id, score = self.classify_person(features)
                            
                            # Draw rectangle around face
                            if person_id != "Unknown":
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green
                                status = "INSIDER"
                                # Save the detected insider face
                                timestamp = int(time.time())
                                cv2.imwrite(f"data/detected/insider_{timestamp}.jpg", face)
                            else:
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red
                                status = "OUTSIDER"
                                
                                # Check if we should capture this outsider
                                if current_time - self.last_outsider_time > self.outsider_capture_cooldown:
                                    # Save the full frame with the outsider
                                    timestamp = int(time.time())
                                    outsider_filename = f"data/outsiders/outsider_{timestamp}.jpg"
                                    cv2.imwrite(outsider_filename, frame)
                                    
                                    # Also save just the face
                                    face_filename = f"data/outsiders/face_{timestamp}.jpg"
                                    cv2.imwrite(face_filename, face)
                                    
                                    self.update_info(f"🚨 Outsider detected!")
                                    self.last_outsider_time = current_time
                                    
                                    # Add a visual indicator
                                    cv2.putText(display_frame, "📸 PHOTO CAPTURED", (x, y+h+20), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Display info
                            info_text = f"{status}: {person_id} ({score:.2f})"
                            cv2.putText(display_frame, info_text, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                       
                            # Update the info panel
                            self.update_info(f"{status}: {person_id} (Score: {score:.2f})")
                        except Exception as e:
                            self.update_info(f"Error: {str(e)}")
                            
                        self.last_detection_time = current_time
                
                self.display_frame(display_frame)
            else:
                self.display_frame(frame)
                
            time.sleep(0.01)  # Small delay to reduce CPU usage
    
    def display_frame(self, frame):
        """Display a frame on the canvas"""
        if frame is None:
            return
            
        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has dimensions
            frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        
        # Update canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def toggle_detection(self):
        """Toggle detection mode on/off"""
        if not self.camera_active:
            self.start_camera()
            time.sleep(0.5)  # Wait for camera to start
        
        if self.detection_active:
            self.detection_active = False
            self.detection_btn.config(text="Start Detection")
            self.update_status("Camera On", "green")
            self.update_info("Detection mode stopped")
        else:
            self.detection_active = True
            self.detection_btn.config(text="Stop Detection")
            self.update_status("Detecting", "blue")
            self.update_info("Detection mode started")
    
    def view_outsider_gallery(self):
        """View captured outsider images"""
        outsider_dir = "data/outsiders"
        if not os.path.exists(outsider_dir):
            messagebox.showinfo("Info", "No outsider images found.")
            return
            
        # Get all outsider images (only full frames, not face crops)
        self.gallery_images = [f for f in os.listdir(outsider_dir) if f.startswith("outsider_")]
        self.gallery_images.sort()  # Sort by name (which includes timestamp)
        
        if len(self.gallery_images) == 0:
            messagebox.showinfo("Info", "No outsider images found.")
            return
            
        # Show gallery controls
        self.gallery_frame.pack(fill="x", padx=5, pady=5)
        
        # Disable camera temporarily if active
        temp_camera_active = self.camera_active
        temp_detection_active = self.detection_active
        if self.camera_active or self.detection_active:
            self.detection_active = False
            self.camera_active = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
        
        # Show first image
        self.gallery_index = 0
        self.show_gallery_image()
        
        # Update status
        self.update_status("Gallery View", "purple")
        self.update_info(f"Viewing outsider gallery ({len(self.gallery_images)} images)")
        
        # Save state to restore after gallery closes
        self.temp_camera_active = temp_camera_active
        self.temp_detection_active = temp_detection_active
    
    def show_gallery_image(self):
        """Show the current gallery image"""
        if not self.gallery_images or self.gallery_index >= len(self.gallery_images):
            return
            
        img_path = os.path.join("data/outsiders", self.gallery_images[self.gallery_index])
        
        # Load and display the image
        img = cv2.imread(img_path)
        if img is None:
            self.update_info(f"Error: Could not read image {img_path}")
            return
            
        # Extract timestamp from filename
        timestamp = self.gallery_images[self.gallery_index].split('_')[1].split('.')[0]
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(timestamp)))
        
        # Display the image with info
        cv2.putText(img, f"Outsider detected on: {time_str}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"Image {self.gallery_index+1}/{len(self.gallery_images)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        self.display_frame(img)
    
    def prev_image(self):
        """Show previous gallery image"""
        if self.gallery_index > 0:
            self.gallery_index -= 1
            self.show_gallery_image()
            
    def next_image(self):
        """Show next gallery image"""
        if self.gallery_index < len(self.gallery_images) - 1:
            self.gallery_index += 1
            self.show_gallery_image()
            
    def register_outsider(self):
        """Register the current outsider"""
        if not self.gallery_images or self.gallery_index >= len(self.gallery_images):
            return
            
        # Get the corresponding face image
        outsider_filename = self.gallery_images[self.gallery_index]
        face_path = os.path.join("data/outsiders", "face_" + outsider_filename.split('_')[1])
        
        if not os.path.exists(face_path):
            messagebox.showerror("Error", "Face image not found.")
            return
            
        # Get person ID
        person_id = simpledialog.askstring("Register Outsider", "Enter ID to register this outsider:")
        if not person_id:
            return
            
        # Read the face image
        face = cv2.imread(face_path)
        
        # Save the face image in registered folder
        face_filename = f"data/registered/{person_id}.jpg"
        cv2.imwrite(face_filename, face)
        
        # Extract features
        features = self.extract_features(face)
        
        # Save to CSV
        if os.path.exists(self.EMBEDDINGS_FILE):
            df = pd.read_csv(self.EMBEDDINGS_FILE)
            # Check if the person_id already exists
            if person_id in df['person_id'].values:
                df = df[df['person_id'] != person_id]  # Remove existing entry
        else:
            df = pd.DataFrame(columns=['person_id'] + [f'feature_{i}' for i in range(len(features))])
            
        # Add new row
        new_row = [person_id] + list(features)
        df.loc[len(df)] = new_row
        
        # Save the updated DataFrame
        df.to_csv(self.EMBEDDINGS_FILE, index=False)
        
        self.update_info(f"✅ Registered outsider as: {person_id}")
        messagebox.showinfo("Success", f"Successfully registered outsider as: {person_id}")
    
    def close_gallery(self):
        """Close the gallery view"""
        self.gallery_frame.pack_forget()
        
        # Clear canvas
        self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), 
                                     self.canvas.winfo_height(), fill="black")
        
        # Restore camera state if needed
        if hasattr(self, 'temp_camera_active') and self.temp_camera_active:
            self.start_camera()
            
            if hasattr(self, 'temp_detection_active') and self.temp_detection_active:
                self.detection_active = True
                self.detection_btn.config(text="Stop Detection")
                self.update_status("Detecting", "blue")
    
    def extract_features(self, img_array):
        """Extract features from image array using FaceNet"""
        # Convert to RGB (FaceNet expects RGB)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = img_array  # Already RGB
        
        # Preprocess for FaceNet (resize to 160x160)
        img = cv2.resize(rgb_img, (160, 160))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.unsqueeze(0) / 255.0  # Add batch dimension and normalize
        img = img.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.facenet_model(img)
        
        return embeddings.cpu().squeeze().numpy()

    def detect_face_yolo(self, frame, conf_threshold=0.7):
        """Detect face in the frame using YOLOv8-face with high confidence threshold"""
        # Run YOLOv8 inference on the frame
        results = self.yolo_model(frame, conf=conf_threshold)
        
        # Check if any faces were detected with high enough confidence
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Sort boxes by confidence (highest first)
            boxes = results[0].boxes
            conf = boxes.conf.cpu().numpy()
            
            if len(conf) > 0:
                # Get the index of the highest confidence detection
                best_idx = np.argmax(conf)
                
                # Extract coordinates
                x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())
                
                # Make sure coordinates are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                # Check if the face region is valid and large enough
                if x2 > x1 and y2 > y1 and (x2-x1) >= 50 and (y2-y1) >= 50:
                    # Extract the face
                    face = frame[y1:y2, x1:x2]
                    return face, (x1, y1, x2-x1, y2-y1)
        
        return None, None

    def classify_person(self, features, threshold=0.7):
        """Classify a person based on facial features"""
        if not os.path.exists(self.EMBEDDINGS_FILE):
            return "Unknown", 0.0
        
        # Load known embeddings
        df = pd.read_csv(self.EMBEDDINGS_FILE)
        
        if len(df) == 0:
            return "Unknown", 0.0
        
        # Extract person IDs and feature vectors
        person_ids = df['person_id'].values
        feature_cols = [col for col in df.columns if col != 'person_id']
        known_features = df[feature_cols].values
        
        # Calculate similarity
        similarities = cosine_similarity([features], known_features)[0]
        max_index = np.argmax(similarities)
        best_match_id = person_ids[max_index]
        similarity_score = similarities[max_index]
        
        if similarity_score > threshold:
            return best_match_id, similarity_score
        else:
            return "Unknown", similarity_score
            
    def exit_app(self):
        """Exit the application"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.camera_active = False
            self.detection_active = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_app)
    root.mainloop()