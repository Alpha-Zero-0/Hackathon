import sqlite3
import time
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import sys
import threading
import serial
import serial.tools.list_ports

# Mediapipe Pose/Holistic setup
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Global variables for camera access
camera_active = False
global_frame = None
global_frame_with_landmarks = None  # Frame with landmarks drawn
webcam_thread = None
camera_lock = threading.Lock()

# ----------------------------------------
# HELPER FUNCTIONS FOR POSTURE DETECTION
# ----------------------------------------

def distance(v1, v2):
    return np.sqrt(((v1 - v2)**2).sum())

def find_angles(vec1, vec2, vec3):
    """
    Returns the angle (in degrees) between the vectors: 
    vec21 = (vec1 - vec2) and vec23 = (vec3 - vec2).
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    vec3 = np.array(vec3)

    vec21 = vec1 - vec2
    vec23 = vec3 - vec2

    dist_vec21 = distance(vec1, vec2)
    dist_vec23 = distance(vec3, vec2)

    if dist_vec21 == 0 or dist_vec23 == 0:
        return 0

    angle = np.degrees(np.arccos(np.dot(vec21, vec23) / (dist_vec21 * dist_vec23)))
    return angle

def camera_thread_function():
    """
    Continuously captures frames from the webcam and updates the global frames.
    """
    global global_frame, global_frame_with_landmarks, camera_active
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        camera_active = False
        return
        
    # Set camera resolution if desired
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while camera_active:
            ret, frame = cap.read()
            if ret:
                with camera_lock:
                    global_frame = frame.copy()
                
                # Process the frame for landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Draw landmarks on a copy of the frame
                annotated_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                with camera_lock:
                    global_frame_with_landmarks = annotated_frame
            else:
                print("Warning: Could not read frame from webcam.")
                time.sleep(0.1)
            
    cap.release()
    print("Camera released")

def start_camera():
    """Start the camera thread."""
    global camera_active, webcam_thread
    
    if not camera_active:
        camera_active = True
        webcam_thread = threading.Thread(target=camera_thread_function)
        webcam_thread.daemon = True
        webcam_thread.start()
        print("Camera thread started")
        time.sleep(1)

def stop_camera():
    """Stop the camera thread."""
    global camera_active, webcam_thread
    
    if camera_active:
        camera_active = False
        if webcam_thread is not None:
            webcam_thread.join(timeout=2.0)
            print("Camera thread stopped")

def analyze_posture(frame):
    """
    Analyzes a frame using Mediapipe to detect posture.
    Returns True if 'slouch' is detected, False if 'good posture' is detected.
    """
    if frame is None:
        return False
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        results = holistic.process(rgb_frame)

    if not results or not results.pose_landmarks:
        return False

    try:
        landmarks = results.pose_landmarks.landmark
        leftKnee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
        leftHip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
        leftShoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
        leftEar = [landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].y]

        if any(l is None for l in (leftKnee + leftHip + leftShoulder + leftEar)):
            return False
    except:
        return False

    virtualPoint = [leftEar[0], leftEar[1] - 0.1]
    angleKHS = find_angles(leftKnee, leftHip, leftShoulder)
    angleHSE = find_angles(leftHip, leftShoulder, leftEar)
    angleSEV = find_angles(leftShoulder, leftEar, virtualPoint)

    validKHS = (75 <= angleKHS <= 105)
    validHSE = (angleHSE >= 165)
    validSEV = (angleSEV >= 165)

    if validKHS and validHSE and validSEV:
        return False  # Good posture
    else:
        return True   # Slouch detected

def get_pose_status():
    """
    Gets the current frame from the global variable and analyzes posture.
    Returns True if 'slouch' is detected, False otherwise.
    """
    global global_frame
    with camera_lock:
        if global_frame is not None:
            current_frame = global_frame.copy()
        else:
            print("Warning: No frame available for posture detection.")
            return False
            
    return analyze_posture(current_frame)


# ----------------------------------------
# TKINTER + SQLITE + ARDUINO INTEGRATION APP
# ----------------------------------------

class PostureTestApp:
    def __init__(self, db_filename="posture_data.db"):
        # Connect to (or create) the SQLite database and create the table
        self.conn = sqlite3.connect(db_filename)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS posture_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                timestamp TEXT,
                status TEXT,
                duration REAL DEFAULT 0.0
            )
        ''')
        self.conn.commit()
        
        # Initialize Arduino serial connection
        self.init_serial()
        
        # Tkinter login window
        self.root = tk.Tk()
        self.root.title("Posture Test - Login")
        tk.Label(self.root, text="Enter your username:").pack(pady=10)
        self.username_entry = tk.Entry(self.root, width=30)
        self.username_entry.pack(pady=5)
        tk.Button(self.root, text="Start Test", command=self.start_test).pack(pady=10)
        
        self.username = None
        
        # Start the camera
        start_camera()

    def init_serial(self):
        """
        Scan available serial ports and prompt the user to select the Arduino port.
        """
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial ports found. Arduino integration disabled.")
            self.serial_inst = None
            return
        
        print("Available serial ports:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device}")
        try:
            selection = input("Select the index for the Arduino port: ")
            index = int(selection)
            chosen_port = ports[index].device
            self.serial_inst = serial.Serial(chosen_port, 9600, timeout=1)
            print(f"Using Arduino port: {chosen_port}")
        except Exception as e:
            print("Error selecting serial port:", e)
            self.serial_inst = None

    def send_serial_command(self, status):
        """
        Sends a serial command to the Arduino:
          - "BP" if bad posture is detected.
          - "GP" if good posture is detected.
        """
        if self.serial_inst:
            command = "BP\n" if status == "Slouch Detected" else "GP\n"
            try:
                self.serial_inst.write(command.encode('utf-8'))
                print(f"Sent command: {command.strip()}")
            except Exception as e:
                self.log_message("Serial command error: " + str(e))

    def start_test(self):
        entered_username = self.username_entry.get().strip()
        if not entered_username:
            messagebox.showwarning("Missing Username", "Please enter a username.")
            return
        
        # (Optional) enforce unique username by checking the database
        self.cursor.execute("SELECT COUNT(*) FROM posture_log WHERE username = ?", (entered_username,))
        count = self.cursor.fetchone()[0]
        # Uncomment to enforce uniqueness:
        # if count > 0:
        #     messagebox.showerror("Username Exists", "This username already exists. Please choose a different username.")
        #     return
        
        self.username = entered_username
        self.root.destroy()
        self.create_test_ui()
    
    def create_test_ui(self):
        self.test_window = tk.Tk()
        self.test_window.title(f"Posture Test - User: {self.username}")
        self.test_window.minsize(800, 600)
        
        main_frame = tk.Frame(self.test_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.status_label = tk.Label(main_frame, text="Status: Initializing...", font=("Arial", 40))
        self.status_label.pack(pady=20)
        
        time_frame = tk.Frame(main_frame)
        time_frame.pack(fill=tk.X, pady=10)
        self.good_time_label = tk.Label(time_frame, text="Good Posture Time: 0s", font=("Arial", 14))
        self.good_time_label.pack(side=tk.LEFT, padx=10)
        self.slouch_time_label = tk.Label(time_frame, text="Slouch Time: 0s", font=("Arial", 14))
        self.slouch_time_label.pack(side=tk.LEFT, padx=10)
        self.ratio_label = tk.Label(time_frame, text="Good Posture Ratio: 0%", font=("Arial", 14))
        self.ratio_label.pack(side=tk.LEFT, padx=10)
        
        log_frame = tk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        tk.Label(log_frame, text="Activity Log:").pack(anchor=tk.W)
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(log_frame, height=15, width=80, yscrollcommand=log_scroll.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        report_button = tk.Button(button_frame, text="Generate Report", command=self.generate_report)
        report_button.pack(side=tk.LEFT, padx=5)
        
        self.preview_active = False
        self.preview_button = tk.Button(button_frame, text="Show Camera Preview", command=self.toggle_preview)
        self.preview_button.pack(side=tk.LEFT, padx=5)
        
        self.preview_canvas = tk.Canvas(main_frame, width=320, height=240, bg="black")
        
        self.last_status = None
        self.last_status_change_time = time.time()
        self.good_posture_time = 0.0
        self.slouch_time = 0.0
        self.start_time = time.time()
        self.update_interval = 2000  # milliseconds
        
        self.log_message("Posture monitoring started")
        self.test_window.after(100, self.update_test)
        self.test_window.after(100, self.update_preview)
        self.test_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.test_window.mainloop()
    
    def toggle_preview(self):
        if self.preview_active:
            self.preview_canvas.pack_forget()
            self.preview_button.config(text="Show Camera Preview")
            self.preview_active = False
        else:
            self.preview_canvas.pack(pady=10)
            self.preview_button.config(text="Hide Camera Preview")
            self.preview_active = True
    
    def update_preview(self):
        if hasattr(self, 'preview_active') and self.preview_active and camera_active:
            with camera_lock:
                if global_frame_with_landmarks is not None:
                    preview_frame = cv2.resize(global_frame_with_landmarks, (320, 240))
                    preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                    self.photo = tk.PhotoImage(data=cv2.imencode('.png', preview_rgb)[1].tobytes())
                    self.preview_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        if hasattr(self, 'test_window'):
            self.test_window.after(100, self.update_preview)
    
    def update_test(self):
        slouch_detected = get_pose_status()
        current_status = "Slouch Detected" if slouch_detected else "Good Posture"
        now = time.time()
        duration = now - self.last_status_change_time
        
        if self.last_status is not None:
            if self.last_status == "Good Posture":
                self.good_posture_time += duration
            else:
                self.slouch_time += duration
            self.update_time_labels()
        
        if current_status != self.last_status:
            color = "red" if slouch_detected else "green"
            self.status_label.config(text=f"Status: {current_status}", fg=color)
            self.log_message(f"Status changed to: {current_status}")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            if self.last_status is not None:
                self.insert_into_db(timestamp, self.last_status, duration)
            self.last_status = current_status
            self.last_status_change_time = now
            
            # Send serial command to Arduino
            self.send_serial_command(current_status)
        
        self.test_window.after(self.update_interval, self.update_test)
    
    def update_time_labels(self):
        self.good_time_label.config(text=f"Good Posture Time: {self.good_posture_time:.1f}s")
        self.slouch_time_label.config(text=f"Slouch Time: {self.slouch_time:.1f}s")
        total_time = self.good_posture_time + self.slouch_time
        if total_time > 0:
            ratio = (self.good_posture_time / total_time) * 100
            self.ratio_label.config(text=f"Good Posture Ratio: {ratio:.1f}%")
    
    def insert_into_db(self, timestamp, status, duration):
        try:
            self.cursor.execute(
                "INSERT INTO posture_log (username, timestamp, status, duration) VALUES (?, ?, ?, ?)",
                (self.username, timestamp, status, duration)
            )
            self.conn.commit()
            self.log_message(f"Record inserted: {status} for {duration:.1f}s")
        except Exception as e:
            self.log_message(f"DB insert error: {e}")
    
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
        self.log_text.see(tk.END)
    
    def generate_report(self):
        try:
            self.cursor.execute('''
                SELECT username,
                       SUM(CASE WHEN status = 'Good Posture' THEN duration ELSE 0 END) AS good_time,
                       SUM(duration) AS total_time
                FROM posture_log
                GROUP BY username
            ''')
            user_data = self.cursor.fetchall()
        except Exception as e:
            self.log_message(f"Error generating report: {e}")
            return
        
        if not user_data:
            self.log_message("No data available for report.")
            return
        
        user_scores = []
        for user, good_time, total_time in user_data:
            ratio = (good_time / total_time * 100) if total_time > 0 else 0
            user_scores.append((user, ratio))
        
        user_scores.sort(key=lambda x: x[1], reverse=True)
        rank = None
        current_ratio = 0
        for idx, (user, ratio) in enumerate(user_scores, start=1):
            if user == self.username:
                rank = idx
                current_ratio = ratio
                break
        if rank is None:
            self.log_message("No data for current user.")
            return
        
        total_users = len(user_scores)
        percentage_beat = ((total_users - rank + 1) / total_users) * 100
        total_session_time = self.good_posture_time + self.slouch_time
        session_ratio = (self.good_posture_time / total_session_time * 100) if total_session_time > 0 else 0
        
        report_window = tk.Toplevel(self.test_window)
        report_window.title("Posture Report")
        report_message = (
            f"User: {self.username}\n\n"
            f"Current Session Stats:\n"
            f"  Good Posture Time: {self.good_posture_time:.1f}s\n"
            f"  Slouch Time: {self.slouch_time:.1f}s\n"
            f"  Session Ratio: {session_ratio:.2f}%\n\n"
            f"Overall Stats:\n"
            f"  Overall Good Posture Ratio: {current_ratio:.2f}%\n"
            f"  Your ranking percentile: {percentage_beat:.2f}%\n\n"
            f"(Best performer always gets 100%, and rankings are relative among all users.)"
        )
        tk.Label(report_window, text=report_message, font=("Arial", 14), justify=tk.LEFT).pack(padx=20, pady=20)
    
    def on_closing(self):
        now = time.time()
        duration = now - self.last_status_change_time
        if self.last_status is not None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.insert_into_db(timestamp, self.last_status, duration)
        
        stop_camera()
        if hasattr(self, 'serial_inst') and self.serial_inst is not None and self.serial_inst.is_open:
            self.serial_inst.close()
        self.conn.close()
        self.test_window.destroy()
        sys.exit(0)

if __name__ == '__main__':
    try:
        app = PostureTestApp()
        app.root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
    finally:
        stop_camera()
