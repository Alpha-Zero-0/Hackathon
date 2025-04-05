import sqlite3
import time
import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import sys
import threading

# Mediapipe Pose/Holistic setup
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Add drawing utilities

# Global variables for camera access
camera_active = False
global_frame = None
global_frame_with_landmarks = None  # New variable for frame with landmarks
webcam_thread = None
camera_lock = threading.Lock()

# ----------------------------------------
# HELPER FUNCTIONS FOR POSTURE DETECTION
# ----------------------------------------

def distance(v1, v2):
    return np.sqrt(((v1 - v2)**2).sum())

def find_angles(vec1, vec2, vec3):
    """
    Returns the angle (in degrees) between the vectors vec21=(vec1-vec2) and vec23=(vec3-vec2).
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    vec3 = np.array(vec3)

    vec21 = vec1 - vec2
    vec23 = vec3 - vec2

    dist_vec21 = distance(vec1, vec2)
    dist_vec23 = distance(vec3, vec2)

    # Avoid division by zero
    if dist_vec21 == 0 or dist_vec23 == 0:
        return 0

    angle = np.degrees(
        np.arccos(
            np.dot(vec21, vec23) / (dist_vec21 * dist_vec23)
        )
    )
    return angle

def camera_thread_function():
    """
    Function that continuously captures frames from the webcam
    and updates the global_frame variable.
    """
    global global_frame, global_frame_with_landmarks, camera_active
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        camera_active = False
        return
        
    # Set camera properties if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize holistic detector outside the loop for better performance
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                             min_tracking_confidence=0.5) as holistic:
        while camera_active:
            ret, frame = cap.read()
            if ret:
                # Save the original frame
                with camera_lock:
                    global_frame = frame.copy()
                
                # Process the frame with MediaPipe Holistic
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Draw landmarks on the frame
                annotated_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame, 
                        results.pose_landmarks, 
                        mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                
                # Save the annotated frame
                with camera_lock:
                    global_frame_with_landmarks = annotated_frame
            else:
                print("Warning: Could not read frame from webcam.")
                time.sleep(0.1)  # Small delay to prevent CPU overload
            
    # Release the webcam when the thread is stopping
    cap.release()
    print("Camera released")

def start_camera():
    """Start the camera in a separate thread."""
    global camera_active, webcam_thread
    
    if not camera_active:
        camera_active = True
        webcam_thread = threading.Thread(target=camera_thread_function)
        webcam_thread.daemon = True  # Thread will close when main program exits
        webcam_thread.start()
        print("Camera thread started")
        
        # Give the camera a moment to initialize
        time.sleep(1)

def stop_camera():
    """Stop the camera thread."""
    global camera_active, webcam_thread
    
    if camera_active:
        camera_active = False
        if webcam_thread is not None:
            webcam_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to end
            print("Camera thread stopped")

def analyze_posture(frame):
    """
    Analyzes a frame using Mediapipe to detect posture.
    Returns True if 'slouch' is detected, False if 'good posture' is detected.
    """
    if frame is None:
        return False
        
    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use holistic to get landmarks
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        results = holistic.process(rgb_frame)

    if not results or not results.pose_landmarks:
        # No landmarks => assume good posture
        return False

    # Extract relevant landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        leftKnee = [
            landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y
        ]
        leftHip = [
            landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y
        ]
        leftShoulder = [
            landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        leftEar = [
            landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].x,
            landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].y
        ]

        # If any landmark was missing (None), skip
        if any(l is None for l in (leftKnee + leftHip + leftShoulder + leftEar)):
            return False
    except:
        # If indexing fails for some reason, treat as no detection
        return False

    # We'll define a "virtual point" slightly above the ear
    virtualPoint = [leftEar[0], leftEar[1] - 0.1]

    # Calculate angles
    angleKHS = find_angles(leftKnee, leftHip, leftShoulder)
    angleHSE = find_angles(leftHip, leftShoulder, leftEar)
    angleSEV = find_angles(leftShoulder, leftEar, virtualPoint)

    # Check if angles are within "good posture" thresholds
    validKHS = (75 <= angleKHS <= 105)
    validHSE = (angleHSE >= 165)
    validSEV = (angleSEV >= 165)

    # If all angles are valid => Good posture => return False (no slouch)
    # Otherwise => Slouch => return True
    if validKHS and validHSE and validSEV:
        return False  # Good posture
    else:
        return True   # Slouch detected

def get_pose_status():
    """
    Gets the current frame from the global variable and analyzes posture.
    Returns True if 'slouch' is detected, False if 'good posture' is detected.
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
# TKINTER + SQLITE APP
# ----------------------------------------

class PostureTestApp:
    def __init__(self, db_filename="posture_data.db"):
        # Connect to (or create) the SQLite database and set up the table
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
        
        # Tkinter main window for user login
        self.root = tk.Tk()
        self.root.title("Posture Test - Login")
        
        # Username entry
        tk.Label(self.root, text="Enter your username:").pack(pady=10)
        self.username_entry = tk.Entry(self.root, width=30)
        self.username_entry.pack(pady=5)
        tk.Button(self.root, text="Start Test", command=self.start_test).pack(pady=10)
        
        self.username = None  # Will be set on login
        
        # Initialize camera when app starts
        start_camera()

    def start_test(self):
        # Get username from entry; if empty, do nothing
        entered_username = self.username_entry.get().strip()
        if not entered_username:
            messagebox.showwarning("Missing Username", "Please enter a username.")
            return
        
        # Check if the username already exists in the database
        self.cursor.execute("SELECT COUNT(*) FROM posture_log WHERE username = ?", (entered_username,))
        count = self.cursor.fetchone()[0]
        
        # Uncomment the next lines if you want to enforce unique usernames
        # if count > 0:
        #     messagebox.showerror("Username Exists", "This username already exists. Please choose a different username.")
        #     return
        
        self.username = entered_username
        # Destroy login window and start the test UI
        self.root.destroy()
        self.create_test_ui()
    
    def create_test_ui(self):
        # Main test window
        self.test_window = tk.Tk()
        self.test_window.title(f"Posture Test - User: {self.username}")
        
        # Set a minimum size for the window
        self.test_window.minsize(800, 600)
        
        # Create main frame for content
        main_frame = tk.Frame(self.test_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Status label (shows current posture status)
        self.status_label = tk.Label(main_frame, text="Status: Initializing...", font=("Arial", 40))
        self.status_label.pack(pady=20)
        
        # Time tracking labels
        time_frame = tk.Frame(main_frame)
        time_frame.pack(fill=tk.X, pady=10)
        
        self.good_time_label = tk.Label(time_frame, text="Good Posture Time: 0s", font=("Arial", 14))
        self.good_time_label.pack(side=tk.LEFT, padx=10)
        
        self.slouch_time_label = tk.Label(time_frame, text="Slouch Time: 0s", font=("Arial", 14))
        self.slouch_time_label.pack(side=tk.LEFT, padx=10)
        
        self.ratio_label = tk.Label(time_frame, text="Good Posture Ratio: 0%", font=("Arial", 14))
        self.ratio_label.pack(side=tk.LEFT, padx=10)
        
        # Log display
        log_frame = tk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        log_label = tk.Label(log_frame, text="Activity Log:")
        log_label.pack(anchor=tk.W)
        
        # Add a scrollbar for the log
        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=15, width=80, yscrollcommand=log_scroll.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Button to manually generate report
        report_button = tk.Button(button_frame, text="Generate Report", command=self.generate_report)
        report_button.pack(side=tk.LEFT, padx=5)
        
        # Button to toggle camera preview
        self.preview_active = False
        self.preview_button = tk.Button(button_frame, text="Show Camera Preview", command=self.toggle_preview)
        self.preview_button.pack(side=tk.LEFT, padx=5)
        
        # Canvas for camera preview (initially hidden)
        self.preview_canvas = tk.Canvas(main_frame, width=320, height=240, bg="black")
        # The canvas is not packed initially - we'll pack it when the user toggles preview
        
        # Initialize internal state variables
        self.last_status = None
        self.last_status_change_time = time.time()
        self.good_posture_time = 0.0
        self.slouch_time = 0.0
        self.start_time = time.time()
        
        # Update interval in milliseconds (e.g., 2000ms = 2 seconds)
        self.update_interval = 2000
        
        # Log startup message
        self.log_message("Posture monitoring started")
        
        # Start periodic test updates
        self.test_window.after(100, self.update_test)  # Start with a short delay
        self.test_window.after(100, self.update_preview)  # Start preview updater (even if not visible yet)
        self.test_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.test_window.mainloop()
    
    def toggle_preview(self):
        """Toggle the camera preview on/off"""
        if self.preview_active:
            # Hide preview
            self.preview_canvas.pack_forget()
            self.preview_button.config(text="Show Camera Preview")
            self.preview_active = False
        else:
            # Show preview
            self.preview_canvas.pack(pady=10)
            self.preview_button.config(text="Hide Camera Preview")
            self.preview_active = True
    
    def update_preview(self):
        """Update the camera preview if active"""
        if hasattr(self, 'preview_active') and self.preview_active and camera_active:
            with camera_lock:
                # Use the frame with landmarks instead of the raw frame
                if global_frame_with_landmarks is not None:
                    # Resize the frame to fit our canvas
                    preview_frame = cv2.resize(global_frame_with_landmarks, (320, 240))
                    # Convert to RGB for tkinter
                    preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                    # Convert to PhotoImage
                    self.photo = tk.PhotoImage(data=cv2.imencode('.png', preview_rgb)[1].tobytes())
                    # Update canvas
                    self.preview_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Schedule next update
        if hasattr(self, 'test_window'):
            self.test_window.after(100, self.update_preview)
    
    def update_test(self):
        # Get posture status
        slouch_detected = get_pose_status()
        current_status = "Slouch Detected" if slouch_detected else "Good Posture"
        
        # Get current time
        now = time.time()
        
        # Calculate duration since last status change
        duration = now - self.last_status_change_time
        
        # Update timers based on current status
        if self.last_status is not None:
            if self.last_status == "Good Posture":
                self.good_posture_time += duration
            else:
                self.slouch_time += duration
            
            # Update time labels
            self.update_time_labels()
        
        # Only update if the status has changed
        if current_status != self.last_status:
            color = "red" if slouch_detected else "green"
            self.status_label.config(text=f"Status: {current_status}", fg=color)
            self.log_message(f"Status changed to: {current_status}")
            
            # Record this change with a timestamp and duration
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Only log if this isn't the first status change
            if self.last_status is not None:
                self.insert_into_db(timestamp, self.last_status, duration)
            
            self.last_status = current_status
            self.last_status_change_time = now
        
        # Schedule the next update
        self.test_window.after(self.update_interval, self.update_test)
    
    def update_time_labels(self):
        # Update the time labels with current durations
        self.good_time_label.config(text=f"Good Posture Time: {self.good_posture_time:.1f}s")
        self.slouch_time_label.config(text=f"Slouch Time: {self.slouch_time:.1f}s")
        
        # Calculate and display ratio
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
        """
        Generate a report that calculates the average "good posture" ratio for each user,
        based on time spent in each posture state.
        """
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
        
        # Calculate ratios and sort users by ratio in descending order
        user_scores = []
        for user, good_time, total_time in user_data:
            ratio = (good_time / total_time * 100) if total_time > 0 else 0
            user_scores.append((user, ratio))
        
        # Sort by ratio (best performer first)
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine rank for current user (1-indexed)
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
        # Ranking formula: percentile = ((total_users - rank + 1) / total_users) * 100
        percentage_beat = ((total_users - rank + 1) / total_users) * 100
        
        # Add current session data
        total_session_time = self.good_posture_time + self.slouch_time
        session_ratio = (self.good_posture_time / total_session_time * 100) if total_session_time > 0 else 0
        
        # Display the report in a new window
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
        # Before closing, log the current status with its duration
        now = time.time()
        duration = now - self.last_status_change_time
        
        if self.last_status is not None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.insert_into_db(timestamp, self.last_status, duration)
        
        # Stop the camera thread
        stop_camera()
        
        # Close the test window, commit and close the database, and exit.
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
        # Make sure camera is stopped if application exits
        stop_camera()