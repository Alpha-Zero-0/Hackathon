import sqlite3
import time
import tkinter as tk
from tkinter import messagebox
import random  # Just for simulation

# Simulated posture detection: randomly returns True (slouch) or False (good posture)
def get_pose_status():
    return random.choice([True, False])

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
                status TEXT
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
        
    def start_test(self):
        # Get username from entry; if empty, do nothing
        entered_username = self.username_entry.get().strip()
        if not entered_username:
            return
        
        # Check if the username already exists in the database
        self.cursor.execute("SELECT COUNT(*) FROM posture_log WHERE username = ?", (entered_username,))
        count = self.cursor.fetchone()[0]
        if count > 0:
            messagebox.showerror("Username Exists", "This username already exists. Please choose a different username.")
            return
        
        self.username = entered_username
        # Destroy login window and start the test UI
        self.root.destroy()
        self.create_test_ui()
    
    def create_test_ui(self):
        # Main test window
        self.test_window = tk.Tk()
        self.test_window.title(f"Posture Test - User: {self.username}")
        
        self.status_label = tk.Label(self.test_window, text="Status: Unknown", font=("Arial", 40))
        self.status_label.pack(pady=20)
        
        self.log_text = tk.Text(self.test_window, height=100, width=180)
        self.log_text.pack(pady=10)
        
        # Button to manually generate report
        report_button = tk.Button(self.test_window, text="Generate Report", command=self.generate_report)
        report_button.pack(pady=5)
        
        # Remove the End Test button so the test continues indefinitely
        
        self.last_status = None
        # Update interval in milliseconds (e.g., 2000ms = 2 seconds)
        self.update_interval = 2000
        
        # Start periodic test updates
        self.test_window.after(self.update_interval, self.update_test)
        self.test_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.test_window.mainloop()
    
    def update_test(self):
        slouch_detected = get_pose_status()
        current_status = "Slouch Detected" if slouch_detected else "Good Posture"
        
        # Only update if the status has changed
        if current_status != self.last_status:
            color = "red" if slouch_detected else "green"
            self.status_label.config(text=f"Status: {current_status}", fg=color)
            self.log_message(f"Status changed to: {current_status}")
            
            # Record this change with a timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.insert_into_db(timestamp, current_status)
            self.last_status = current_status
        
        # Schedule the next update
        self.test_window.after(self.update_interval, self.update_test)
    
    def insert_into_db(self, timestamp, status):
        try:
            self.cursor.execute(
                "INSERT INTO posture_log (username, timestamp, status) VALUES (?, ?, ?)",
                (self.username, timestamp, status)
            )
            self.conn.commit()
            self.log_message("Record inserted into database.")
        except Exception as e:
            self.log_message(f"DB insert error: {e}")
    
    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
        self.log_text.see(tk.END)
    
    def generate_report(self):
        """
        Generate a report that calculates the average "good posture" ratio for each user,
        then ranks users and shows how much the current user 'beats' themselves.
        Ranking formula: percentile = ((total_users - rank + 1) / total_users) * 100.
        """
        try:
            self.cursor.execute('''
                SELECT username, AVG(CASE WHEN status = 'Good Posture' THEN 1.0 ELSE 0 END) AS ratio
                FROM posture_log
                GROUP BY username
            ''')
            user_scores = self.cursor.fetchall()
        except Exception as e:
            self.log_message(f"Error generating report: {e}")
            return
        
        if not user_scores:
            self.log_message("No data available for report.")
            return
        
        # Sort users by ratio in descending order (best performer first)
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine rank for current user (1-indexed)
        rank = None
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
        
        # Display the report in a new window
        report_window = tk.Toplevel(self.test_window)
        report_window.title("Posture Report")
        report_message = (f"User: {self.username}\n"
                          f"Good Posture Ratio: {current_ratio*100:.2f}%\n"
                          f"Your ranking percentile: {percentage_beat:.2f}%\n\n"
                          f"(Best performer always gets 100%, and rankings are relative among all users.)")
        tk.Label(report_window, text=report_message, font=("Arial", 30), justify=tk.LEFT).pack(padx=10, pady=10)
    
    def on_closing(self):
        # Close the test window, commit and close the database, and exit.
        self.conn.close()
        self.test_window.destroy()

if __name__ == '__main__':
    app = PostureTestApp()
    app.root.mainloop()
