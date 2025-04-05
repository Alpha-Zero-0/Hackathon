import tkinter as tk

def show_alert():
    label.config(text="⚠️ Slouch Detected! ⚠️", fg="red")

root = tk.Tk()
root.title("Posture Police")

label = tk.Label(root, text="Good posture", font=("Arial", 24), fg="green")
label.pack(padx=20, pady=20)

# For testing button to simulate posture alert
test_button = tk.Button(root, text="Test Slouch", command=show_alert)
test_button.pack(pady=10)

root.mainloop()
