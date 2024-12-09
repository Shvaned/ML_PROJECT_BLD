import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests

# Function to upload video
def upload_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if filepath:
        try:
            with open(filepath, 'rb') as f:
                response = requests.post('http://127.0.0.1:5000/upload', files={'file': f})
                if response.ok:
                    result = response.json()

                    if 'insights' in result:
                        insights = "\n".join(
                            f"{label}: {prob}" for label, prob in result['insights']['predictions'].items())
                        messagebox.showinfo("Video Insights", f"Summary: {result['insights']['summary']}\n\n{insights}")

                        img_path = result['insights']['conclusion_image']
                        img = Image.open(img_path)
                        img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
                        img_tk = ImageTk.PhotoImage(img)

                        img_window = tk.Toplevel(root)
                        img_window.title("Conclusion Image")
                        img_window.geometry("600x600")
                        img_label = tk.Label(img_window, image=img_tk, bg="#333333")
                        img_label.image = img_tk
                        img_label.pack(expand=True)
                    else:
                        messagebox.showerror("Error", "No insights found in the response.")
                else:
                    messagebox.showerror("Error", response.json().get('error', 'Failed to process the video.'))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Video Insight Extractor")
root.geometry("500x700")
root.config(bg="#000000")

# Title Label
title_label = tk.Label(root, text="Body Language Analyser", bg="#000000", fg="#FFFFFF",
                       font=("Helvetica", 18, "bold"), pady=20)
title_label.pack()

# Subtitle
subtitle_label = tk.Label(
    root,
    text="Upload your video to extract insights.",
    bg="#000000",
    fg="#888888",
    font=("Helvetica", 12),
    pady=10
)
subtitle_label.pack()

# Upload Button
upload_btn = tk.Button(
    root,
    text="Upload Video",
    command=upload_file,
    bg="#4CAF50",
    fg="#FFFFFF",
    font=("Helvetica", 14, "bold"),
    activebackground="#45A049",
    activeforeground="#FFFFFF",
    relief="flat",
    padx=20,
    pady=10
)
upload_btn.pack(pady=20)

# Placeholder for the result or image
result_frame = tk.Frame(root, bg="#111111", padx=10, pady=10)
result_frame.pack(expand=True, fill="both")



# Run the Tkinter main loop
root.mainloop()
