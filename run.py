import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from PIL import Image, ImageTk

# Function to upload video
def upload_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    if filepath:
        try:
            with open(filepath, 'rb') as f:
                # Send POST request to Flask server
                response = requests.post('http://127.0.0.1:5000/upload', files={'file': f})

                if response.ok:
                    result = response.json()

                    # Check if there is valid 'insights' returned
                    if 'insights' in result:
                        # Display insights in a messagebox
                        insights = "\n".join(
                            f"{label}: {prob}%" for label, prob in result['insights']['predictions'].items())
                        messagebox.showinfo("Video Insights", f"Summary: {result['insights']['summary']}\n\n{insights}")

                        # Display the saved image in a new Tkinter window
                        img_path = result['insights']['conclusion_image']

                        # Open the image
                        img = Image.open(img_path)

                        # Resize image to fit within the window while maintaining aspect ratio
                        max_width = 1920
                        max_height = 1080
                        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

                        # Convert the image to Tkinter-compatible format
                        img_tk = ImageTk.PhotoImage(img)

                        # Create a new top-level window for displaying the image
                        img_window = tk.Toplevel(root)
                        img_window.title("Conclusion Image")
                        img_window.geometry("400x400")  # Set size for the new window

                        # Create a label to display the image in the new window
                        img_label = tk.Label(img_window, image=img_tk)
                        img_label.image = img_tk  # Keep a reference to avoid garbage collection
                        img_label.pack(pady=10)  # Add some padding for better layout

                    else:
                        messagebox.showerror("Error", "No insights found in the response.")

                else:
                    messagebox.showerror("Error", response.json().get('error', 'Failed to process the video.'))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Video Insight Extractor")
root.geometry("400x700")  # Adjusted to accommodate both the button and image preview

# Upload Button
upload_btn = tk.Button(root, text="Upload Video", command=upload_file, bg="blue", fg="white", padx=20, pady=10)
upload_btn.pack(expand=True)

# Run the Tkinter main loop
root.mainloop()
