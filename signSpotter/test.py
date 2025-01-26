from inference_sdk import InferenceHTTPClient
import customtkinter as ctk
import os
import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
import playsound

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Uv06rawidNIWdABmMvGC"
)

# Define output directory for processed files
output_dir = 'output'  # Change this to any valid path

# Variable to store the last detected sign
last_detected_sign = None

# Function to process image files
def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not load the image {file_path}.")
        return

    # Use Roboflow model for inference
    result = CLIENT.infer(file_path, model_id="road-sign-detection-in-real-time/3")

    for prediction in result['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)
        confidence = prediction['confidence']
        class_name = prediction['class']

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow(f'Roboflow Detection - {os.path.basename(file_path)}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process video files
def process_video(file_path):
    video = cv2.VideoCapture(file_path)

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(output_dir, f"processed_{os.path.basename(file_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Use Roboflow model for inference
        result = CLIENT.infer(frame, model_id="road-sign-detection-in-real-time/3")

        new_sign_detected = False

        for prediction in result['predictions']:
            x1 = int(prediction['x'] - prediction['width'] / 2)
            y1 = int(prediction['y'] - prediction['height'] / 2)
            x2 = int(prediction['x'] + prediction['width'] / 2)
            y2 = int(prediction['y'] + prediction['height'] / 2)
            confidence = prediction['confidence']
            class_name = prediction['class']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            sign_image = frame[y1:y2, x1:x2]
            if sign_image is not None and (last_detected_sign is None or (last_detected_sign != sign_image).any()):
                last_detected_sign = sign_image
                new_sign_detected = True

        if new_sign_detected and last_detected_sign is not None:
            cv2.imshow("Last Detected Sign", last_detected_sign)

        out.write(frame)

        cv2.imshow(f'Processing {os.path.basename(file_path)}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved as {output_path}")

# Function to process live feed
def process_live_feed():
    cap = cv2.VideoCapture(selected_camera_index)
    if not cap.isOpened():
        print("Error: Unable to connect to the selected camera.")
        return

    global last_detected_sign
    alarm_played = False

    confidence_threshold = 0.8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use Roboflow model for inference
        result = CLIENT.infer(frame, model_id="road-sign-detection-in-real-time/3")

        new_sign_detected = False

        for prediction in result['predictions']:
            x1 = int(prediction['x'] - prediction['width'] / 2)
            y1 = int(prediction['y'] - prediction['height'] / 2)
            x2 = int(prediction['x'] + prediction['width'] / 2)
            y2 = int(prediction['y'] + prediction['height'] / 2)
            confidence = prediction['confidence']
            class_name = prediction['class']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if confidence >= confidence_threshold:
                sign_image = frame[y1:y2, x1:x2]
                if sign_image is not None:
                    if last_detected_sign is None:
                        last_detected_sign = sign_image
                        new_sign_detected = True
                    else:
                        resized_sign_image = cv2.resize(sign_image,
                                                        (last_detected_sign.shape[1], last_detected_sign.shape[0]))

                        if not (last_detected_sign == resized_sign_image).all():
                            last_detected_sign = resized_sign_image
                            new_sign_detected = True

        if new_sign_detected and last_detected_sign is not None:
            cv2.imshow("Last Detected Sign", last_detected_sign)

            if confidence >= confidence_threshold:
                playsound.playsound("alarm.mp3")
                alarm_played = True

        cv2.imshow("Live Feed - Roboflow Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Functions to trigger from buttons
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        process_image(file_path)

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        process_video(file_path)

def start_live_feed():
    process_live_feed()

# Set the appearance mode of the app
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

# Create the main application window
app = ctk.CTk()
app.geometry("1024x768")
app.title("SignSpotter - YOLO Model Detection")
app.configure(bg="white")

# Load the image after the root window is created
image_path = "resources/visigen.png"  # Replace with the path to your image
image = Image.open(image_path)
image = image.resize((400, 200))  # Resize the image to fit in the frame
image_tk = ImageTk.PhotoImage(image)

def open_settings():
    settings_window = ctk.CTkToplevel(app)
    settings_window.title("Settings")
    settings_window.geometry("400x400")

    settings_label = ctk.CTkLabel(settings_window, text="Settings", font=("Arial", 20))
    settings_label.pack(pady=20)

    # Confidence Threshold Slider
    confidence_label = ctk.CTkLabel(settings_window, text="Confidence Threshold:")
    confidence_label.pack(pady=5)
    confidence_slider = ctk.CTkSlider(settings_window, from_=0.1, to=1.0, number_of_steps=10)
    confidence_slider.set(0.8)
    confidence_slider.pack(pady=10)

    # Theme Selection
    theme_label = ctk.CTkLabel(settings_window, text="Choose Theme:")
    theme_label.pack(pady=5)
    theme_dropdown = ctk.CTkOptionMenu(settings_window, values=["Light", "Dark", "System"], command=set_theme)
    theme_dropdown.pack(pady=10)

    # Camera Selection
    camera_label = ctk.CTkLabel(settings_window, text="Select Camera:")
    camera_label.pack(pady=5)
    camera_names = [f"Camera {i}" for i in range(2)]  # Assuming up to 10 cameras
    camera_dropdown = ctk.CTkOptionMenu(settings_window, values=camera_names)
    camera_dropdown.set(f"Camera {selected_camera_index}")  # Set the default value
    camera_dropdown.pack(pady=10)

    # Save Button
    save_button = ctk.CTkButton(
        settings_window,
        text="Save",
        command=lambda: save_settings(confidence_slider.get(), camera_dropdown.get())
    )
    save_button.pack(pady=20)

def save_settings(confidence, selected_camera):
    global confidence_threshold, selected_camera_index
    confidence_threshold = confidence
    selected_camera_index = int(selected_camera.split()[-1])  # Extract the camera index
    print(f"New confidence threshold: {confidence}")
    print(f"Selected camera: {selected_camera_index}")

def set_theme(theme):
    ctk.set_appearance_mode(theme)

selected_camera_index = 0  # Default to camera 0

# Main frame for navigation
main_frame = ctk.CTkFrame(master=app, corner_radius=10, fg_color="#eeeee4")
main_frame.pack(fill="both", expand=True, pady=(20, 10), padx=10)

# Placeholder for the logo
logo_placeholder = ctk.CTkLabel(master=main_frame, image=image_tk, text="")
logo_placeholder.image = image_tk  # Keep a reference to avoid garbage collection
logo_placeholder.pack(pady=20)

# Title label
title_label = ctk.CTkLabel(master=main_frame, text="Welcome to SignSpotter", font=("Arial", 24), bg_color="#eeeee4")
title_label.pack(pady=20)

# Buttons for functionalities
button_live_feed = ctk.CTkButton(
    master=main_frame,
    text="Start Live Feed",
    command=start_live_feed,
    fg_color="#ba3b0a",
    text_color="white",
    width=250,  # Main button width
    height=60   # Main button height
)
button_live_feed.pack(pady=20)

button_image_upload = ctk.CTkButton(
    master=main_frame,
    text="Upload Image",
    command=upload_image,
    fg_color="#ba3b0a",
    text_color="white",
    width=250,
    height=60
)
button_image_upload.pack(pady=10)

button_video_upload = ctk.CTkButton(
    master=main_frame,
    text="Upload Video",
    command=upload_video,
    fg_color="#ba3b0a",
    text_color="white",
    width=250,
    height=60
)
button_video_upload.pack(pady=10)

# Navigation buttons at the bottom
bottom_frame = ctk.CTkFrame(master=app, fg_color="white")  # Define bottom_frame
bottom_frame.pack(fill="x", side="bottom")  # Position it at the bottom of the main window

settings_button = ctk.CTkButton(
    master=bottom_frame,
    text="Settings",
    fg_color="#ba3b0a",
    text_color="white",
    command=open_settings,
    width=140,  # Smaller navigation button width
    height=45   # Smaller navigation button height
)
settings_button.pack(side="left", expand=True, pady=10, padx=10)

info_button = ctk.CTkButton(
    master=bottom_frame,
    text="Information",
    fg_color="#ba3b0a",
    text_color="white",
    command=lambda: print("information clicked"),
    width=140,
    height=45
)
info_button.pack(side="left", expand=True, pady=10, padx=10)

exit_button = ctk.CTkButton(
    master=bottom_frame,
    text="Exit",
    fg_color="#ba3b0a",
    text_color="white",
    command=app.quit,
    width=140,
    height=45
)
exit_button.pack(side="left", expand=True, pady=10, padx=10)

# Run the application
app.mainloop()