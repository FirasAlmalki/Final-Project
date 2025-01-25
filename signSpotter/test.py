import customtkinter as ctk
import os
from ultralytics import YOLO
import cv2
from tkinter import filedialog
import playsound

# Initialize the YOLO model with the trained weights
model = YOLO('C:/Users/Admin/Desktop/Project/SignSpotter/runs/detect/train4/weights/best.pt')

# Define output directory for processed files
output_dir = 'C:/Users/Admin/Desktop/result'  # You can change this to any valid path

# Variable to store the last detected sign
last_detected_sign = None

# Function to process image files
def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not load the image {file_path}.")
        return

    results = model.predict(source=image, imgsz=640)

    for result in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = result
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{model.names[int(class_id)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow(f'YOLOv8 Detection - {os.path.basename(file_path)}', image)
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

        results = model.predict(source=frame, imgsz=640)

        new_sign_detected = False

        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = result
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(class_id)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            sign_image = frame[int(y1):int(y2), int(x1):int(x2)]
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
    camera_id = int(selected_camera.get())
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Unable to connect to the camera device {camera_id}.")
        return

    global last_detected_sign
    alarm_played = False

    confidence_threshold = 0.8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640)

        new_sign_detected = False

        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = result
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(class_id)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            if confidence >= confidence_threshold:
                sign_image = frame[int(y1):int(y2), int(x1):int(x2)]
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

        cv2.imshow("Live Feed - YOLOv8 Detection", frame)

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
app.geometry("600x700")
app.title("SignSpotter - YOLOv8 Detection")
app.configure(bg="white")

# Initialize the camera selection variable
selected_camera = ctk.StringVar(value="0")

# Main frame for navigation
main_frame = ctk.CTkFrame(master=app, corner_radius=10, fg_color="white")
main_frame.pack(fill="both", expand=True, pady=(20, 10), padx=10)

# Placeholder for the logo
logo_placeholder = ctk.CTkLabel(master=main_frame, text="[LOGO HERE]", font=("Arial", 20), bg_color="white")
logo_placeholder.pack(pady=20)

# Title label
title_label = ctk.CTkLabel(master=main_frame, text="Welcome to SignSpotter", font=("Arial", 24), bg_color="white")
title_label.pack(pady=20)

# Subtitle
subtitle_label = ctk.CTkLabel(master=main_frame, text="Select which camera you want to use", font=("Arial", 16), bg_color="white")
subtitle_label.pack(pady=10)

# Camera selection dropdown
camera_dropdown = ctk.CTkOptionMenu(master=main_frame, variable=selected_camera, values=["Snapchat camera", "Driod cam"], fg_color="#E34234", text_color="white")
camera_dropdown.pack(pady=10)

# Buttons for functionalities
button_live_feed = ctk.CTkButton(master=main_frame, text="Start Live Feed", command=start_live_feed, fg_color="#E34234", text_color="white")
button_live_feed.pack(pady=20)

button_image_upload = ctk.CTkButton(master=main_frame, text="Upload Image", command=upload_image, fg_color="#E34234", text_color="white")
button_image_upload.pack(pady=10)

button_video_upload = ctk.CTkButton(master=main_frame, text="Upload Video", command=upload_video, fg_color="#E34234", text_color="white")
button_video_upload.pack(pady=10)

# Navigation buttons at the bottom
bottom_frame = ctk.CTkFrame(master=app, fg_color="white")
bottom_frame.pack(fill="x", side="bottom")

settings_button = ctk.CTkButton(master=bottom_frame, text="Settings", fg_color="#E34234", text_color="white", command=lambda: print("Settings clicked"))
settings_button.pack(side="left", expand=True, pady=10, padx=10)

record_button = ctk.CTkButton(master=bottom_frame, text="Record", fg_color="#E34234", text_color="white", command=lambda: print("Record clicked"))
record_button.pack(side="left", expand=True, pady=10, padx=10)

exit_button = ctk.CTkButton(master=bottom_frame, text="Exit", fg_color="#E34234", text_color="white", command=app.quit)
exit_button.pack(side="left", expand=True, pady=10, padx=10)

# Run the application
app.mainloop()
