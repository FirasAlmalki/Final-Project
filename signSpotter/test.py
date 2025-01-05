import os
from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog
import playsound  # Import the playsound library

# Initialize the model with the trained weights
model = YOLO('C:/Users/Admin/Desktop/Project/SignSpotter/runs/detect/train3/weights/best.pt')

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

    # Predict objects in the image
    results = model.predict(source=image, imgsz=640)

    for result in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = result
        # Draw rectangle around detected object
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Add label text (class and confidence)
        cv2.putText(image, f'{model.names[int(class_id)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the image with detections
    cv2.imshow(f'YOLOv8 Detection - {os.path.basename(file_path)}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process video files
def process_video(file_path):
    video = cv2.VideoCapture(file_path)

    # Get the video parameters (width, height, frames per second)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the processed video
    output_path = os.path.join(output_dir, f"processed_{os.path.basename(file_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 videos
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame in the video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Predict objects in the frame
        results = model.predict(source=frame, imgsz=640)

        # Flag to detect when a new sign is detected
        new_sign_detected = False

        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = result
            # Draw rectangle around detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label text (class and confidence)
            cv2.putText(frame, f'{model.names[int(class_id)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Capture the sign image if a new sign is detected
            sign_image = frame[int(y1):int(y2), int(x1):int(x2)]
            if sign_image is not None and (last_detected_sign is None or (last_detected_sign != sign_image).any()):
                last_detected_sign = sign_image
                new_sign_detected = True

        # If a new sign was detected, show it in a separate window
        if new_sign_detected and last_detected_sign is not None:
            cv2.imshow("Last Detected Sign", last_detected_sign)

        # Write the frame to the new video file
        out.write(frame)

        # Optional: Display the frame being processed
        cv2.imshow(f'Processing {os.path.basename(file_path)}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved as {output_path}")


def process_live_feed():
    # Open the webcam feed from DroidCam (assuming DroidCam is running and accessible)
    cap = cv2.VideoCapture(1)  # Use device index (0 is usually the default camera)

    if not cap.isOpened():
        print("Error: Unable to connect to the live feed from DroidCam.")
        return

    global last_detected_sign
    alarm_played = False  # Flag to prevent repeated alarm sound

    # Define confidence threshold (e.g., 0.8)
    confidence_threshold = 0.8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict objects in the frame
        results = model.predict(source=frame, imgsz=640)

        # Flag to detect when a new sign is detected
        new_sign_detected = False

        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = result
            # Draw rectangle around detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label text (class and confidence)
            cv2.putText(frame, f'{model.names[int(class_id)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Only capture the sign if the confidence is above the threshold
            if confidence >= confidence_threshold:
                sign_image = frame[int(y1):int(y2), int(x1):int(x2)]
                if sign_image is not None:
                    # If last_detected_sign is None, set it to the current sign
                    if last_detected_sign is None:
                        last_detected_sign = sign_image
                        new_sign_detected = True
                    else:
                        # Resize the new sign to the same size as the last detected sign
                        resized_sign_image = cv2.resize(sign_image,
                                                        (last_detected_sign.shape[1], last_detected_sign.shape[0]))

                        # Compare the resized new sign with the last detected sign
                        if not (last_detected_sign == resized_sign_image).all():
                            last_detected_sign = resized_sign_image
                            new_sign_detected = True

        # If a new sign was detected and confidence was high, show it in a separate window
        if new_sign_detected and last_detected_sign is not None:
            cv2.imshow("Last Detected Sign", last_detected_sign)

            # Play alarm sound if not already played
            if confidence >= confidence_threshold:
                playsound.playsound("signSpotter/alarm.mp3")  # Replace with your own sound file path
                alarm_played = True  # Set flag to prevent continuous sound

        # Display the frame
        cv2.imshow("Live Feed - YOLOv8 Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


# Function to handle image file upload
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        process_image(file_path)

# Function to handle video file upload
def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        process_video(file_path)

# Function to start live feed
def start_live_feed():
    process_live_feed()

# Create the main window
root = tk.Tk()
root.title("SignSpotter - YOLOv8 Detection")

# Set window size
root.geometry("300x200")

# Create buttons for different functionalities
button_live_feed = tk.Button(root, text="Start Live Feed", command=start_live_feed)
button_live_feed.pack(pady=20)

button_image_upload = tk.Button(root, text="Upload Image", command=upload_image)
button_image_upload.pack(pady=10)

button_video_upload = tk.Button(root, text="Upload Video", command=upload_video)
button_video_upload.pack(pady=10)

# Run the GUI loop
root.mainloop()
