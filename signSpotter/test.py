from inference_sdk import InferenceHTTPClient
import customtkinter as ctk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import playsound
import numpy as np
import cv2
from ultralytics import YOLO

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Uv06rawidNIWdABmMvGC"
)

# Define output directory for processed files
output_dir = 'output'  # Change this to any valid path

class ImageProcessor:
    def __init__(self):
        self.model_path = "C:/Users/Admin/Desktop/Final/Final-Project/signSpotter/runs/detect/train2/best.pt"
        self.model = YOLO(self.model_path)

    def process_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load the image {file_path}.")
            return

        # Perform inference
        results = self.model(image)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                class_id = int(box.cls[0].item())  # Class ID
                class_name = self.model.names[class_id]  # Class name

                # Draw bounding box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{class_name} {confidence:.2f}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (36, 255, 12), 2)

        cv2.imshow(f'YOLOv8 Detection - {os.path.basename(file_path)}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class VideoProcessor:
    def __init__(self):
        self.model_path = "C:/Users/Admin/Desktop/Final/Final-Project/signSpotter/runs/detect/train2/best.pt"
        self.model = YOLO(self.model_path)
        self.last_detected_sign = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def process_video(self, file_path):
        video = cv2.VideoCapture(file_path)

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(self.output_dir, f"processed_{os.path.basename(file_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving video
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        frame_skip = 3  # Process every 3rd frame to optimize performance

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            results = self.model(frame)
            new_sign_detected = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = self.model.names[class_id]

                    if confidence >= 0.8:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {confidence:.2f}',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (36, 255, 12), 2)

                        sign_image = frame[y1:y2, x1:x2]
                        if sign_image is not None:
                            if self.last_detected_sign is None:
                                self.last_detected_sign = sign_image
                                new_sign_detected = True
                            else:
                                resized_sign_image = cv2.resize(sign_image,
                                                                (self.last_detected_sign.shape[1],
                                                                 self.last_detected_sign.shape[0]))
                                if not np.array_equal(self.last_detected_sign, resized_sign_image):
                                    self.last_detected_sign = resized_sign_image
                                    new_sign_detected = True

            if new_sign_detected and self.last_detected_sign is not None:
                sign_resized = cv2.resize(self.last_detected_sign, (x2 - x1, y2 - y1))
                cv2.imshow("Last Detected Sign", sign_resized)

            # Save the processed frame
            if new_sign_detected:
                frame_filename = os.path.join(self.output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)

            out.write(frame)  # Save processed frame to video
            cv2.imshow(f'Processing {os.path.basename(file_path)}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Processed video saved as {output_path}")


class LiveFeedProcessor:
    def __init__(self):
        self.last_detected_sign = None
        self.alarm_played = False
        self.confidence_threshold = 0.8
        self.model_path = "C:/Users/Admin/Desktop/Final/Final-Project/signSpotter/runs/detect/train2/best.pt"

        # Load the local YOLOv8 model
        self.model = YOLO(self.model_path)

    def process_live_feed(self, selected_camera_index):
        cap = cv2.VideoCapture(selected_camera_index)
        if not cap.isOpened():
            print("Error: Unable to connect to the selected camera.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 inference on the frame
            results = self.model(frame)

            new_sign_detected = False

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                    confidence = box.conf[0].item()  # Confidence score
                    class_id = int(box.cls[0].item())  # Class ID
                    class_name = self.model.names[class_id]  # Class name

                    if confidence >= self.confidence_threshold:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{class_name} {confidence:.2f}',
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (36, 255, 12), 2)

                        sign_image = frame[y1:y2, x1:x2]
                        if sign_image is not None:
                            if self.last_detected_sign is None:
                                self.last_detected_sign = sign_image
                                new_sign_detected = True
                            else:
                                resized_sign_image = cv2.resize(sign_image,
                                                                (self.last_detected_sign.shape[1],
                                                                 self.last_detected_sign.shape[0]))
                                if not np.array_equal(self.last_detected_sign, resized_sign_image):
                                    self.last_detected_sign = resized_sign_image
                                    new_sign_detected = True

            if new_sign_detected and self.last_detected_sign is not None:
                cv2.imshow("Last Detected Sign", self.last_detected_sign)
                playsound.playsound("alarm.mp3")  # Play alarm sound
                self.alarm_played = True

            cv2.imshow("Live Feed - YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class SignSpotterApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.geometry("1024x768")
        self.app.title("SignSpotter - YOLO Model Detection")
        self.app.configure(bg="white")

        self.selected_camera_index = 0
        self.confidence_threshold = 0.8

        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.live_feed_processor = LiveFeedProcessor()

        self.setup_ui()

    def setup_ui(self):
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("dark-blue")

        # Main frame for navigation
        self.main_frame = ctk.CTkFrame(master=self.app, corner_radius=10)  # No initial color here
        self.main_frame.pack(fill="both", expand=True, pady=(20, 10), padx=10)

        # Set the default theme
        self.set_theme("Light")  # You can choose "Dark" or "Light" as default theme here

        # Placeholder for the logo
        image_path = "resources/visigen.png"  # Replace with the path to your image
        image = Image.open(image_path)
        image = image.resize((400, 200))  # Resize the image to fit in the frame
        image_tk = ImageTk.PhotoImage(image)

        logo_placeholder = ctk.CTkLabel(master=self.main_frame, image=image_tk, text="")
        logo_placeholder.image = image_tk  # Keep a reference to avoid garbage collection
        logo_placeholder.pack(pady=20)

        # Title label
        title_label = ctk.CTkLabel(master=self.main_frame, text="Welcome to SignSpotter", font=("Arial", 24))
        title_label.pack(pady=20)

        # Buttons for functionalities
        button_live_feed = ctk.CTkButton(
            master=self.main_frame,
            text="Start Live Feed",
            command=self.start_live_feed,
            fg_color="#ba3b0a",
            text_color="white",
            hover_color="#9a2b07", width=250,  # Main button width
            height=60
        )
        button_live_feed.pack(pady=20)

        button_image_upload = ctk.CTkButton(
            master=self.main_frame,
            text="Upload Image",
            command=self.upload_image,
            fg_color="#ba3b0a",
            text_color="white",
            hover_color="#9a2b07", width=250,
            height=60
        )
        button_image_upload.pack(pady=10)

        button_video_upload = ctk.CTkButton(
            master=self.main_frame,
            text="Upload Video",
            command=self.upload_video,
            fg_color="#ba3b0a",
            text_color="white",
            hover_color="#9a2b07", width=250,
            height=60
        )
        button_video_upload.pack(pady=10)

        # Navigation buttons at the bottom
        bottom_frame = ctk.CTkFrame(master=self.app, fg_color="white")
        bottom_frame.pack(fill="x", side="bottom")

        settings_button = ctk.CTkButton(
            master=bottom_frame,
            text="Settings",
            fg_color="#ba3b0a",
            text_color="white",
            command=self.open_settings,
            hover_color="#9a2b07", width=140,  # Smaller navigation button width
            height=45
        )
        settings_button.pack(side="left", expand=True, pady=10, padx=10)

        info_button = ctk.CTkButton(
            master=bottom_frame,
            text="Information",
            fg_color="#ba3b0a",
            text_color="white",
            command=self.open_information,  # Updated to call open_information
            hover_color="#9a2b07",    width=140,
            height=45
        )
        info_button.pack(side="left", expand=True, pady=10, padx=10)

        exit_button = ctk.CTkButton(
            master=bottom_frame,
            text="Exit",
            fg_color="#ba3b0a",
            text_color="white",
            command=self.app.quit,
            width=140,
            hover_color="#9a2b07",
            height=45
        )
        exit_button.pack(side="left", expand=True, pady=10, padx=10)

    # Add the save_settings method here:
    def save_settings(self, confidence, selected_camera):
        self.confidence_threshold = confidence
        self.selected_camera_index = int(selected_camera.split()[-1])  # Extract the camera index
        print(f"New confidence threshold: {confidence}")
        print(f"Selected camera: {self.selected_camera_index}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            self.image_processor.process_image(file_path)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            self.video_processor.process_video(file_path)

    def start_live_feed(self):
        self.live_feed_processor.process_live_feed(self.selected_camera_index)

    def open_settings(self):
        settings_window = ctk.CTkToplevel(self.app)
        settings_window.title("Settings")
        settings_window.geometry("400x400")

        settings_label = ctk.CTkLabel(settings_window, text="Settings", font=("Arial", 20))
        settings_label.pack(pady=20)

        # Confidence Threshold Slider
        confidence_label = ctk.CTkLabel(settings_window, text="Confidence Threshold:")
        confidence_label.pack(pady=5)
        confidence_slider = ctk.CTkSlider(settings_window, from_=0.1, to=1.0, number_of_steps=10,
                                          button_color="#ba3b0a", button_hover_color="#ba3b0a")
        confidence_slider.set(self.confidence_threshold)
        confidence_slider.pack(pady=10)

        # Theme Selection
        theme_label = ctk.CTkLabel(settings_window, text="Choose Theme:")
        theme_label.pack(pady=5)

        theme_dropdown = ctk.CTkOptionMenu(
            settings_window,
            values=["Light", "Dark"],
            fg_color="#ba3b0a",
            button_color="#ba3b0a",
            button_hover_color="#9a2b07",
            command=lambda theme: [
                print(f"Theme dropdown selected: {theme}"),  # Debugging print
                self.set_theme(theme)
            ]
        )
        theme_dropdown.pack(pady=10)

        # Camera Selection
        camera_label = ctk.CTkLabel(settings_window, text="Select Camera:")
        camera_label.pack(pady=5)
        camera_names = [f"Camera {i}" for i in range(2)]  # Assuming up to 2 cameras
        camera_dropdown = ctk.CTkOptionMenu(settings_window, values=camera_names, fg_color="#ba3b0a",
                                            button_color="#ba3b0a", button_hover_color="#9a2b07")
        camera_dropdown.set(f"Camera {self.selected_camera_index}")  # Set the default value
        camera_dropdown.pack(pady=10)

        # Save Button
        save_button = ctk.CTkButton(
            settings_window,
            text="Save",
            fg_color="#ba3b0a",
            hover_color="#9a2b07",
            command=lambda: [self.save_settings(confidence_slider.get(), camera_dropdown.get())]
        )
        save_button.pack(pady=20)

    def set_theme(self, theme):

        if theme == "Dark":
            self.main_frame.configure(fg_color="#2E2E2E")  # Dark frame background
            ctk.set_appearance_mode("dark")  # Change overall appearance mode
            # Update title label color for dark theme
            title_label = ctk.CTkLabel(master=self.main_frame, text="Welcome to SignSpotter", font=("Arial", 24),
                                       bg_color="#2E2E2E", text_color="white")
            title_label.pack(pady=20)

        elif theme == "Light":
            self.main_frame.configure(fg_color="#eeeee4")  # Light frame background
            ctk.set_appearance_mode("light")  # Change overall appearance mode
            # Update title label color for light theme
            title_label = ctk.CTkLabel(master=self.main_frame, text="Welcome to SignSpotter", font=("Arial", 24),
                                       bg_color="#eeeee4", text_color="black")
            title_label.pack(pady=20)

    def open_information(self):
        info_window = ctk.CTkToplevel(self.app)
        info_window.title("Information")
        info_window.geometry("800x1300")
        info_window.configure(fg_color="#f4f4f4")  # Light gray background (darker than white)

        # Add Title
        title_label = ctk.CTkLabel(info_window, text="Welcome to Traffic Sign Spotter", font=("Arial", 30, "bold"),
                                   text_color="black")
        title_label.pack(pady=20)

        # Add App Brief
        app_brief = ctk.CTkLabel(info_window,
                                 text="This app uses a machine learning model to detect traffic signs in images, videos, and live camera feeds. Follow the steps below to configure the app to your preferences.",
                                 font=("Arial", 20), wraplength=750, text_color="black")
        app_brief.pack(pady=10)

        # Step 1: Information Image
        step1_image = Image.open("resources/step1_image.png")  # Replace with the actual image path
        step1_image = step1_image.resize((700, 350))
        step1_image_tk = ImageTk.PhotoImage(step1_image)
        step1_image_label = ctk.CTkLabel(info_window, image=step1_image_tk)
        step1_image_label.image = step1_image_tk  # Keep reference
        step1_image_label.pack(pady=20)

        step1_text = ctk.CTkLabel(info_window,
                                  text="Step 1: Click on the 'Settings' button to learn about the app and configure",
                                  font=("Arial", 18), text_color="black")
        step1_text.pack(pady=5)

        # Step 2-5: Steps with One Image
        step2_5_text = ctk.CTkLabel(info_window, text="Steps 2-5: Now, follow these steps to configure the app.",
                                    font=("Arial", 20, "bold"), text_color="black")
        step2_5_text.pack(pady=15)

        # Step 2-5 Image
        step2_5_image = Image.open("resources/step2_5_image.png")  # Replace with the actual image path
        step2_5_image = step2_5_image.resize((500, 300))
        step2_5_image_tk = ImageTk.PhotoImage(step2_5_image)
        step2_5_image_label = ctk.CTkLabel(info_window, image=step2_5_image_tk)
        step2_5_image_label.image = step2_5_image_tk  # Keep reference
        step2_5_image_label.pack(pady=20)

        # Step 2: Confidence Threshold Selection
        step2_text = ctk.CTkLabel(info_window,
                                  text="Step 2: Adjust the Confidence Threshold. A lower threshold detects more signs, including those with low confidence, which may lead to false positives. A higher threshold gives more accurate detections but might miss some signs. For better accuracy, set it higher.",
                                  font=("Arial", 18), wraplength=750, text_color="black")
        step2_text.pack(pady=5)

        # Step 3: Theme Selection
        step3_text = ctk.CTkLabel(info_window,
                                  text="Step 3: Choose between Light and Dark themes to personalize the app's appearance.",
                                  font=("Arial", 18), wraplength=750, text_color="black")
        step3_text.pack(pady=5)

        # Step 4: Camera Selection
        step4_text = ctk.CTkLabel(info_window,
                                  text="Step 4: Select the camera you wish to use. Choose from the available cameras on your device.",
                                  font=("Arial", 18), wraplength=750, text_color="black")
        step4_text.pack(pady=5)

        # Step 5: Save Button
        step5_text = ctk.CTkLabel(info_window,
                                  text="Step 5: Click 'Save' to store your settings and start using the app.",
                                  font=("Arial", 18), wraplength=750, text_color="black")
        step5_text.pack(pady=5)

        # Add Final Text
        final_text = ctk.CTkLabel(info_window,
                                  text="Thank you for using Traffic Sign Spotter! We hope this tool helps you detect traffic signs effectively and accurately.",
                                  font=("Arial", 20), wraplength=750, text_color="black")
        final_text.pack(pady=20)

        # Close Button
        close_button = ctk.CTkButton(info_window, text="Close", command=info_window.destroy, fg_color="#ba3b0a",
                                     text_color="white")
        close_button.pack(pady=10)

    def set_theme(self, theme):
        ctk.set_appearance_mode(theme)

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    app = SignSpotterApp()
    app.run()