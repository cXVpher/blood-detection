import sys
import cv2
import numpy as np
import mss # Untuk menangkap layar
import time
import pandas as pd # Untuk logging ke file Excel
import subprocess
import torch
from PyQt5.QtCore import QTimer, Qt # Untuk pengatur waktu dan kontrol posisi
from PyQt5.QtGui import QImage, QPixmap # Untuk konversi frame ke GUI
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget # Elemen UI
from ultralytics import YOLO

class ScreenClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Classifier - YOLOv8")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("Press 'Start' to begin", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(800, 450)

        self.btn_start = QPushButton("Start Detection")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("Stop Detection")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.btn_save_video = QPushButton("Start Recording")
        self.btn_save_video.clicked.connect(self.toggle_video_recording)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_save_video)
        self.setLayout(layout)

        self.model = YOLO('best.pt')
        self.model.to('cuda')
        print("Using device:", "CUDA" if torch.cuda.is_available() else "CPU")

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_screen)
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]

        self.recording = False
        self.video_writer = None
        self.video_filename = None

        self.log_file = "blood_detection_log.xlsx"
        self.log_data = []
        self.frame_count = 0

        self.start_time = None
        self.frame_timestamps = []
        self.frame_durations = []

    def start_detection(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.start_time = time.time()
        self.frame_count = 0
        self.frame_durations = []
        self.frame_timestamps = []
        self.log_data = []
        self.timer.start(30)

    def stop_detection(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.label.setText("Detection Stopped")
        if self.recording:
            self.toggle_video_recording()
        self.save_log_to_excel()

    def toggle_video_recording(self):
        if self.recording:
            self.recording = False
            self.btn_save_video.setText("Start Recording")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print(f"Video saved: {self.video_filename}")
                self.adjust_video_fps()
        else:
            self.recording = True
            self.btn_save_video.setText("Stop Recording")
            self.video_filename = f"recorded_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.frame_timestamps = []
            self.video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, 10,
                (self.monitor["width"], self.monitor["height"])
            )
            print(f"Recording started: {self.video_filename}")

    def capture_screen(self):
        self.frame_count += 1
        start_frame_time = time.time()

        screenshot = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        results = self.model.predict(frame, verbose=False)

        blood_detected = False
        detection_confidence = None

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                class_id = int(cls)
                confidence = float(conf)

                if class_id == 0:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"Blood: {confidence:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    blood_detected = True
                    detection_confidence = confidence

                    # Logging detail
                    log_entry = [
                        self.frame_count,
                        confidence,
                        x1, y1, x2, y2,
                        result.speed["preprocess"],
                        result.speed["inference"],
                        result.speed["postprocess"]
                    ]
                    self.log_data.append(log_entry)
                    print(f"Blood detected on frame {self.frame_count} (Confidence: {confidence:.2f})")

        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            self.frame_timestamps.append(time.time())

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

        end_frame_time = time.time()
        self.frame_durations.append(end_frame_time - start_frame_time)

    def save_log_to_excel(self):
        if self.log_data:
            columns = [
                "Frame", "Confidence", "X1", "Y1", "X2", "Y2",
                "Preprocess Time (ms)", "Inference Time (ms)", "Postprocess Time (ms)"
            ]
            df = pd.DataFrame(self.log_data, columns=columns)
            df.to_excel(self.log_file, index=False)
            print(f"Blood detection log saved to {self.log_file}")

    def adjust_video_fps(self):
        if self.video_filename and self.frame_timestamps:
            num_frames = len(self.frame_timestamps)
            if num_frames < 2:
                print("⚠️ Not enough frames to calculate FPS.")
                return

            total_duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if total_duration <= 0:
                print("⚠️ Invalid duration. FPS = 0.")
                return

            correct_fps = num_frames / total_duration
            if correct_fps > 60:
                correct_fps = 60

            print(f"Fixing FPS from 10 to {correct_fps:.2f} using FFmpeg...")
            fixed_filename = self.video_filename.replace(".mp4", "_fixed.mp4")
            command = f'ffmpeg -i "{self.video_filename}" -r {correct_fps:.2f} "{fixed_filename}"'
            subprocess.run(command, shell=True)
            print(f"Fixed video saved as {fixed_filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScreenClassifierApp()
    window.show()
    sys.exit(app.exec_())
