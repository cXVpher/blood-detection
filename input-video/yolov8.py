import sys
import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Cek argumen input
if len(sys.argv) < 2:
    print("Usage: python yolov8.py <video_path>")
    sys.exit(1)

video_source = sys.argv[1]
model = YOLO('best.pt')

# Cek apakah video bisa dibuka
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Cannot open video source.")
    sys.exit(1)

# Buat folder untuk menyimpan hasil
output_folder = "output_videos"
os.makedirs(output_folder, exist_ok=True)

# Ambil nama file asli tanpa ekstensi
video_name = os.path.splitext(os.path.basename(video_source))[0]
output_video_path = os.path.join(output_folder, f"{video_name}_detected.avi")
output_log_path = os.path.join(output_folder, f"{video_name}_log.xlsx")

# Ambil informasi dari video input
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Buat objek VideoWriter untuk menyimpan hasil deteksi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Skala ukuran tampilan jendela (misalnya 50% dari ukuran asli)
scale_percent = 50  # Ubah sesuai kebutuhan

# Buat list untuk menyimpan log deteksi
log_data = []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Jalankan deteksi YOLOv8
    results = model(frame)
    annotated_frame = results[0].plot()

    # Simpan hasil ke video output
    out.write(annotated_frame)

    # Ambil informasi dari hasil deteksi
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())  # ID kelas objek yang terdeteksi
            confidence = float(box.conf[0].item())  # Confidence score
            bbox = [round(coord.item(), 2) for coord in box.xyxy[0]]  # Bounding box [x1, y1, x2, y2]

            # Simpan ke log list
            log_data.append([frame_count, class_id, confidence, *bbox, result.speed["preprocess"], result.speed["inference"], result.speed["postprocess"]])

    # Resize frame untuk tampilan lebih kecil
    width = int(annotated_frame.shape[1] * scale_percent / 100)
    height = int(annotated_frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(annotated_frame, (width, height), interpolation=cv2.INTER_AREA)

    # Tampilkan hasil deteksi dalam jendela kecil
    cv2.imshow('YOLOv8 Detection', resized_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Simpan log sebagai file Excel
df = pd.DataFrame(log_data, columns=["Frame", "Class_ID", "Confidence", "X1", "Y1", "X2", "Y2", "Preprocess_Time", "Inference_Time", "Postprocess_Time"])
df.to_excel(output_log_path, index=False)

# Tutup video dan simpan file
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detection video saved in: {output_video_path}")
print(f"Detection log saved in: {output_log_path}")