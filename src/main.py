import os
import cv2
import time
from ultralytics import YOLO
import supervision as sv

# --- CẤU HÌNH ---
SOURCE_VIDEO_PATH = "/content/data/input/test_video.mp4"
TARGET_VIDEO_PATH = "/content/data/output/output_phase1.mp4"
os.makedirs("data/output", exist_ok=True)

# 1. LOAD MODEL
print("📦 Đang load model YOLOv8s...")
model = YOLO('yolov8s.pt')

# 2. ĐỌC VIDEO VÀ KHỞI TẠO WRITER
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not cap.isOpened():
    print("❌ Lỗi: Không đọc được video đầu vào!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (width, height))

# 3. KHỞI TẠO TRACKER & ANNOTATORS (ĐÃ FIX LỖI Ở ĐÂY)
tracker = sv.ByteTrack()

# Bút vẽ Cầu thủ (Màu mặc định tự đổi theo ID)
player_box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=3)

# Bút vẽ Bóng (Ép cứng màu Đỏ bằng sv.Color)
RED_COLOR = sv.Color.from_hex("#FF0000")
ball_box_annotator = sv.BoxAnnotator(color=RED_COLOR, thickness=2)
trace_annotator = sv.TraceAnnotator(color=RED_COLOR, thickness=2, trace_length=fps * 2)

# 4. VÒNG LẶP XỬ LÝ CHÍNH
print(f"🚀 Bắt đầu xử lý {total_frames} frames...")
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # AI Inference
    results = model(frame, classes=[0, 32], device='cuda', imgsz=640, half=True, verbose=False)[0]
    
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    players = detections[detections.class_id == 0]
    ball = detections[detections.class_id == 32]

    annotated_frame = frame.copy()

    # Vẽ cầu thủ
    if len(players) > 0:
        annotated_frame = player_box_annotator.annotate(annotated_frame, players)
        labels = [f"#{t_id}" for t_id in players.tracker_id]
        annotated_frame = label_annotator.annotate(annotated_frame, players, labels=labels)

    # Vẽ bóng bằng bút màu đỏ đã chuẩn bị
    if len(ball) > 0:
        annotated_frame = ball_box_annotator.annotate(annotated_frame, ball)
        annotated_frame = trace_annotator.annotate(annotated_frame, ball)

    # LƯU FRAME
    out.write(annotated_frame)
    frame_count += 1

    if frame_count % 50 == 0:
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed
        print(f"👉 Tiến độ: {frame_count}/{total_frames} frames | Tốc độ: {current_fps:.1f} FPS")

# DỌN DẸP
cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"✅ HOÀN TẤT! Tổng thời gian: {total_time:.1f} giây.")
print(f"🎬 Hãy mở file: {TARGET_VIDEO_PATH} để xem thành quả!")