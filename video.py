import cv2
import torch
import numpy as np
import threading
import time
from queue import Queue
from PIL import Image
from mlaf import MlafNet
model_path = 'new_model.pth'
# 训练模型路径
model = MlafNet().to('cuda:0')
model.load_state_dict(torch.load(model_path))
model.eval()
video_path = "01R002LYME_part.mp4"
# 视频路径
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video file:", video_path)
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale_factor = 0.6
frame_skip = 1

print(f"📹 Video Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {frame_count}")
print(f"🔄 Processing Strategy: Scaling to {int(scale_factor * 100)}% + Parallel Inference")

frame_queue = Queue()
processed_queue = Queue()

frame_index = 0
start_time = time.time()
target_frame_duration = 1.0 / fps

def inference_thread():
    while True:
        if not frame_queue.empty():
            frame_resized = frame_queue.get()
            infer_start = time.time()

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            with torch.no_grad():
                _, _, D = model.predict(pil_image)

            infer_time = time.time() - infer_start
            print(f"⏳ Model Inference Time: {infer_time:.4f} sec")

            def pil_to_cv(img):
                if isinstance(img, Image.Image):
                    img = np.array(img)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            frame_specular_removed = pil_to_cv(D)
            frame_specular_removed_resized = cv2.resize(frame_specular_removed, (frame_resized.shape[1], frame_resized.shape[0]))

            processed_queue.put((frame_resized, frame_specular_removed_resized))

thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

while cap.isOpened():
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing completed")
        break

    frame_index += 1

    if frame_index % frame_skip != 0:
        continue

    expected_time = frame_index * target_frame_duration
    actual_time = time.time() - start_time
    sleep_time = max(0, expected_time - actual_time)
    time.sleep(sleep_time)

    elapsed_time = time.time() - start_time
    actual_fps = frame_index / elapsed_time
    playback_delay = actual_time - expected_time

    print(f"🎥 Real-Time FPS: {actual_fps:.2f} | ⏱️ Playback Delay: {playback_delay:.4f} sec")

    frame_resized = cv2.resize(frame, (int(frame_width * scale_factor), int(frame_height * scale_factor)))

    if frame_queue.qsize() < 3:
        frame_queue.put(frame_resized)

    if not processed_queue.empty():
        original_frame, processed_frame = processed_queue.get()
        combined_frame = np.hstack((original_frame, processed_frame))
        cv2.imshow('Original (Left) | Specular Removed (Right)', combined_frame)

    frame_latency = time.time() - frame_start_time
    print(f"⚡ Frame Processing Latency: {frame_latency:.4f} sec")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()