from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt


midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  
midas.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

model = YOLO('yolov8n.pt')
model.to('cpu')
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture("customdata/Aerial view of traffic - SGK Tech (1080p, h264).mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

depth_out = cv2.VideoWriter('depth_output.mp4', fourcc, 30.0,
                            (int(cap.get(3)), int(cap.get(4))))



frame_count = 0
prev_centers = {}
fps = 24

while frame_count < 500:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    input_batch = transform(frame).to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label in ['car', 'truck', 'bus', 'motorbike']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        h, w = depth_map.shape
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        box_width = x2 - x1
        object_depth = depth_map[cy, cx]
        if box_width == 0 or object_depth == 0:
            continue


        if label == 'car': width_real = 1.8
        elif label == 'bus': width_real = 2.5
        elif label == 'truck': width_real = 2.4
        elif label == 'motorbike': width_real = 0.8
        mpp = width_real / box_width * object_depth/np.mean(depth_map)



        if track_id in prev_centers:
            
            prev_cx, prev_cy = prev_centers[track_id]
            pixel_distance = math.hypot(cx - prev_cx, cy - prev_cy)
            speed_m_s = (pixel_distance * mpp) * fps
            speed_kmh = speed_m_s * 3.6

            cv2.putText(frame, f"{speed_kmh:.1f} km/h", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        prev_centers[track_id] = (cx, cy)

    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)  
    depth_out.write(depth_vis)
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
depth_out.release()
