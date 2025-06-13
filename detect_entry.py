#!/usr/bin/env python3
# detect_entry_hardstart.py

import os
import csv
import cv2
from ultralytics import YOLO

def detect_first_person_after(video_path: str,
                              model: YOLO,
                              start_time: float = 70.0,
                              conf_thresh: float = 0.3,
                              min_area_ratio: float = 0.05,
                              persist_frames: int = 5) -> int | None:
    """
    跳过前 start_time 秒，再逐帧检测 person。
    只有当检测到的最大 person bbox 面积 ≥ min_area_ratio*frame_area 且
    连续 persist_frames 帧都满足，才算“入镜”，返回该帧的索引。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_area = width * height

    # 跳到 start_time 对应的帧
    start_frame = int(fps * start_time)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx    = start_frame
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 用 YOLOv8 检测 person (class 0)
        results = model(frame[..., ::-1],
                        classes=[0],
                        conf=conf_thresh,
                        verbose=False)

        # 找到最大 bbox 面积
        max_area = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxyn[0]  # 归一化坐标
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            area = max(0, (px2 - px1) * (py2 - py1))
            if area > max_area:
                max_area = area

        # 判断是否达到稳定入镜条件
        if max_area >= min_area_ratio * frame_area:
            stable_count += 1
            if stable_count >= persist_frames:
                # 回退到第一次满足条件的帧
                cap.release()
                return frame_idx - persist_frames + 1
        else:
            stable_count = 0

        frame_idx += 1

    cap.release()
    return None


def main():
    video_dir = "run13"
    vids = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(".mp4"))
    if not vids:
        print("[ERROR] No .mp4 files found in run13/")
        return

    print("[INFO] Loading YOLOv8-n model…")
    model = YOLO("yolov8n.pt")  # 若无本地权重，会自动下载

    entries = {}
    for fn in vids:
        path = os.path.join(video_dir, fn)
        print(f"[INFO] Processing {fn} …", end="", flush=True)
        idx = detect_first_person_after(
            path,
            model,
            start_time=70.0,       # 跳过前 70 秒
            conf_thresh=0.3,
            min_area_ratio=0.05,   # bbox ≥ 5% 画面面积
            persist_frames=5       # 连续 5 帧
        )
        print(f" first stable at frame {idx}")
        entries[fn] = idx

    # 以第一路为基准，计算偏移
    base = vids[0]
    base_idx = entries[base]
    csv_path = "hardstart_entry_offsets.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "entry_frame", "offset_to_base"])
        for fn in vids:
            idx = entries[fn]
            offset = None if (idx is None or base_idx is None) else idx - base_idx
            writer.writerow([fn, idx, offset])

    print(f"\n[INFO] Written alignment table → {csv_path}")


if __name__ == "__main__":
    main()