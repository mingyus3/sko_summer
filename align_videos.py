#!/usr/bin/env python3
import os, cv2

# ——— 1) 这里填你的偏移（帧数） ———
offsets = {
    "curvecam.mp4": 0,
    "curvecam2.mp4": 53,
    "sidecam.mp4": 359,
}

# ——— 2) 参数配置 ———
fps     = 56.0    # ← 改成你的实际帧率
src_dir = "run13"
dst_dir = "aligned"

os.makedirs(dst_dir, exist_ok=True)

# ——— 3) 主循环 ———
for fn, off in offsets.items():
    in_path  = os.path.join(src_dir, fn)
    out_path = os.path.join(dst_dir, fn)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {in_path}")
        continue

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # 从偏移帧开始读
    cap.set(cv2.CAP_PROP_POS_FRAMES, off)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Aligned {fn}: dropped first {off} frames, wrote {frame_count} frames to {out_path}")
