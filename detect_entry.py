#!/usr/bin/env python3
# universal_sync_detection

import os
import csv
import cv2
from ultralytics import YOLO

def detect_person_universal(video_path: str,
                           model: YOLO,
                           search_start: float = 72.0,
                           search_end: float = 82.0,
                           base_conf_thresh: float = 0.3,
                           base_area_ratio: float = 0.05) -> tuple[int, float, dict] | None:
    """
    Universal detection: supports person detection entering from any direction
    Returns (frame, time, detection_info)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_area = width * height
    
    video_name = os.path.basename(video_path)
    
    # Adjust parameters based on different camera types
    if "sidecam" in video_name.lower():
        # Side camera: person may be small, significantly reduce thresholds
        conf_thresh = base_conf_thresh * 0.5  # Reduce to 0.15
        min_area_ratio = base_area_ratio * 0.25  # Reduce to 1.25%
        persist_frames = 2  # Only need 2 consecutive frames
        use_edge_detection = True  # Enable edge detection
        print(f"[INFO] Sidecam detected - using very relaxed thresholds")
    elif "curve" in video_name.lower():
        # Curve camera: use standard parameters
        conf_thresh = base_conf_thresh
        min_area_ratio = base_area_ratio
        persist_frames = 5
        use_edge_detection = False
        print(f"[INFO] Curvecam detected - using standard thresholds")
    else:
        # Default parameters, but enable edge detection just in case
        conf_thresh = base_conf_thresh
        min_area_ratio = base_area_ratio
        persist_frames = 5
        use_edge_detection = True
        print(f"[INFO] Unknown camera type - using standard thresholds with edge detection")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {video_name}")
    print(f"Search window: {search_start:.1f}s - {search_end:.1f}s")
    print(f"Confidence threshold: {conf_thresh:.3f}")
    print(f"Min area ratio: {min_area_ratio:.4f} ({min_area_ratio * frame_area:.0f} pixels)")
    print(f"Persist frames: {persist_frames}")
    print(f"Edge detection: {'Enabled' if use_edge_detection else 'Disabled'}")
    print(f"{'='*60}")

    # Calculate search range
    start_frame = int(fps * search_start)
    end_frame = int(fps * search_end)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    stable_count = 0
    all_detections = []
    edge_detections = {
        'left': None,    # Left edge detection (x < 0.2)
        'right': None,   # Right edge detection (x > 0.8)
        'top': None,     # Top edge detection (y < 0.2)
        'bottom': None   # Bottom edge detection (y > 0.8)
    }
    
    def get_position_info(center_x, center_y):
        """Get position information of detection box"""
        position = []
        if center_x <= 0.2:
            position.append('left')
        elif center_x >= 0.8:
            position.append('right')
        else:
            position.append('center')
            
        if center_y <= 0.2:
            position.append('top')
        elif center_y >= 0.8:
            position.append('bottom')
        else:
            position.append('middle')
            
        return position
    
    def calculate_position_score(center_x, center_y, area_ratio):
        """Calculate position weight score"""
        # Base score
        base_score = area_ratio
        
        # Edge bonus: closer to edge, more likely to be a person just entering
        edge_bonus = 0
        
        # Left-right edge bonus
        if center_x <= 0.15 or center_x >= 0.85:
            edge_bonus += 0.5  # Strong edge bonus
        elif center_x <= 0.25 or center_x >= 0.75:
            edge_bonus += 0.2  # Weak edge bonus
            
        # Top-bottom edge bonus (smaller, as people usually enter from left/right)
        if center_y <= 0.15 or center_y >= 0.85:
            edge_bonus += 0.1
            
        return base_score + edge_bonus
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # YOLO detection
        results = model(frame[..., ::-1],
                        classes=[0],
                        conf=conf_thresh,
                        verbose=False)

        # Analyze detection results
        max_area = 0
        best_detection = None
        detection_count = len(results[0].boxes) if results[0].boxes is not None else 0
        all_frame_detections = []

        if detection_count > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                
                px1, py1 = int(x1 * width), int(y1 * height)
                px2, py2 = int(x2 * width), int(y2 * height)
                
                box_width = px2 - px1
                box_height = py2 - py1
                area = max(0, box_width * box_height)
                area_ratio = area / frame_area
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Position information
                position_info = get_position_info(center_x, center_y)
                position_score = calculate_position_score(center_x, center_y, area_ratio)
                
                detection_info = {
                    'conf': conf,
                    'area': area,
                    'area_ratio': area_ratio,
                    'center_x': center_x,
                    'center_y': center_y,
                    'position': position_info,
                    'position_score': position_score,
                    'bbox': (px1, py1, px2, py2)
                }
                
                all_frame_detections.append(detection_info)
                
                if area > max_area:
                    max_area = area
                    best_detection = detection_info

        # Record edge detections
        if use_edge_detection and best_detection:
            center_x = best_detection['center_x']
            center_y = best_detection['center_y']
            area_ratio = best_detection['area_ratio']
            
            # Only record sufficiently large detections as edge candidates
            min_edge_area = min_area_ratio * 0.5  # Minimum area requirement for edge detection
            
            if area_ratio >= min_edge_area:
                # Left edge
                if center_x <= 0.2 and edge_detections['left'] is None:
                    edge_detections['left'] = {
                        'frame': frame_idx,
                        'time': current_time,
                        'detection': best_detection
                    }
                    print(f"📍 Left edge detection: t={current_time:.1f}s, center_x={center_x:.3f}, area_ratio={area_ratio:.4f}")
                
                # Right edge
                if center_x >= 0.8 and edge_detections['right'] is None:
                    edge_detections['right'] = {
                        'frame': frame_idx,
                        'time': current_time,
                        'detection': best_detection
                    }
                    print(f"📍 Right edge detection: t={current_time:.1f}s, center_x={center_x:.3f}, area_ratio={area_ratio:.4f}")
                
                # Top edge
                if center_y <= 0.2 and edge_detections['top'] is None:
                    edge_detections['top'] = {
                        'frame': frame_idx,
                        'time': current_time,
                        'detection': best_detection
                    }
                    print(f"📍 Top edge detection: t={current_time:.1f}s, center_y={center_y:.3f}, area_ratio={area_ratio:.4f}")
                
                # Bottom edge
                if center_y >= 0.8 and edge_detections['bottom'] is None:
                    edge_detections['bottom'] = {
                        'frame': frame_idx,
                        'time': current_time,
                        'detection': best_detection
                    }
                    print(f"📍 Bottom edge detection: t={current_time:.1f}s, center_y={center_y:.3f}, area_ratio={area_ratio:.4f}")

        # Record frame information
        frame_info = {
            'frame': frame_idx,
            'time': current_time,
            'detection_count': detection_count,
            'max_area': max_area,
            'max_area_ratio': max_area / frame_area,
            'meets_threshold': max_area >= min_area_ratio * frame_area,
            'best_detection': best_detection,
            'all_detections': all_frame_detections
        }
        all_detections.append(frame_info)

        # Detailed output (every 0.2 seconds)
        if int((frame_idx - start_frame) % (fps * 0.2)) == 0:
            status = "✓" if max_area >= min_area_ratio * frame_area else "✗"
            extra_info = ""
            if best_detection:
                pos_str = "+".join(best_detection['position'])
                extra_info = f", pos={pos_str}, score={best_detection['position_score']:.3f}"
            
            print(f"{status} t={current_time:.1f}s: {detection_count} persons, "
                  f"max_area_ratio={max_area/frame_area:.4f}{extra_info}")

        # Stability check
        if max_area >= min_area_ratio * frame_area:
            stable_count += 1
            
            if stable_count >= persist_frames:
                first_detection_frame = frame_idx - persist_frames + 1
                first_detection_time = first_detection_frame / fps
                
                print(f"\n🎯 STABLE DETECTION ACHIEVED!")
                print(f"First stable frame: {first_detection_frame} (time {first_detection_time:.2f}s)")
                if best_detection:
                    pos_str = "+".join(best_detection['position'])
                    print(f"Final detection: conf={best_detection['conf']:.3f}, "
                          f"area_ratio={best_detection['area_ratio']:.4f}, position={pos_str}")
                
                detection_result = {
                    'method': 'stable',
                    'position': best_detection['position'] if best_detection else [],
                    'confidence': best_detection['conf'] if best_detection else 0,
                    'area_ratio': best_detection['area_ratio'] if best_detection else 0
                }
                
                cap.release()
                return (first_detection_frame, first_detection_time, detection_result)
        else:
            stable_count = 0
            
        frame_idx += 1

    cap.release()
    
    # If no stable detection, use edge detection strategy
    print(f"\n❌ NO STABLE DETECTION")
    
    # Priority: left > right > top > bottom
    edge_priority = ['left', 'right', 'top', 'bottom']
    
    for edge_type in edge_priority:
        if edge_detections[edge_type] is not None:
            edge_info = edge_detections[edge_type]
            det = edge_info['detection']
            
            print(f"🎯 USING {edge_type.upper()} EDGE DETECTION!")
            print(f"Frame {edge_info['frame']} (t={edge_info['time']:.2f}s)")
            print(f"Detection: conf={det['conf']:.3f}, area_ratio={det['area_ratio']:.4f}")
            print(f"Position: center_x={det['center_x']:.3f}, center_y={det['center_y']:.3f}")
            
            detection_result = {
                'method': f'{edge_type}_edge',
                'position': det['position'],
                'confidence': det['conf'],
                'area_ratio': det['area_ratio']
            }
            
            return (edge_info['frame'], edge_info['time'], detection_result)
    
    # Final fallback strategy
    valid_detections = [d for d in all_detections if d['detection_count'] > 0]
    
    if valid_detections:
        print(f"Stats: {len(valid_detections)} frames with detections")
        
        # Select detection with highest position score
        best_by_position = max(valid_detections, 
                              key=lambda x: x['best_detection']['position_score'] if x['best_detection'] else 0)
        
        if best_by_position['best_detection']:
            det = best_by_position['best_detection']
            if det['area_ratio'] >= min_area_ratio * 0.3:  # 30% threshold
                pos_str = "+".join(det['position'])
                print(f"Using best position score detection: Frame {best_by_position['frame']} "
                      f"(t={best_by_position['time']:.1f}s), position={pos_str}, "
                      f"area_ratio={det['area_ratio']:.4f}")
                
                detection_result = {
                    'method': 'best_position',
                    'position': det['position'],
                    'confidence': det['conf'],
                    'area_ratio': det['area_ratio']
                }
                
                return (best_by_position['frame'], best_by_position['time'], detection_result)
    
    return None


def main():
    video_dir = "run13"
    vids = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(".mp4"))
    if not vids:
        print("[ERROR] No .mp4 files found in run13/")
        return

    print("[INFO] Loading YOLOv8-n model for universal sync detection…")
    model = YOLO("yolov8n.pt")

    results = {}
    
    for fn in vids:
        path = os.path.join(video_dir, fn)
        result = detect_person_universal(
            path,
            model,
            search_start=72.0,
            search_end=82.0,
            base_conf_thresh=0.3,
            base_area_ratio=0.05
        )
        results[fn] = result

    # Results analysis
    print(f"\n{'='*80}")
    print("UNIVERSAL SYNC DETECTION RESULTS")
    print(f"{'='*80}")
    
    for fn in vids:
        result = results[fn]
        if result:
            frame_idx, timestamp, detection_info = result
            method = detection_info['method']
            position = "+".join(detection_info['position'])
            print(f"{fn:20} | Frame: {frame_idx:6d} | Time: {timestamp:6.2f}s | "
                  f"Method: {method:12s} | Pos: {position:12s} | "
                  f"Conf: {detection_info['confidence']:.3f}")
        else:
            print(f"{fn:20} | No detection")

    # Calculate synchronization offsets
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        print(f"\nSYNCHRONIZATION ANALYSIS:")
        
        # Find earliest detection as reference
        times = [(fn, result[1]) for fn, result in valid_results.items()]
        times.sort(key=lambda x: x[1])
        
        base_video, base_time = times[0]
        print(f"Base reference: {base_video} at {base_time:.2f}s")
        
        sync_offsets = []
        for fn, timestamp in times:
            time_offset = timestamp - base_time
            sync_offsets.append((fn, time_offset))
            print(f"{fn:20} | Entry: {timestamp:6.2f}s | Offset: {time_offset:+6.3f}s")
        
        # Analyze synchronization quality
        max_offset = max(offset for _, offset in sync_offsets)
        print(f"\nMax desync: {max_offset:.3f} seconds")
        
        if max_offset > 2.0:
            print("⚠️  Large desync detected - may need manual verification")
        elif max_offset > 0.5:
            print("⚠️  Moderate desync - acceptable for most applications")
        else:
            print("✅ Good synchronization")

    # Save detailed results
    csv_path = "universal_sync_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "entry_frame", "entry_time", "sync_offset_sec", 
                        "detection_method", "position", "confidence", "area_ratio"])
        
        if valid_results:
            base_time = min(result[1] for result in valid_results.values())
            for fn in vids:
                result = results[fn]
                if result:
                    frame_idx, timestamp, detection_info = result
                    sync_offset = timestamp - base_time
                    position_str = "+".join(detection_info['position'])
                    writer.writerow([fn, frame_idx, timestamp, sync_offset,
                                   detection_info['method'], position_str,
                                   detection_info['confidence'], detection_info['area_ratio']])
                else:
                    writer.writerow([fn, None, None, None, None, None, None, None])

    print(f"\n[INFO] Universal sync results saved to {csv_path}")


if __name__ == "__main__":
    main()