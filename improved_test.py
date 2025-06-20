#!/usr/bin/env python3
# enhanced_jump_sync_system.py
# Enhanced version with takeoff detection and manual verification

import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# ==================== Module 1: Video Trimming Tools ====================

class VideoTrimmer:
    """Reusable video trimming utility using OpenCV only"""
    
    @staticmethod
    def trim_video(video_path: Union[str, Path], 
                   output_path: Union[str, Path],
                   start_time: float = 70.0,
                   end_time: Optional[float] = None) -> bool:
        """
        Trim video using OpenCV, removing first N seconds
        
        Args:
            video_path: Input video path
            output_path: Output video path  
            start_time: Start time in seconds (default 70.0 to remove first 70s)
            end_time: End time in seconds, None for full video
            
        Returns:
            Success status
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            print(f"[ERROR] Input video not found: {video_path}")
            return False
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(fps * start_time)
        end_frame = int(fps * end_time) if end_time else total_frames
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"[INFO] Trimming {video_path.name} using OpenCV")
        print(f"       Frame range: {start_frame} - {end_frame}")
        
        frame_count = 0
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Progress report every 10 seconds
            if frame_count % int(fps * 10) == 0:
                print(f"       Processed: {frame_count / fps:.1f}s")
        
        cap.release()
        out.release()
        
        print(f"[INFO] Trimming complete: {output_path}")
        return True

    @staticmethod
    def batch_trim_videos(video_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         start_time: float = 70.0,
                         end_time: Optional[float] = None,
                         file_pattern: str = "*.mp4") -> Dict[str, str]:
        """
        Batch trim videos in directory
        
        Returns:
            Mapping dict {original_filename: trimmed_file_path}
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_files = list(video_dir.glob(file_pattern))
        if not video_files:
            print(f"[ERROR] No video files matching {file_pattern} found in {video_dir}")
            return {}
        
        print(f"[INFO] Batch trimming {len(video_files)} video files")
        
        mapping = {}
        for video_file in video_files:
            output_file = output_dir / f"trimmed_{video_file.name}"
            
            if VideoTrimmer.trim_video(video_file, output_file, start_time, end_time):
                mapping[video_file.name] = str(output_file)
            else:
                print(f"[ERROR] Failed to trim: {video_file.name}")
        
        return mapping

# ==================== Module 2: Pose Data Structures ====================

@dataclass
class PoseKeypoint:
    """Single keypoint data structure"""
    x: float
    y: float
    confidence: float
    name: str = ""

@dataclass 
class PersonPose:
    """Complete pose data for one person"""
    keypoints: List[PoseKeypoint]
    bbox: Optional[Tuple[float, float, float, float]] = None
    person_confidence: float = 0.0
    is_main_subject: bool = False

@dataclass
class FramePoseData:
    """Pose data for a single frame"""
    frame_idx: int
    timestamp: float
    people: List[PersonPose]
    frame_width: int = 0
    frame_height: int = 0

# ==================== Module 3: Main Subject Identifier ====================

class MainSubjectIdentifier:
    """Identify main subject (Kaze) from multiple detected people"""
    
    def __init__(self, min_visibility_threshold: float = 0.7):
        """
        Args:
            min_visibility_threshold: Minimum visible keypoint ratio threshold
        """
        self.min_visibility_threshold = min_visibility_threshold
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def identify_main_subject(self, people: List[PersonPose]) -> Optional[PersonPose]:
        """
        Identify main subject from detected people
        
        Strategy:
        1. Select person with most complete keypoints
        2. Prefer person who is more centered and larger in frame
        3. Apply anti-interference filtering for background people
        """
        if not people:
            return None
        
        if len(people) == 1:
            people[0].is_main_subject = True
            return people[0]
        
        # Calculate main subject score for each person
        scored_people = []
        
        for person in people:
            score = self._calculate_main_subject_score(person)
            scored_people.append((score, person))
        
        # Select highest scoring person
        scored_people.sort(key=lambda x: x[0], reverse=True)
        best_score, best_person = scored_people[0]
        
        if best_score > 0.3:  # Minimum score threshold
            best_person.is_main_subject = True
            return best_person
        
        return None
    
    def _calculate_main_subject_score(self, person: PersonPose) -> float:
        """Calculate main subject identification score"""
        score = 0.0
        
        # 1. Keypoint completeness score (40%)
        visible_keypoints = sum(1 for kp in person.keypoints if kp.confidence > 0.5)
        completeness_score = visible_keypoints / len(person.keypoints)
        score += completeness_score * 0.4
        
        # 2. Average confidence score (30%)
        avg_confidence = np.mean([kp.confidence for kp in person.keypoints if kp.confidence > 0.5])
        score += avg_confidence * 0.3
        
        # 3. Size score (20%) - larger person more likely to be main subject
        if person.bbox:
            x1, y1, x2, y2 = person.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            # Normalize to [0,1], assuming max area is 50% of frame
            size_score = min(bbox_area / (1920 * 1080 * 0.5), 1.0)
            score += size_score * 0.2
        
        # 4. Position score (10%) - more centered person preferred
        center_keypoints = [person.keypoints[i] for i in [11, 12]]  # Hip keypoints
        valid_centers = [kp for kp in center_keypoints if kp.confidence > 0.5]
        
        if valid_centers:
            avg_x = np.mean([kp.x for kp in valid_centers])
            avg_y = np.mean([kp.y for kp in valid_centers])
            
            # Normalized distance from frame center
            center_distance = np.sqrt((avg_x - 0.5)**2 + (avg_y - 0.5)**2)
            position_score = max(0, 1 - center_distance * 2)  # Higher score for center position
            score += position_score * 0.1
        
        return score

# ==================== Module 4: YOLO11 Pose Estimator ====================

class YOLO11PoseEstimator:
    """YOLO11m-pose estimation engine"""
    
    def __init__(self, model_name: str = "yolo11m-pose.pt"):
        print(f"[INFO] Loading {model_name} model...")
        self.model = YOLO(model_name)
        self.main_subject_identifier = MainSubjectIdentifier()
        
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        print(f"[INFO] Model loaded successfully")
    
    def estimate_frame_poses(self, frame: np.ndarray, conf_thresh: float = 0.3) -> List[PersonPose]:
        """
        Estimate poses for single frame
        
        Args:
            frame: Input frame (BGR format)
            conf_thresh: Confidence threshold for detection
            
        Returns:
            List of PersonPose objects
        """
        # YOLO prediction
        results = self.model(frame, conf=conf_thresh, verbose=False)
        
        people = []
        
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # Get bounding boxes
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None
            
            for person_idx, keypoints_data in enumerate(results[0].keypoints.data):
                kpts = keypoints_data.cpu().numpy()  # Shape: (17, 3)
                
                # Create keypoint objects
                keypoints = []
                for i, (x, y, conf) in enumerate(kpts):
                    keypoint = PoseKeypoint(
                        x=float(x), y=float(y), confidence=float(conf),
                        name=self.keypoint_names[i] if i < len(self.keypoint_names) else f"point_{i}"
                    )
                    keypoints.append(keypoint)
                
                # Get bounding box
                bbox = None
                if boxes is not None and person_idx < len(boxes):
                    bbox = tuple(boxes[person_idx])
                
                # Calculate person confidence
                person_confidence = np.mean([kp.confidence for kp in keypoints if kp.confidence > 0.5])
                
                person = PersonPose(
                    keypoints=keypoints,
                    bbox=bbox,
                    person_confidence=person_confidence
                )
                people.append(person)
        
        return people

# ==================== Module 5: Takeoff Detector (Enhanced) ====================
class TakeoffDetector:
    """Detect high jump takeoff moment based on foot-ground separation with stricter criteria"""

    def __init__(self, ground_threshold: float = 0.9):
        """
        Args:
            ground_threshold: Normalized Y position threshold for ground contact
        """
        self.ground_threshold = ground_threshold
        self.min_confidence = 0.5       # increase min confidence
        self.min_vertical_velocity = 0.01  # require clearer upward motion

    def detect_takeoff(self,
                       person: PersonPose,
                       prev_person: Optional[PersonPose] = None) -> Dict:
        """
        Detect takeoff based on both feet leaving the ground and upward velocity.
        Returns a dict with detection flag, score, velocities, etc.
        """
        # default result
        result = {
            'takeoff_detected': False,
            'takeoff_score': 0.0,
            'left_foot_airborne': False,
            'right_foot_airborne': False,
            'vertical_velocity': 0.0,
            'confidence': 0.0
        }

        # must have full keypoints
        if not person.keypoints or len(person.keypoints) < 17:
            return result

        left_ankle = person.keypoints[15]
        right_ankle = person.keypoints[16]
        left_hip = person.keypoints[11]
        right_hip = person.keypoints[12]

        # check ankle confidence
        if left_ankle.confidence < self.min_confidence or right_ankle.confidence < self.min_confidence:
            return result

        # determine airborne each foot
        result['left_foot_airborne'] = self._is_foot_airborne(left_ankle, left_hip)
        result['right_foot_airborne'] = self._is_foot_airborne(right_ankle, right_hip)

        # calculate vertical velocity if possible
        if prev_person and prev_person.keypoints:
            prev_left_hip = prev_person.keypoints[11]
            prev_right_hip = prev_person.keypoints[12]
            if prev_left_hip.confidence > self.min_confidence and prev_right_hip.confidence > self.min_confidence:
                curr_y = (left_hip.y + right_hip.y) / 2
                prev_y = (prev_left_hip.y + prev_right_hip.y) / 2
                result['vertical_velocity'] = prev_y - curr_y

        # only detect takeoff if both feet airborne AND sufficient upward motion
        if result['left_foot_airborne'] and result['right_foot_airborne']:
            if result['vertical_velocity'] > self.min_vertical_velocity:
                result['takeoff_detected'] = True
                result['takeoff_score'] = 1.0

        # confidence = average ankle confidence
        result['confidence'] = (left_ankle.confidence + right_ankle.confidence) / 2
        return result

    def _is_foot_airborne(self, ankle: PoseKeypoint, hip: PoseKeypoint) -> bool:
        """Check if foot is airborne by relative position to hip."""
        # already guaranteed ankle.confidence >= min_confidence
        # simple test: ankle must be at least ground_threshold above bottom
        if ankle.y < self.ground_threshold:
            return True
        # or ankle is significantly above hip
        if hip.confidence > self.min_confidence:
            if (hip.y - ankle.y) < 0.3:
                return True
        return False


# ==================== Module 6: Landing Detector (Enhanced) ====================
class LandingDetector:
    """Detect landing on mat after takeoff with stricter criteria"""

    def __init__(self, mat_threshold: float = 0.85):
        """
        Args:
            mat_threshold: Y position threshold for mat contact
        """
        self.mat_threshold = mat_threshold
        self.min_confidence = 0.5  # increase min confidence

    def detect_landing(self,
                       person: PersonPose,
                       prev_person: Optional[PersonPose] = None) -> Dict:
        """
        Detect landing based on both feet contacting mat and downward motion.
        Returns a dict with detection flag, score, type, etc.
        """
        result = {
            'landing_detected': False,
            'landing_score': 0.0,
            'landing_type': 'NO_LANDING',
            'left_foot_contact': False,
            'right_foot_contact': False,
            'downward_velocity': 0.0,
            'confidence': 0.0
        }

        if not person.keypoints or len(person.keypoints) < 17:
            return result

        left_ankle = person.keypoints[15]
        right_ankle = person.keypoints[16]
        left_knee = person.keypoints[13]
        right_knee = person.keypoints[14]

        # require good confidence
        if left_ankle.confidence < self.min_confidence or right_ankle.confidence < self.min_confidence:
            return result

        # determine contact each foot
        result['left_foot_contact'] = self._is_foot_on_mat(left_ankle, left_knee)
        result['right_foot_contact'] = self._is_foot_on_mat(right_ankle, right_knee)

        # downward velocity if prev available
        if prev_person and prev_person.keypoints:
            prev_la = prev_person.keypoints[15]
            prev_ra = prev_person.keypoints[16]
            if prev_la.confidence > self.min_confidence:
                dv = left_ankle.y - prev_la.y
                if dv > 0:
                    result['downward_velocity'] = max(result['downward_velocity'], dv)
            if prev_ra.confidence > self.min_confidence:
                dv = right_ankle.y - prev_ra.y
                if dv > 0:
                    result['downward_velocity'] = max(result['downward_velocity'], dv)

        # decide landing type
        if result['left_foot_contact'] and result['right_foot_contact']:
            result['landing_detected'] = True
            result['landing_type'] = 'BOTH_FEET'
            result['landing_score'] = 1.0
        elif result['left_foot_contact']:
            result['landing_detected'] = True
            result['landing_type'] = 'LEFT_FOOT'
            result['landing_score'] = 0.8
        elif result['right_foot_contact']:
            result['landing_detected'] = True
            result['landing_type'] = 'RIGHT_FOOT'
            result['landing_score'] = 0.8

        # confidence = average ankle confidence
        result['confidence'] = (left_ankle.confidence + right_ankle.confidence) / 2
        return result

    def _is_foot_on_mat(self, ankle: PoseKeypoint, knee: PoseKeypoint) -> bool:
        """Check if foot is on mat based on ankle and knee position."""
        # simple test: ankle below mat threshold
        if ankle.y > self.mat_threshold:
            return True
        # or ankle below knee and near mat
        if knee.confidence > self.min_confidence and ankle.y > knee.y:
            return True
        return False


# ==================== Module 7: Jump Event Analyzer (Updated with full event fields) ====================
class JumpEventAnalyzer:
    """Analyze complete jump sequence with multi-frame validation"""

    def __init__(self, model_name: str = "yolo11m-pose.pt"):
        self.pose_estimator = YOLO11PoseEstimator(model_name)
        self.takeoff_detector = TakeoffDetector()
        self.landing_detector = LandingDetector()
        # Minimum flight time to avoid confusing running steps with a jump
        self.expected_flight_time = (0.3, 3.0)
        # Require this many consecutive frames to confirm takeoff/landing
        self.buffer_size = 3

    def find_jump_events(self,
                         video_path: Union[str, Path],
                         search_start: float = 0.0,
                         search_end: float = 30.0,
                         conf_thresh: float = 0.3) -> Optional[Dict]:
        """
        Find complete jump sequence in video with buffered detection.
        Returns a dict containing 'takeoff', 'landing', 'flight_time', 'video_name',
        or None if detection failed.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(fps * search_start)
        end_frame = int(fps * search_end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        state = 'SEARCHING_TAKEOFF'
        takeoff_event, landing_event = None, None
        prev_person = None

        # Buffers for multi-frame validation
        takeoff_buffer: List[bool] = []
        landing_buffer: List[bool] = []

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            current_time = frame_idx / fps

            # Estimate poses and identify main subject
            people = self.pose_estimator.estimate_frame_poses(frame, conf_thresh)
            main = self.pose_estimator.main_subject_identifier.identify_main_subject(people)
            if not main:
                # reset buffers if subject lost
                takeoff_buffer.clear()
                landing_buffer.clear()
                prev_person = None
                continue

            # === SEARCHING_TAKEOFF ===
            if state == 'SEARCHING_TAKEOFF':
                res = self.takeoff_detector.detect_takeoff(main, prev_person)
                takeoff_buffer.append(res['takeoff_detected'])
                if len(takeoff_buffer) > self.buffer_size:
                    takeoff_buffer.pop(0)

                # Confirm takeoff only after N consecutive detections
                if len(takeoff_buffer) == self.buffer_size and all(takeoff_buffer):
                    takeoff_event = {
                        'frame': frame_idx,
                        'time': current_time,
                        'score': res['takeoff_score'],
                        'analysis': res,
                        'frame_image': frame.copy()
                    }
                    state = 'IN_FLIGHT'
                    takeoff_buffer.clear()
                    print(f"[INFO] Takeoff detected at {current_time:.3f}s after {self.buffer_size} frames")

            # === IN_FLIGHT ===
            elif state == 'IN_FLIGHT':
                flight_time = current_time - takeoff_event['time']
                # only start landing detection after minimum flight time
                if flight_time >= self.expected_flight_time[0]:
                    res = self.landing_detector.detect_landing(main, prev_person)
                    landing_buffer.append(res['landing_detected'])
                    if len(landing_buffer) > self.buffer_size:
                        landing_buffer.pop(0)

                    # Confirm landing only after N consecutive detections
                    if len(landing_buffer) == self.buffer_size and all(landing_buffer):
                        landing_event = {
                            'frame': frame_idx,
                            'time': current_time,
                            'score': res['landing_score'],
                            'landing_type': res['landing_type'],
                            'analysis': res,
                            'frame_image': frame.copy()
                        }
                        state = 'COMPLETE'
                        print(f"[INFO] Landing detected at {current_time:.3f}s after {self.buffer_size} frames")
                        break

            prev_person = main

        cap.release()

        # Return result if both events found
        if takeoff_event and landing_event:
            flight_duration = landing_event['time'] - takeoff_event['time']
            return {
                'takeoff': takeoff_event,
                'landing': landing_event,
                'flight_time': flight_duration,
                'video_name': Path(video_path).name
            }
        else:
            print(f"[WARN] Incomplete jump sequence in {Path(video_path).name}")
            return None




# ==================== Module 8: Multi-Camera Jump Synchronizer ====================

class MultiCameraJumpSynchronizer:
    """Synchronize multiple cameras using jump events"""
    
    def __init__(self, model_name: str = "yolo11m-pose.pt"):
        self.jump_analyzer = JumpEventAnalyzer(model_name)
    
    def sync_cameras_by_landing(self, 
                               video_paths: List[Union[str, Path]],
                               search_start: float = 0.0,
                               search_end: float = 30.0) -> Dict:
        """
        Synchronize cameras using landing as key frame
        
        Args:
            video_paths: List of video file paths
            search_start: Search start time
            search_end: Search end time
            
        Returns:
            Synchronization results
        """
        print(f"[INFO] Multi-camera synchronization using jump events")
        print(f"       Videos: {len(video_paths)}")
        
        jump_results = {}
        
        # Step 1: Find jump events in each video
        print(f"\n{'='*60}")
        print("STEP 1: DETECTING JUMP EVENTS")
        print(f"{'='*60}")
        
        for video_path in video_paths:
            video_name = Path(video_path).name
            jump_event = self.jump_analyzer.find_jump_events(
                video_path, search_start, search_end
            )
            
            if jump_event:
                jump_results[video_name] = jump_event
        
        if len(jump_results) < 2:
            print("[ERROR] Need at least 2 videos with detected jump events")
            return {'error': 'Insufficient jump detections'}
        
        # Step 2: Calculate synchronization offsets based on landing
        print(f"\n{'='*60}")
        print("STEP 2: SYNCHRONIZATION ANALYSIS (LANDING-BASED)")
        print(f"{'='*60}")
        
        video_names = list(jump_results.keys())
        base_video = video_names[0]
        base_landing_time = jump_results[base_video]['landing']['time']
        
        sync_offsets = {}
        
        print(f"Base reference: {base_video}")
        print(f"Base landing time: {base_landing_time:.3f}s")
        print(f"\n{'Video':<20} | {'Takeoff':<10} | {'Landing':<10} | {'Flight':<8} | {'Offset':<8} | {'Type':<12}")
        print("-" * 80)
        
        for video_name, result in jump_results.items():
            offset = result['landing']['time'] - base_landing_time
            sync_offsets[video_name] = offset
            
            print(f"{video_name:<20} | "
                  f"{result['takeoff']['time']:>7.3f}s | "
                  f"{result['landing']['time']:>7.3f}s | "
                  f"{result['flight_time']:>5.3f}s | "
                  f"{offset:>+7.3f}s | "
                  f"{result['landing']['landing_type']:<12}")
        
        max_offset = max(abs(offset) for offset in sync_offsets.values())
        print(f"\nMaximum offset: {max_offset:.3f} seconds")
        
        # Quality assessment
        if max_offset < 0.05:
            quality = "Excellent"
        elif max_offset < 0.1:
            quality = "Good"
        elif max_offset < 0.2:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        print(f"Sync quality: {quality}")
        
        return {
            'jump_results': jump_results,
            'sync_offsets': sync_offsets,
            'max_offset': max_offset,
            'sync_quality': quality,
            'base_video': base_video
        }

# ==================== Module 9: Manual Verification UI ====================

class ManualVerificationUI:
    """Manual verification interface for key frames"""
    
    @staticmethod
    def create_verification_report(sync_results: Dict, output_dir: Union[str, Path]):
        """
        Create visual verification report with key frames
        
        Args:
            sync_results: Synchronization analysis results
            output_dir: Output directory for report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots for each video
        num_videos = len(sync_results['jump_results'])
        fig, axes = plt.subplots(num_videos, 2, figsize=(16, 6 * num_videos))
        
        if num_videos == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (video_name, jump_result) in enumerate(sync_results['jump_results'].items()):
            # Takeoff frame
            ax_takeoff = axes[idx, 0]
            takeoff_frame = jump_result['takeoff']['frame_image']
            takeoff_frame_rgb = cv2.cvtColor(takeoff_frame, cv2.COLOR_BGR2RGB)
            ax_takeoff.imshow(takeoff_frame_rgb)
            ax_takeoff.set_title(f"{video_name} - Takeoff\n"
                               f"Time: {jump_result['takeoff']['time']:.3f}s, "
                               f"Frame: {jump_result['takeoff']['frame']}")
            ax_takeoff.axis('off')
            
            # Landing frame
            ax_landing = axes[idx, 1]
            landing_frame = jump_result['landing']['frame_image']
            landing_frame_rgb = cv2.cvtColor(landing_frame, cv2.COLOR_BGR2RGB)
            ax_landing.imshow(landing_frame_rgb)
            ax_landing.set_title(f"{video_name} - Landing ({jump_result['landing']['landing_type']})\n"
                               f"Time: {jump_result['landing']['time']:.3f}s, "
                               f"Frame: {jump_result['landing']['frame']}, "
                               f"Offset: {sync_results['sync_offsets'][video_name]:+.3f}s")
            ax_landing.axis('off')
        
        plt.tight_layout()
        
        # Save report
        report_path = output_dir / "key_frames_verification.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Verification report saved to: {report_path}")
        
        # Also save individual frames
        frames_dir = output_dir / "individual_frames"
        frames_dir.mkdir(exist_ok=True)
        
        for video_name, jump_result in sync_results['jump_results'].items():
            # Save takeoff frame
            takeoff_path = frames_dir / f"{video_name}_takeoff_f{jump_result['takeoff']['frame']}.jpg"
            cv2.imwrite(str(takeoff_path), jump_result['takeoff']['frame_image'])
            
            # Save landing frame
            landing_path = frames_dir / f"{video_name}_landing_f{jump_result['landing']['frame']}.jpg"
            cv2.imwrite(str(landing_path), jump_result['landing']['frame_image'])
        
        print(f"[INFO] Individual frames saved to: {frames_dir}")
    
    @staticmethod
    def create_sync_timeline(sync_results: Dict, output_path: Union[str, Path]):
        """
        Create timeline visualization of synchronization
        
        Args:
            sync_results: Synchronization results
            output_path: Output file path
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        video_names = list(sync_results['jump_results'].keys())
        base_video = sync_results['base_video']
        
        # Set up timeline
        y_positions = {name: i for i, name in enumerate(video_names)}
        colors = plt.cm.Set3(np.linspace(0, 1, len(video_names)))
        
        for idx, (video_name, jump_result) in enumerate(sync_results['jump_results'].items()):
            y_pos = y_positions[video_name]
            color = colors[idx]
            
            # Get times
            takeoff_time = jump_result['takeoff']['time']
            landing_time = jump_result['landing']['time']
            offset = sync_results['sync_offsets'][video_name]
            
            # Adjusted times (relative to base)
            adjusted_takeoff = takeoff_time - offset
            adjusted_landing = landing_time - offset
            
            # Draw flight duration bar
            flight_duration = landing_time - takeoff_time
            rect = Rectangle((takeoff_time, y_pos - 0.2), flight_duration, 0.4,
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Mark takeoff and landing
            ax.scatter(takeoff_time, y_pos, s=100, c='green', marker='^', 
                      edgecolors='black', linewidth=2, zorder=5)
            ax.scatter(landing_time, y_pos, s=100, c='red', marker='v', 
                      edgecolors='black', linewidth=2, zorder=5)
            
            # Add labels
            ax.text(takeoff_time - 0.5, y_pos, f"T: {takeoff_time:.2f}s", 
                   fontsize=9, ha='right', va='center')
            ax.text(landing_time + 0.5, y_pos, f"L: {landing_time:.2f}s", 
                   fontsize=9, ha='left', va='center')
            
            # Add offset label
            offset_text = f"Offset: {offset:+.3f}s" if video_name != base_video else "BASE"
            ax.text(landing_time, y_pos + 0.3, offset_text, 
                   fontsize=8, ha='center', va='bottom', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # Customize plot
        ax.set_yticks(range(len(video_names)))
        ax.set_yticklabels(video_names)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_title('Multi-Camera Jump Event Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='g', 
                  markersize=10, label='Takeoff', markeredgecolor='black'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='r', 
                  markersize=10, label='Landing', markeredgecolor='black'),
            patches.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.7, 
                            edgecolor='black', label='Flight Duration')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set x-axis limits with padding
        all_times = []
        for jump_result in sync_results['jump_results'].values():
            all_times.extend([jump_result['takeoff']['time'], jump_result['landing']['time']])
        ax.set_xlim(min(all_times) - 1, max(all_times) + 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Timeline visualization saved to: {output_path}")

# ==================== Module 10: Result Export and Management ====================

class EnhancedResultExporter:
    """Export synchronization results with jump event data"""
    
    @staticmethod
    def save_sync_results(sync_results: Dict, output_dir: Union[str, Path]):
        """Save complete synchronization analysis results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main sync analysis CSV
        csv_path = output_dir / "jump_sync_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video', 'takeoff_frame', 'takeoff_time', 'landing_frame', 
                           'landing_time', 'flight_time', 'sync_offset_sec', 
                           'landing_type', 'takeoff_score', 'landing_score'])
            
            for video_name, result in sync_results['jump_results'].items():
                offset = sync_results['sync_offsets'][video_name]
                writer.writerow([
                    video_name, 
                    result['takeoff']['frame'],
                    result['takeoff']['time'],
                    result['landing']['frame'],
                    result['landing']['time'],
                    result['flight_time'],
                    offset,
                    result['landing']['landing_type'],
                    result['takeoff']['score'],
                    result['landing']['score']
                ])
        
        # Save detailed JSON data
        json_path = output_dir / "detailed_jump_analysis.json"
        detailed_data = {}
        
        for video_name, result in sync_results['jump_results'].items():
            detailed_data[video_name] = {
                'takeoff_event': {
                    'frame': result['takeoff']['frame'],
                    'time': result['takeoff']['time'],
                    'score': result['takeoff']['score'],
                    'analysis': result['takeoff']['analysis']
                },
                'landing_event': {
                    'frame': result['landing']['frame'],
                    'time': result['landing']['time'],
                    'score': result['landing']['score'],
                    'landing_type': result['landing']['landing_type'],
                    'analysis': result['landing']['analysis']
                },
                'flight_time': result['flight_time'],
                'sync_offset': sync_results['sync_offsets'][video_name]
            }
        
        # Remove frame images from JSON (too large)
        for video_data in detailed_data.values():
            if 'frame_image' in video_data.get('takeoff_event', {}):
                del video_data['takeoff_event']['frame_image']
            if 'frame_image' in video_data.get('landing_event', {}):
                del video_data['landing_event']['frame_image']
        
        with open(json_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Save sync summary
        summary_path = output_dir / "sync_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Multi-Camera Jump Synchronization Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Base Video: {sync_results['base_video']}\n")
            f.write(f"Maximum Offset: {sync_results['max_offset']:.3f} seconds\n")
            f.write(f"Sync Quality: {sync_results['sync_quality']}\n\n")
            
            f.write("Video Offsets (relative to base):\n")
            for video_name, offset in sync_results['sync_offsets'].items():
                f.write(f"  {video_name}: {offset:+.3f}s\n")
        
        print(f"[INFO] Results saved to {output_dir}:")
        print(f"       - {csv_path.name}: Sync analysis summary")
        print(f"       - {json_path.name}: Detailed jump data")
        print(f"       - {summary_path.name}: Human-readable summary")

# ==================== Main System Interface ====================

def sync_cameras_by_jump_events(video_dir: str = "run13", 
                               model_name: str = "yolo11m-pose.pt",
                               trim_start_time: float = 70.0,
                               search_start: float = 0.0,
                               search_end: float = 30.0,
                               create_verification_report: bool = True):
    """
    Main function for camera synchronization using jump events
    
    Args:
        video_dir: Directory containing video files
        model_name: YOLO model name (yolo11m-pose.pt recommended)
        trim_start_time: Time to trim from start (remove first N seconds)
        search_start: Start time for jump search in trimmed videos
        search_end: End time for jump search in trimmed videos
        create_verification_report: Whether to create visual verification report
        
    Returns:
        Complete sync analysis results
    """
    video_dir = Path(video_dir)
    
    # Find video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        print(f"[ERROR] No video files found in {video_dir}")
        return None
    
    print(f"[INFO] Found {len(video_files)} videos: {[v.name for v in video_files]}")
    
    # Step 1: Trim videos (remove first 70 seconds)
    print(f"\n{'='*60}")
    print("STEP 1: VIDEO PREPROCESSING (TRIMMING)")
    print(f"{'='*60}")
    
    trimmed_dir = video_dir / "trimmed_videos"
    trimmed_mapping = VideoTrimmer.batch_trim_videos(
        video_dir, trimmed_dir, start_time=trim_start_time 
    )
    
    if not trimmed_mapping:
        print("[ERROR] Video trimming failed")
        return None
    
    # Step 2: Multi-camera synchronization using jump events
    print(f"\n{'='*60}")
    print("STEP 2: JUMP EVENT DETECTION AND SYNCHRONIZATION")
    print(f"{'='*60}")
    
    synchronizer = MultiCameraJumpSynchronizer(model_name)
    trimmed_videos = list(trimmed_mapping.values())
    
    sync_results = synchronizer.sync_cameras_by_landing(
        trimmed_videos, search_start, search_end
    )
    
    if 'error' in sync_results:
        return sync_results
    
    # Step 3: Save results
    results_dir = video_dir / "jump_sync_results"
    EnhancedResultExporter.save_sync_results(sync_results, results_dir)
    
    # Step 4: Create verification report
    if create_verification_report:
        print(f"\n{'='*60}")
        print("STEP 3: CREATING VERIFICATION REPORT")
        print(f"{'='*60}")
        
        ManualVerificationUI.create_verification_report(sync_results, results_dir)
        ManualVerificationUI.create_sync_timeline(
            sync_results, results_dir / "sync_timeline.png"
        )
    
    print(f"\n[INFO] Analysis complete!")
    print(f"       Max camera desync: {sync_results['max_offset']:.3f} seconds")
    print(f"       Sync quality: {sync_results['sync_quality']}")
    
    return {
        **sync_results,
        'trimmed_videos': trimmed_mapping,
        'results_directory': str(results_dir)
    }

# ==================== Main Execution ====================

if __name__ == "__main__":
    print("Enhanced High Jump Video Synchronization System")
    print("Using takeoff and landing detection for precise sync")
    print("=" * 60)
    
    # Run synchronization with default parameters
    results = sync_cameras_by_jump_events(
        video_dir="run13",
        model_name="yolo11m-pose.pt",
        trim_start_time=70.0,
        search_start=0.0,
        search_end=30.0,
        create_verification_report=True
    )
    
    if results and 'error' not in results:
        print(f"\nSynchronization completed successfully!")
        print(f"Results and verification report saved to: {results['results_directory']}")