#!/usr/bin/env python3
# refactored_jump_sync_system.py
# Enhanced version based on the original foot-ground contact sync detection

import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

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
    
    def get_raw_prediction_data(self, frame: np.ndarray, conf_thresh: float = 0.3) -> Dict:
        """
        Get raw YOLO prediction data (similar to official website example output)
        
        Returns:
            dict: Contains xy coordinates, normalized coordinates, visibility etc.
        """
        results = self.model(frame, conf=conf_thresh, verbose=False)
        
        raw_data = {
            'detected_persons': 0,
            'keypoints_data': []
        }
        
        if results[0].keypoints is not None:
            raw_data['detected_persons'] = len(results[0].keypoints)
            
            for person_idx, keypoints in enumerate(results[0].keypoints):
                person_data = {
                    'person_id': person_idx,
                    'xy': keypoints.xy[0].cpu().numpy().tolist() if len(keypoints.xy) > 0 else [],  # x,y coordinates
                    'xyn': keypoints.xyn[0].cpu().numpy().tolist() if len(keypoints.xyn) > 0 else [],  # normalized coordinates  
                    'conf': keypoints.conf[0].cpu().numpy().tolist() if keypoints.conf is not None else [],  # confidence
                    'data': keypoints.data[0].cpu().numpy().tolist()  # raw data [x,y,visibility]
                }
                raw_data['keypoints_data'].append(person_data)
        
        return raw_data

# ==================== Module 5: Ground Contact Detector ====================

class GroundContactDetector:
    """Foot-ground contact detection based on ankle position and dynamics"""
    
    def __init__(self, min_area_ratio: float = 0.02, contact_threshold: float = 0.6):
        """
        Args:
            min_area_ratio: Minimum pose area ratio (similar to original parameter)
            contact_threshold: Contact detection threshold (similar to original parameter)
        """
        self.min_area_ratio = min_area_ratio
        self.contact_threshold = contact_threshold
    
    def detect_ground_contact(self, person: PersonPose, frame_area: float = 1.0) -> Dict:
        """
        Detect foot-ground contact based on pose analysis
        
        Args:
            person: Person pose data
            frame_area: Frame area for normalization
            
        Returns:
            Contact analysis results
        """
        if not person.keypoints or len(person.keypoints) < 17:
            return {'contact_score': 0.0, 'contact_detected': False}
        
        # Get key ankle and leg keypoints (using same indices as original)
        left_ankle = person.keypoints[15]   # left ankle
        right_ankle = person.keypoints[16]  # right ankle  
        left_knee = person.keypoints[13]    # left knee
        right_knee = person.keypoints[14]   # right knee
        left_hip = person.keypoints[11]     # left hip
        right_hip = person.keypoints[12]    # right hip
        
        analysis = {
            'contact_score': 0.0,
            'contact_detected': False,
            'contact_type': 'NO_CONTACT',
            'left_foot_score': 0.0,
            'right_foot_score': 0.0,
            'contact_confidence': 0.0,
            'visible_keypoints': sum(1 for kp in person.keypoints if kp.confidence > 0.5)
        }
        
        # Analyze left and right foot contact
        left_score = self._analyze_foot_contact(left_ankle, left_knee, left_hip)
        right_score = self._analyze_foot_contact(right_ankle, right_knee, right_hip)
        
        analysis['left_foot_score'] = left_score
        analysis['right_foot_score'] = right_score
        
        # Determine contact type (similar to original logic)
        left_contact = left_score >= self.contact_threshold
        right_contact = right_score >= self.contact_threshold
        
        if left_contact and right_contact:
            analysis['contact_type'] = 'BOTH_FEET'
            analysis['contact_score'] = max(left_score, right_score) * 0.9
        elif left_contact:
            analysis['contact_type'] = 'LEFT_FOOT'
            analysis['contact_score'] = left_score
        elif right_contact:
            analysis['contact_type'] = 'RIGHT_FOOT'
            analysis['contact_score'] = right_score
        else:
            analysis['contact_type'] = 'NO_CONTACT'
            analysis['contact_score'] = max(left_score, right_score) * 0.3
        
        analysis['contact_detected'] = analysis['contact_score'] >= self.contact_threshold
        
        # Calculate contact confidence
        ankle_confidences = []
        if left_ankle.confidence > 0.5:
            ankle_confidences.append(left_ankle.confidence)
        if right_ankle.confidence > 0.5:
            ankle_confidences.append(right_ankle.confidence)
        
        analysis['contact_confidence'] = np.mean(ankle_confidences) if ankle_confidences else 0.0
        
        return analysis
    
    def _analyze_foot_contact(self, ankle: PoseKeypoint, knee: PoseKeypoint, 
                             hip: PoseKeypoint) -> float:
        """
        Analyze single foot contact score
        Enhanced to handle soft mat contact (as per Kaze's suggestion)
        """
        if ankle.confidence < 0.5:
            return 0.0
        
        score = 0.0
        
        # 1. Ankle height analysis (primary indicator)
        # Closer to bottom of frame = more likely ground contact
        ankle_height_ratio = 1.0 - ankle.y  # Distance from bottom
        
        if ankle_height_ratio < 0.1:  # Very close to bottom
            score += 0.6
        elif ankle_height_ratio < 0.2:  # Close to bottom
            score += 0.4
        elif ankle_height_ratio < 0.3:  # Moderately close
            score += 0.2
        
        # 2. Relative position analysis
        # Ankle should be below knee
        if knee.confidence > 0.5 and ankle.y > knee.y:
            score += 0.2
        
        # Ankle should be below hip
        if hip.confidence > 0.5 and ankle.y > hip.y:
            score += 0.1
        
        # 3. Soft mat contact handling (Kaze's enhancement)
        # Account for mat compression - allow for "sinking" effect
        if ankle_height_ratio < 0.3:  # Within bottom 30% of frame
            score += 0.1  # Soft mat bonus
        
        return min(score, 1.0)

# ==================== Module 6: Multi-Camera Sync Analyzer ====================

class MultiCameraSyncAnalyzer:
    """Multi-camera synchronization analyzer using ground contact detection"""
    
    def __init__(self, model_name: str = "yolo11m-pose.pt"):
        self.pose_estimator = YOLO11PoseEstimator(model_name)
        self.contact_detector = GroundContactDetector()
    
    def find_ground_contact_moments(self, 
                                  video_path: Union[str, Path],
                                  search_start: float = 0.0,
                                  search_end: float = 30.0,
                                  conf_thresh: float = 0.3) -> Optional[Dict]:
        """
        Find ground contact moments in video (similar to original find_takeoff_moment)
        
        Args:
            video_path: Path to video file
            search_start: Search start time in seconds
            search_end: Search end time in seconds  
            conf_thresh: YOLO confidence threshold
            
        Returns:
            Best ground contact moment data
        """
        video_path = Path(video_path)
        print(f"[INFO] Analyzing {video_path.name} for ground contact")
        print(f"       Search window: {search_start:.1f}s - {search_end:.1f}s")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(fps * search_start)
        end_frame = int(fps * search_end)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        best_contact = None
        best_score = 0.0
        candidates = []
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_idx / fps
            
            # Pose estimation
            people = self.pose_estimator.estimate_frame_poses(frame, conf_thresh)
            
            # Identify main subject (Kaze)
            main_subject = self.pose_estimator.main_subject_identifier.identify_main_subject(people)
            
            if main_subject:
                # Get raw prediction data
                raw_prediction = self.pose_estimator.get_raw_prediction_data(frame, conf_thresh)
                
                # Ground contact detection
                contact_analysis = self.contact_detector.detect_ground_contact(main_subject)
                
                if contact_analysis['contact_detected']:
                    contact_moment = {
                        'frame': frame_idx,
                        'time': current_time,
                        'contact_score': contact_analysis['contact_score'],
                        'contact_type': contact_analysis['contact_type'],
                        'contact_confidence': contact_analysis['contact_confidence'],
                        'analysis': contact_analysis,
                        'raw_prediction': raw_prediction
                    }
                    candidates.append(contact_moment)
                    
                    if contact_analysis['contact_score'] > best_score:
                        best_score = contact_analysis['contact_score']
                        best_contact = contact_moment
            
            # Progress report every 2 seconds
            if (frame_idx - start_frame) % int(fps * 2) == 0:
                print(f"       t={current_time:.1f}s, best_score={best_score:.3f}")
        
        cap.release()
        
        if best_contact:
            print(f"[INFO] Ground contact detected:")
            print(f"       Time: {best_contact['time']:.3f}s (frame {best_contact['frame']})")
            print(f"       Type: {best_contact['contact_type']}")
            print(f"       Score: {best_contact['contact_score']:.3f}")
            print(f"       Candidates: {len(candidates)}")
        else:
            print(f"[WARN] No ground contact detected in {video_path.name}")
        
        return best_contact
    
    def sync_multiple_cameras(self, 
                            video_paths: List[Union[str, Path]],
                            search_start: float = 0.0,
                            search_end: float = 30.0,
                            time_series_window: float = 3.0) -> Dict:
        """
        Synchronize multiple cameras using ground contact detection
        
        Args:
            video_paths: List of video file paths
            search_start: Search start time
            search_end: Search end time
            time_series_window: Time window for series extraction (±seconds)
            
        Returns:
            Sync analysis results
        """
        print(f"[INFO] Multi-camera sync analysis starting")
        print(f"       Videos: {len(video_paths)}")
        
        contact_results = {}
        
        # Step 1: Find ground contact moments in each video
        print(f"\n{'='*60}")
        print("STEP 1: DETECTING GROUND CONTACT MOMENTS")
        print(f"{'='*60}")
        
        for video_path in video_paths:
            video_name = Path(video_path).name
            contact_moment = self.find_ground_contact_moments(
                video_path, search_start, search_end
            )
            
            if contact_moment:
                contact_results[video_name] = contact_moment
        
        if len(contact_results) < 2:
            print("[ERROR] Need at least 2 videos with detected ground contact")
            return {'error': 'Insufficient ground contact detections'}
        
        # Step 2: Calculate synchronization offsets
        print(f"\n{'='*60}")
        print("STEP 2: SYNCHRONIZATION ANALYSIS")
        print(f"{'='*60}")
        
        video_names = list(contact_results.keys())
        base_video = video_names[0]
        base_time = contact_results[base_video]['time']
        
        sync_offsets = {}
        
        print(f"Base reference: {base_video} at {base_time:.3f}s")
        print(f"{'Video':<20} | {'Contact Time':<12} | {'Offset':<8} | {'Type':<12} | {'Score':<6}")
        print("-" * 70)
        
        for video_name, result in contact_results.items():
            offset = result['time'] - base_time
            sync_offsets[video_name] = offset
            print(f"{video_name:<20} | {result['time']:>9.3f}s | {offset:>+7.3f}s | "
                  f"{result['contact_type']:<12} | {result['contact_score']:<6.3f}")
        
        max_offset = max(abs(offset) for offset in sync_offsets.values())
        print(f"\nMaximum offset: {max_offset:.3f} seconds")
        
        # Quality assessment
        if max_offset < 0.1:
            quality = "Excellent"
        elif max_offset < 0.5:
            quality = "Good"
        elif max_offset < 1.0:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        print(f"Sync quality: {quality}")
        
        return {
            'contact_results': contact_results,
            'sync_offsets': sync_offsets,
            'max_offset': max_offset,
            'sync_quality': quality,
            'base_video': base_video,
            'time_series_window': time_series_window
        }

# ==================== Module 7: Result Export and Management ====================

class SyncResultExporter:
    """Export and manage synchronization analysis results"""
    
    @staticmethod
    def save_sync_results(sync_results: Dict, output_dir: Union[str, Path]):
        """Save synchronization analysis results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main sync analysis CSV
        csv_path = output_dir / "ground_contact_sync_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video', 'contact_frame', 'contact_time', 'sync_offset_sec',
                           'contact_score', 'contact_type', 'contact_confidence'])
            
            for video_name, result in sync_results['contact_results'].items():
                offset = sync_results['sync_offsets'][video_name]
                writer.writerow([
                    video_name, result['frame'], result['time'], offset,
                    result['contact_score'], result['contact_type'], result['contact_confidence']
                ])
        
        # Save detailed pose data JSON
        json_path = output_dir / "detailed_pose_analysis.json"
        detailed_data = {}
        
        for video_name, result in sync_results['contact_results'].items():
            detailed_data[video_name] = {
                'contact_event': {
                    'frame': result['frame'],
                    'time': result['time'],
                    'contact_score': result['contact_score'],
                    'contact_type': result['contact_type']
                },
                'raw_yolo_prediction': result['raw_prediction']
            }
        
        with open(json_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        print(f"[INFO] Results saved to {output_dir}:")
        print(f"       - {csv_path.name}: Sync analysis summary")
        print(f"       - {json_path.name}: Detailed pose data")

# ==================== Main System Interface ====================

def sync_cameras_by_ground_contact(video_dir: str = "run13", 
                                  model_name: str = "yolo11m-pose.pt",
                                  trim_start_time: float = 70.0,
                                  search_start: float = 0.0,
                                  search_end: float = 30.0):
    """
    Main function for camera synchronization using ground contact detection
    Enhanced version of the original sync_cameras_by_ground_contact function
    
    Args:
        video_dir: Directory containing video files
        model_name: YOLO model name (yolo11m-pose.pt recommended)
        trim_start_time: Time to trim from start (remove first N seconds)
        search_start: Start time for contact search in trimmed videos
        search_end: End time for contact search in trimmed videos
        
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
   
    # Step 2: Multi-camera synchronization analysis
    print(f"\n{'='*60}")
    print("STEP 2: MULTI-CAMERA SYNCHRONIZATION")
    print(f"{'='*60}")
   
    analyzer = MultiCameraSyncAnalyzer(model_name)
    trimmed_videos = list(trimmed_mapping.values())
   
    sync_results = analyzer.sync_multiple_cameras(

        trimmed_videos, search_start, search_end
    )
   
    if 'error' in sync_results:
        return sync_results
   
    # Step 3: Save results
    results_dir = video_dir / "sync_analysis_results"
    SyncResultExporter.save_sync_results(sync_results, results_dir)
   
    print(f"\n[INFO] Analysis complete!")
    print(f"       Max camera desync: {sync_results['max_offset']:.3f} seconds")
    print(f"       Sync quality: {sync_results['sync_quality']}")
   
    return {
        **sync_results,
        'trimmed_videos': trimmed_mapping,
        'results_directory': str(results_dir)
    }

# ==================== Standalone Utility Functions ====================

def trim_single_video_standalone(input_path: str, output_path: str, 
                               start_time: float = 70.0, end_time: Optional[float] = None):
   """
   Standalone video trimming function - can be used independently
   
   Args:
       input_path: Input video file path
       output_path: Output video file path
       start_time: Start time in seconds (default 70.0 to remove first 70s)
       end_time: End time in seconds, None for full video
       
   Returns:
       Success status
   """
   return VideoTrimmer.trim_video(input_path, output_path, start_time, end_time)

def analyze_single_video_pose_standalone(video_path: str, output_path: str, 
                                       model_name: str = "yolo11m-pose.pt"):
   """
   Standalone pose analysis function - can be used independently
   
   Args:
       video_path: Input video file path
       output_path: Output JSON file path for results
       model_name: YOLO model name
       
   Returns:
       Pose analysis data
   """
   estimator = YOLO11PoseEstimator(model_name)
   
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print(f"[ERROR] Cannot open video: {video_path}")
       return None
   
   fps = cap.get(cv2.CAP_PROP_FPS)
   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
   all_frames_data = []
   frame_idx = 0
   
   print(f"[INFO] Analyzing pose in video: {Path(video_path).name}")
   print(f"       Total frames: {total_frames}, FPS: {fps:.2f}")
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       current_time = frame_idx / fps
       
       # Pose estimation
       people = estimator.estimate_frame_poses(frame)
       
       # Get raw YOLO prediction data (similar to official website output)
       raw_prediction = estimator.get_raw_prediction_data(frame)
       
       # Identify main subject
       main_subject = estimator.main_subject_identifier.identify_main_subject(people)
       
       frame_data = {
           'frame_idx': frame_idx,
           'timestamp': current_time,
           'detected_people': len(people),
           'main_subject_detected': main_subject is not None,
           'raw_yolo_output': raw_prediction,
           'processed_poses': []
       }
       
       for person_idx, person in enumerate(people):
           person_data = {
               'person_id': person_idx,
               'is_main_subject': person.is_main_subject,
               'person_confidence': person.person_confidence,
               'visible_keypoints': len([kp for kp in person.keypoints if kp.confidence > 0.5]),
               'keypoints': [
                   {
                       'name': kp.name,
                       'x': kp.x,
                       'y': kp.y,
                       'confidence': kp.confidence
                   } for kp in person.keypoints
               ]
           }
           frame_data['processed_poses'].append(person_data)
       
       all_frames_data.append(frame_data)
       frame_idx += 1
       
       # Progress report every 5 seconds
       if frame_idx % int(fps * 5) == 0:
           print(f"       Processed: {current_time:.1f}s")
   
   cap.release()
   
   # Save results
   with open(output_path, 'w') as f:
       json.dump(all_frames_data, f, indent=2)
   
   print(f"[INFO] Pose analysis complete")
   print(f"       Results saved to: {output_path}")
   print(f"       Total frames analyzed: {len(all_frames_data)}")
   
   return all_frames_data

def extract_time_series_around_event(video_path: str, 
                                  center_time: float,
                                  window_before: float = 3.0,
                                  window_after: float = 3.0,
                                  model_name: str = "yolo11m-pose.pt") -> List[Dict]:
   """
   Extract pose time series around a specific event time
   
   Args:
       video_path: Video file path
       center_time: Center time of interest in seconds
       window_before: Time window before center time
       window_after: Time window after center time
       model_name: YOLO model name
       
   Returns:
       List of frame analysis data in chronological order
   """
   estimator = YOLO11PoseEstimator(model_name)
   contact_detector = GroundContactDetector()
   
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print(f"[ERROR] Cannot open video: {video_path}")
       return []
   
   fps = cap.get(cv2.CAP_PROP_FPS)
   
   start_time = max(0, center_time - window_before)
   end_time = center_time + window_after
   start_frame = int(fps * start_time)
   end_frame = int(fps * end_time)
   
   cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
   
   time_series = []
   
   print(f"[INFO] Extracting time series from {Path(video_path).name}")
   print(f"       Center time: {center_time:.2f}s, Window: {start_time:.2f}s to {end_time:.2f}s")
   
   for frame_idx in range(start_frame, end_frame):
       ret, frame = cap.read()
       if not ret:
           break
       
       current_time = frame_idx / fps
       
       # Pose estimation
       people = estimator.estimate_frame_poses(frame)
       main_subject = estimator.main_subject_identifier.identify_main_subject(people)
       
       frame_data = {
           'frame_idx': frame_idx,
           'timestamp': current_time,
           'time_offset_from_center': current_time - center_time,
           'poses_detected': len(people),
           'main_subject_present': main_subject is not None
       }
       
       if main_subject:
           # Ground contact analysis
           contact_analysis = contact_detector.detect_ground_contact(main_subject)
           frame_data.update({
               'contact_score': contact_analysis['contact_score'],
               'contact_type': contact_analysis['contact_type'],
               'contact_detected': contact_analysis['contact_detected']
           })
       else:
           frame_data.update({
               'contact_score': 0.0,
               'contact_type': 'NO_SUBJECT',
               'contact_detected': False
           })
       
       time_series.append(frame_data)
   
   cap.release()
   
   print(f"[INFO] Time series extraction complete: {len(time_series)} frames")
   return time_series

# ==================== Main Execution Interface ====================

def main():
   """
   Main execution function with command line interface
   Maintains compatibility with original function naming and parameters
   """
   import argparse
   
   parser = argparse.ArgumentParser(description="High Jump Video Synchronization System")
   parser.add_argument("command", choices=["sync", "trim", "pose", "timeseries"], 
                      help="Command to execute")
   parser.add_argument("--video_dir", default="run13", help="Video directory path")
   parser.add_argument("--input", help="Input file path")
   parser.add_argument("--output", help="Output file path")
   parser.add_argument("--trim_start_time", type=float, default=70.0, 
                      help="Trim start time (remove first N seconds)")
   parser.add_argument("--search_start", type=float, default=0.0, 
                      help="Search start time in trimmed videos")
   parser.add_argument("--search_end", type=float, default=30.0, 
                      help="Search end time in trimmed videos")
   parser.add_argument("--model_name", default="yolo11m-pose.pt", 
                      help="YOLO model name")
   parser.add_argument("--center_time", type=float, help="Center time for time series")
   parser.add_argument("--window", type=float, default=3.0, help="Time window size")
   
   args = parser.parse_args()
   
   if args.command == "sync":
       # Full synchronization analysis (main function)
       results = sync_cameras_by_ground_contact(
           video_dir=args.video_dir,
           model_name=args.model_name,
           trim_start_time=args.trim_start_time,
           search_start=args.search_start,
           search_end=args.search_end
       )
       
   elif args.command == "trim":
       # Standalone video trimming
       if not args.input or not args.output:
           print("[ERROR] --input and --output required for trim command")
           return
       
       success = trim_single_video_standalone(
           args.input, args.output, args.trim_start_time
       )
       print(f"[INFO] Trimming {'successful' if success else 'failed'}")
       
   elif args.command == "pose":
       # Standalone pose analysis
       if not args.input or not args.output:
           print("[ERROR] --input and --output required for pose command")
           return
       
       analyze_single_video_pose_standalone(
           args.input, args.output, args.model_name
       )
       
   elif args.command == "timeseries":
       # Time series extraction
       if not args.input or not args.center_time:
           print("[ERROR] --input and --center_time required for timeseries command")
           return
       
       time_series = extract_time_series_around_event(
           args.input, args.center_time, args.window, args.window, args.model_name
       )
       
       if args.output:
           with open(args.output, 'w') as f:
               json.dump(time_series, f, indent=2)
           print(f"[INFO] Time series saved to {args.output}")

if __name__ == "__main__":
   # Default execution - maintain compatibility with original usage
   print("High Jump Video Synchronization System")
   print("Enhanced version based on foot-ground contact detection")
   print("=" * 60)
   
   # Example usage similar to original
   results = sync_cameras_by_ground_contact(
       video_dir="run13",
       model_name="yolo11m-pose.pt",
       trim_start_time=70.0,
       search_start=0.0,
       search_end=30.0
   )
   
   if results and 'error' not in results:
       print(f"\nSynchronization analysis completed successfully!")
       print(f"Maximum offset: {results['max_offset']:.3f} seconds")
       print(f"Sync quality: {results['sync_quality']}")
       print(f"Results saved to: {results['results_directory']}")