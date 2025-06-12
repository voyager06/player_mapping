import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from collections import defaultdict, deque

# Initialize YOLO model
model = YOLO("C:/playerdetect/best.pt")  # Replace with your model path

# Open video files
broadcast_cap = cv2.VideoCapture("C:/playerdetect/broadcast.mp4")
tacticam_cap = cv2.VideoCapture("C:/playerdetect/tacticam.mp4")

# Check if videos opened successfully
if not broadcast_cap.isOpened() or not tacticam_cap.isOpened():
    raise ValueError("Could not open one or both video files")

# Get video properties
broadcast_fps = broadcast_cap.get(cv2.CAP_PROP_FPS)
tacticam_fps = tacticam_cap.get(cv2.CAP_PROP_FPS)
broadcast_width = int(broadcast_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
broadcast_height = int(broadcast_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
tacticam_width = int(tacticam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
tacticam_height = int(tacticam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Broadcast: {broadcast_width}x{broadcast_height} @ {broadcast_fps} FPS")
print(f"Tacticam: {tacticam_width}x{tacticam_height} @ {tacticam_fps} FPS")

# Create output directory for visualizations
os.makedirs("C:/playerdetect/output", exist_ok=True)

class PlayerTracker:
    """Enhanced player tracking across frames"""
    def __init__(self, max_age=10, distance_threshold=100):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.distance_threshold = distance_threshold
        self.frame_count = 0
    
    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_count += 1
        
        if not self.tracks:
            # Initialize tracks
            for det in detections:
                self.tracks[self.next_id] = {
                    'centroid': det['centroid'],
                    'features': det,
                    'age': 0,
                    'last_seen': self.frame_count
                }
                self.next_id += 1
            return list(self.tracks.keys())
        
        # Match existing tracks with new detections
        if not detections:
            # Age all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        # Calculate cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track_centroid = self.tracks[track_id]['centroid']
            for j, det in enumerate(detections):
                distance = np.linalg.norm(np.array(track_centroid) - np.array(det['centroid']))
                cost_matrix[i, j] = distance
        
        # Hungarian matching
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_tracks = []
            matched_detections = set()
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.distance_threshold:
                    track_id = track_ids[i]
                    self.tracks[track_id]['centroid'] = detections[j]['centroid']
                    self.tracks[track_id]['features'] = detections[j]
                    self.tracks[track_id]['age'] = 0
                    self.tracks[track_id]['last_seen'] = self.frame_count
                    matched_tracks.append(track_id)
                    matched_detections.add(j)
            
            # Create new tracks for unmatched detections
            for j, det in enumerate(detections):
                if j not in matched_detections:
                    self.tracks[self.next_id] = {
                        'centroid': det['centroid'],
                        'features': det,
                        'age': 0,
                        'last_seen': self.frame_count
                    }
                    matched_tracks.append(self.next_id)
                    self.next_id += 1
            
            # Age unmatched tracks
            for track_id in track_ids:
                if track_id not in [track_ids[i] for i in row_ind if cost_matrix[i, col_ind[list(row_ind).index(i)]] < self.distance_threshold]:
                    self.tracks[track_id]['age'] += 1
                    if self.tracks[track_id]['age'] > self.max_age:
                        del self.tracks[track_id]
            
            return matched_tracks
        
        return []

def get_adaptive_threshold(detections, base_threshold=0.3):
    """Calculate adaptive confidence threshold"""
    confidences = []
    for result in detections:
        for box in result.boxes:
            confidences.append(box.conf.item())
    
    if len(confidences) > 0:
        return max(base_threshold, np.percentile(confidences, 25))
    return base_threshold

def get_dominant_colors(image, k=3):
    """Extract dominant colors using K-means clustering"""
    if image.size == 0:
        return np.zeros((k, 3))
    
    # Reshape image to be a list of pixels
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    return centers.astype(np.uint8)

def extract_texture_features(gray_patch):
    """Extract texture features using Local Binary Pattern"""
    if gray_patch.size == 0:
        return np.zeros(10)
    
    # Calculate LBP
    lbp = local_binary_pattern(gray_patch, 8, 1, method='uniform')
    
    # Calculate histogram
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    
    # Normalize
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist

def get_enhanced_player_features(frame, results, frame_width, frame_height):
    """Extract comprehensive features for detected players"""
    features = []
    threshold = get_adaptive_threshold(results)
    
    for result in results:
        for box in result.boxes:
            if box.conf.item() < threshold:
                continue
            cls = int(box.cls.item())
            if cls not in [1, 2]:  # Only goalkeepers (1) and players (2)
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract player patch
            player_patch = frame[y1:y2, x1:x2]
            if player_patch.size == 0:
                continue
            
            # 1. Spatial features
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            width, height = x2 - x1, y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            
            # Normalize spatial features by frame dimensions
            norm_centroid = (centroid[0] / frame_width, centroid[1] / frame_height)
            norm_width = width / frame_width
            norm_height = height / frame_height
            norm_area = area / (frame_width * frame_height)
            
            # 2. Enhanced color features
            try:
                hsv = cv2.cvtColor(player_patch, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(player_patch, cv2.COLOR_BGR2LAB)
                
                # Multiple color histograms
                hsv_hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
                lab_hist = cv2.calcHist([lab], [1, 2], None, [8, 8], [-127, 127, -127, 127])
                
                # Normalize histograms
                hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()
                lab_hist = cv2.normalize(lab_hist, lab_hist).flatten()
                
                # Dominant colors
                dominant_colors = get_dominant_colors(player_patch, k=3)
                dominant_colors_flat = dominant_colors.flatten()
                
            except cv2.error:
                # Fallback for problematic patches
                hsv_hist = np.zeros(144)
                lab_hist = np.zeros(64)
                dominant_colors_flat = np.zeros(9)
            
            # 3. Texture features
            gray = cv2.cvtColor(player_patch, cv2.COLOR_BGR2GRAY)
            texture_features = extract_texture_features(gray)
            
            features.append({
                "centroid": centroid,
                "norm_centroid": norm_centroid,
                "spatial": [norm_width, norm_height, aspect_ratio, norm_area],
                "hsv_hist": hsv_hist,
                "lab_hist": lab_hist,
                "dominant_colors": dominant_colors_flat,
                "texture": texture_features,
                "box": (x1, y1, x2, y2),
                "confidence": box.conf.item(),
                "class": cls
            })
    
    return features

def calculate_feature_similarity(feat1, feat2, weights=None):
    """Calculate similarity between two feature vectors"""
    if weights is None:
        weights = {
            'spatial': 0.3,
            'hsv_color': 0.25,
            'lab_color': 0.2,
            'dominant_color': 0.15,
            'texture': 0.1
        }
    
    # Spatial cost (normalized position)
    spatial_cost = np.linalg.norm(
        np.array(feat1["norm_centroid"]) - np.array(feat2["norm_centroid"])
    )
    
    # Size similarity
    size_cost = np.linalg.norm(
        np.array(feat1["spatial"][:3]) - np.array(feat2["spatial"][:3])
    )
    
    # Color similarities
    try:
        hsv_similarity = 1 - cosine_similarity(
            feat1["hsv_hist"].reshape(1, -1),
            feat2["hsv_hist"].reshape(1, -1)
        )[0][0]
        
        lab_similarity = 1 - cosine_similarity(
            feat1["lab_hist"].reshape(1, -1),
            feat2["lab_hist"].reshape(1, -1)
        )[0][0]
        
        # Dominant color similarity
        dominant_similarity = np.linalg.norm(
            feat1["dominant_colors"] - feat2["dominant_colors"]
        ) / 255.0  # Normalize by max color value
        
    except (ValueError, ZeroDivisionError):
        hsv_similarity = 1.0
        lab_similarity = 1.0 
        dominant_similarity = 1.0
    
    # Texture similarity
    texture_cost = np.linalg.norm(feat1["texture"] - feat2["texture"])
    
    # Combine with weights
    total_cost = (
        weights['spatial'] * (spatial_cost + size_cost) +
        weights['hsv_color'] * hsv_similarity +
        weights['lab_color'] * lab_similarity +
        weights['dominant_color'] * dominant_similarity +
        weights['texture'] * texture_cost
    )
    
    return total_cost

def advanced_player_matching(b_features, t_features, cost_threshold=0.8):
    """Multi-criteria matching with weighted importance"""
    if not b_features or not t_features:
        return {}
    
    n_b, n_t = len(b_features), len(t_features)
    cost_matrix = np.zeros((n_b, n_t))
    
    # Calculate cost matrix
    for i, b_feat in enumerate(b_features):
        for j, t_feat in enumerate(t_features):
            cost_matrix[i, j] = calculate_feature_similarity(b_feat, t_feat)
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Filter out poor matches based on cost threshold
    valid_matches = {}
    match_costs = {}
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < cost_threshold:
            valid_matches[f"broadcast_{i}"] = f"tacticam_{j}"
            match_costs[f"broadcast_{i}"] = cost_matrix[i, j]
    
    return valid_matches, match_costs

def estimate_homography(b_features, t_features, matches):
    """Estimate homography between camera views"""
    if len(matches) < 4:
        return None
    
    b_points = []
    t_points = []
    
    for b_id, t_id in matches.items():
        try:
            b_idx = int(b_id.split('_')[1])
            t_idx = int(t_id.split('_')[1])
            b_points.append(b_features[b_idx]["centroid"])
            t_points.append(t_features[t_idx]["centroid"])
        except (IndexError, ValueError):
            continue
    
    if len(b_points) < 4:
        return None
    
    b_points = np.array(b_points, dtype=np.float32)
    t_points = np.array(t_points, dtype=np.float32)
    
    try:
        H, mask = cv2.findHomography(b_points, t_points, cv2.RANSAC, 5.0)
        return H
    except cv2.error:
        return None

def validate_matches_across_frames(all_matches, window_size=5):
    """Validate matches using temporal consistency"""
    if len(all_matches) < window_size:
        return {}
    
    # Count match frequency across frames
    match_frequency = defaultdict(int)
    total_frames = len(all_matches)
    
    for matches, _ in all_matches:
        for b_id, t_id in matches.items():
            match_frequency[(b_id, t_id)] += 1
    
    # Keep matches that appear in at least 30% of frames
    min_frequency = max(1, int(0.3 * total_frames))
    validated_matches = {}
    
    for (b_id, t_id), freq in match_frequency.items():
        if freq >= min_frequency:
            validated_matches[b_id] = t_id
    
    return validated_matches

def is_keyframe(frame_idx, keyframe_interval=5):
    """Determine if frame should be processed as keyframe"""
    return frame_idx % keyframe_interval == 0

def parallel_detection(frames, model_func):
    """Process detections in parallel"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(model_func, frames))
    return results

# Initialize trackers
broadcast_tracker = PlayerTracker(max_age=15, distance_threshold=150)
tacticam_tracker = PlayerTracker(max_age=15, distance_threshold=150)

# Feature storage
broadcast_features = []
tacticam_features = []
all_matches = []
keyframe_matches = []

# Homography matrix
homography_matrix = None

print("Processing videos...")

# Process videos frame by frame
frame_idx = 0
keyframe_count = 0
max_frames = min(int(broadcast_cap.get(cv2.CAP_PROP_FRAME_COUNT)), 
                int(tacticam_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

while frame_idx < max_frames:
    # Read frames
    ret1, broadcast_frame = broadcast_cap.read()
    ret2, tacticam_frame = tacticam_cap.read()
    if not ret1 or not ret2:
        break
    
    # Process keyframes with full analysis
    if is_keyframe(frame_idx, keyframe_interval=10):
        print(f"Processing keyframe {keyframe_count + 1} (frame {frame_idx}/{max_frames})")
        
        # Detect players in both feeds
        broadcast_detections = model(broadcast_frame)
        tacticam_detections = model(tacticam_frame)
        
        # Extract enhanced features
        broadcast_frame_features = get_enhanced_player_features(
            broadcast_frame, broadcast_detections, broadcast_width, broadcast_height
        )
        tacticam_frame_features = get_enhanced_player_features(
            tacticam_frame, tacticam_detections, tacticam_width, tacticam_height
        )
        
        # Update trackers
        broadcast_tracker.update(broadcast_frame_features)
        tacticam_tracker.update(tacticam_frame_features)
        
        # Store features
        broadcast_features.append(broadcast_frame_features)
        tacticam_features.append(tacticam_frame_features)
        
        # Perform matching
        if broadcast_frame_features and tacticam_frame_features:
            matches, costs = advanced_player_matching(
                broadcast_frame_features, 
                tacticam_frame_features,
                cost_threshold=0.7
            )
            all_matches.append((matches, costs))
            
            # Save first keyframe for visualization
            if keyframe_count == 0:
                # Draw detections on broadcast frame
                vis_broadcast = broadcast_frame.copy()
                for i, player in enumerate(broadcast_frame_features):
                    box = player["box"]
                    color = (0, 255, 0)  # Green for broadcast
                    cv2.rectangle(vis_broadcast, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(vis_broadcast, f"B{i} ({player['confidence']:.2f})", 
                               (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw detections on tacticam frame
                vis_tacticam = tacticam_frame.copy()
                for j, player in enumerate(tacticam_frame_features):
                    box = player["box"]
                    color = (0, 0, 255)  # Red for tacticam
                    cv2.rectangle(vis_tacticam, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(vis_tacticam, f"T{j} ({player['confidence']:.2f})", 
                               (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw matches
                for b_id, t_id in matches.items():
                    b_idx = int(b_id.split('_')[1])
                    t_idx = int(t_id.split('_')[1])
                    cost = costs[b_id]
                    
                    # Add match indicator
                    b_box = broadcast_frame_features[b_idx]["box"]
                    t_box = tacticam_frame_features[t_idx]["box"]
                    
                    cv2.putText(vis_broadcast, f"->T{t_idx} ({cost:.2f})", 
                               (b_box[0], b_box[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(vis_tacticam, f"<-B{b_idx} ({cost:.2f})", 
                               (t_box[0], t_box[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Save visualization
                cv2.imwrite("C:/playerdetect/output/broadcast_enhanced.jpg", vis_broadcast)
                cv2.imwrite("C:/playerdetect/output/tacticam_enhanced.jpg", vis_tacticam)
                
                # Estimate homography if enough matches
                if len(matches) >= 4:
                    homography_matrix = estimate_homography(
                        broadcast_frame_features, tacticam_frame_features, matches
                    )
                    if homography_matrix is not None:
                        print(f"Homography estimated with {len(matches)} point pairs")
        
        keyframe_count += 1
    
    frame_idx += 1

# Release video captures
broadcast_cap.release()
tacticam_cap.release()

print(f"\nProcessed {frame_idx} frames ({keyframe_count} keyframes)")

# Final analysis and results
if all_matches:
    print(f"\nFound matches in {len(all_matches)} keyframes")
    
    # Get temporal validation
    validated_matches = validate_matches_across_frames(all_matches, window_size=3)
    
    print(f"\nTemporally Validated Player Mappings ({len(validated_matches)} pairs):")
    print("-" * 50)
    
    if validated_matches:
        for b_id, t_id in validated_matches.items():
            # Calculate average cost for this match
            match_costs = []
            for matches, costs in all_matches:
                if b_id in matches and matches[b_id] == t_id:
                    match_costs.append(costs[b_id])
            
            avg_cost = np.mean(match_costs) if match_costs else 0
            confidence = max(0, (1 - avg_cost) * 100)  # Convert to confidence percentage
            
            print(f"{b_id} -> {t_id} (confidence: {confidence:.1f}%, appears in {len(match_costs)} keyframes)")
    
    # Save detailed results
    results_file = "C:/playerdetect/output/matching_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Enhanced Player Mapping Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Processed {frame_idx} frames ({keyframe_count} keyframes)\n")
        f.write(f"Video Properties:\n")
        f.write(f"  Broadcast: {broadcast_width}x{broadcast_height} @ {broadcast_fps} FPS\n")
        f.write(f"  Tacticam: {tacticam_width}x{tacticam_height} @ {tacticam_fps} FPS\n\n")
        
        if homography_matrix is not None:
            f.write(f"Homography Matrix:\n{homography_matrix}\n\n")
        
        f.write(f"Validated Matches:\n")
        for b_id, t_id in validated_matches.items():
            match_costs = []
            for matches, costs in all_matches:
                if b_id in matches and matches[b_id] == t_id:
                    match_costs.append(costs[b_id])
            avg_cost = np.mean(match_costs) if match_costs else 0
            confidence = max(0, (1 - avg_cost) * 100)
            f.write(f"  {b_id} -> {t_id} (confidence: {confidence:.1f}%)\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("Enhanced visualizations saved to output/ directory")
    
else:
    print("No matches found. Check video quality and detection model performance.")

print("\nEnhanced processing complete!")