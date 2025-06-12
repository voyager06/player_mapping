# Player Detection and Cross-Camera Matching System

A sophisticated computer vision system that automatically detects, tracks, and matches players across multiple camera feeds of the same sports event. This system is particularly useful for sports analytics, allowing analysts to correlate player actions and positions across different camera angles.

## 🎯 Features

- **Multi-Camera Player Detection**: Simultaneously processes broadcast and tactical camera feeds
- **Advanced Feature Extraction**: Analyzes spatial, color, and texture features for robust player identification
- **Temporal Tracking**: Maintains player identities across video frames using Hungarian algorithm optimization
- **Cross-Camera Matching**: Automatically matches the same players between different camera views
- **Homography Estimation**: Calculates geometric transformations between camera perspectives
- **Temporal Validation**: Ensures match consistency across multiple frames to reduce false positives
- **Comprehensive Visualization**: Generates annotated output images and detailed analysis reports

## 🛠 Requirements

### Dependencies
```bash
pip install opencv-python
pip install numpy
pip install ultralytics
pip install scipy
pip install scikit-learn
pip install scikit-image
pip install concurrent-futures
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended for YOLO inference)
- Minimum 8GB RAM
- Custom trained YOLO model for player detection

## 📁 Project Structure

```
playerdetect/
├── player_detect.py          # Main script
├── best.pt                   # Custom YOLO model (required)
├── broadcast.mp4             # Broadcast camera feed
├── tacticam.mp4             # Tactical camera feed
└── output/                  # Generated results
    ├── broadcast_enhanced.jpg
    ├── tacticam_enhanced.jpg
    └── matching_results.txt
```

## 🚀 Usage

### Basic Usage
```python
python player_detect.py
```

### File Setup
1. Place your custom YOLO model file (`best.pt`) in the project directory
2. Add your video files:
   - `broadcast.mp4` - Main broadcast camera feed
   - `tacticam.mp4` - Tactical/overhead camera feed
3. Update file paths in the script if needed:
   ```python
   model = YOLO("C:/playerdetect/best.pt")
   broadcast_cap = cv2.VideoCapture("C:/playerdetect/broadcast.mp4")
   tacticam_cap = cv2.VideoCapture("C:/playerdetect/tacticam.mp4")
   ```

### Configuration Options

#### Tracker Parameters
```python
# Adjust tracking sensitivity
broadcast_tracker = PlayerTracker(max_age=15, distance_threshold=150)
tacticam_tracker = PlayerTracker(max_age=15, distance_threshold=150)
```

#### Processing Parameters
```python
# Keyframe interval (process every Nth frame)
keyframe_interval = 10

# Matching cost threshold
cost_threshold = 0.7

# Feature similarity weights
weights = {
    'spatial': 0.3,
    'hsv_color': 0.25,
    'lab_color': 0.2,
    'dominant_color': 0.15,
    'texture': 0.1
}
```

## 🔧 How It Works

### 1. **Player Detection**
- Uses custom YOLO model to detect players and goalkeepers
- Applies adaptive confidence thresholds based on detection quality
- Processes both video feeds simultaneously

### 2. **Feature Extraction**
For each detected player, the system extracts:
- **Spatial features**: Position, size, aspect ratio (normalized)
- **Color features**: HSV/LAB histograms, dominant colors via K-means
- **Texture features**: Local Binary Pattern analysis
- **Temporal features**: Tracking confidence and movement patterns

### 3. **Cross-Camera Matching**
- Calculates multi-dimensional similarity between players
- Uses Hungarian algorithm for optimal player assignment
- Applies cost thresholds to filter poor matches

### 4. **Temporal Validation**
- Validates matches across multiple frames
- Removes inconsistent matches (requires 30% frame consistency)
- Maintains robust player correspondences

### 5. **Geometric Analysis**
- Estimates homography matrix between camera views
- Provides geometric relationship understanding
- Requires minimum 4 validated match points

## 📊 Output

### Visualizations
- **broadcast_enhanced.jpg**: Annotated broadcast feed with detections and match indicators
- **tacticam_enhanced.jpg**: Annotated tactical feed with detections and match indicators

### Analysis Report
- **matching_results.txt**: Comprehensive results including:
  - Processing statistics
  - Validated player mappings with confidence scores
  - Homography matrix (if available)
  - Temporal consistency analysis

### Console Output
```
Processing videos...
Processing keyframe 1 (frame 0/1500)
Processing keyframe 2 (frame 10/1500)
...
Processed 1500 frames (150 keyframes)

Temporally Validated Player Mappings (8 pairs):
--------------------------------------------------
broadcast_0 -> tacticam_2 (confidence: 87.3%, appears in 12 keyframes)
broadcast_1 -> tacticam_0 (confidence: 92.1%, appears in 15 keyframes)
...
```

## ⚙️ Technical Details

### Algorithm Components
- **YOLO v8**: Object detection for player identification
- **Hungarian Algorithm**: Optimal bipartite matching
- **K-means Clustering**: Dominant color extraction
- **Local Binary Patterns**: Texture feature analysis
- **RANSAC**: Robust homography estimation

### Performance Optimizations
- **Keyframe Processing**: Analyzes every 10th frame for efficiency
- **Parallel Detection**: Multi-threaded processing capability
- **Adaptive Thresholds**: Dynamic confidence adjustment
- **Memory Management**: Efficient feature storage and retrieval

## 🎮 Use Cases

- **Sports Analytics**: Player performance analysis across camera angles
- **Broadcast Production**: Automated camera switching and player tracking
- **Tactical Analysis**: Coach review and strategy development
- **Research**: Multi-view sports video analysis studies

## 🐛 Troubleshooting

### Common Issues

**No matches found**
- Check video quality and lighting conditions
- Verify YOLO model is trained for your specific sport
- Adjust `cost_threshold` parameter (try 0.8-1.0)

**Poor tracking performance**
- Increase `keyframe_interval` for better temporal consistency
- Adjust `distance_threshold` in PlayerTracker
- Ensure videos are temporally synchronized

**Memory issues**
- Reduce video resolution or frame rate
- Increase `keyframe_interval`
- Process shorter video segments

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🔮 Future Enhancements

- [ ] Real-time processing capability
- [ ] Support for more than 2 camera feeds
- [ ] Deep learning-based feature extraction
- [ ] Web-based visualization interface
- [ ] Integration with sports analytics platforms
- [ ] Player jersey number recognition
- [ ] Action recognition and event detection
