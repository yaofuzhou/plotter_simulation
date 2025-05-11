import cv2
import numpy as np
import os
import re
from collections import defaultdict

def combine_videos_in_grid(video_paths, output_path, output_resolution=(1080, 1080), grid_size=(2, 2)):
    """
    Combine multiple videos into a grid layout with specified output resolution using OpenCV.
    
    Args:
        video_paths (list): List of paths to the input videos
        output_path (str): Path to save the output video
        output_resolution (tuple): Output resolution (width, height)
        grid_size (tuple): Grid size (rows, cols)
    """
    # Open all videos with OpenCV
    video_captures = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            # Clean up already opened videos
            for opened_cap in video_captures:
                opened_cap.release()
            return
        video_captures.append(cap)
        
        # Print video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Loaded: {os.path.basename(path)} - Duration: {duration:.2f}s, Size: {width}x{height}, Frames: {frame_count}")
    
    # Calculate the size for each video in the grid
    grid_width = output_resolution[0] // grid_size[1]
    grid_height = output_resolution[1] // grid_size[0]
    
    # Get video metadata
    fps = video_captures[0].get(cv2.CAP_PROP_FPS)
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_captures]
    max_frames = max(frame_counts)
    
    print(f"Videos have {frame_counts} frames respectively")
    print(f"Maximum frames: {max_frames}, FPS: {fps}")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_resolution)
    
    # Cache the last frames for each video
    last_frames = [None] * len(video_captures)
    
    # Process frame by frame
    for frame_idx in range(max_frames):
        # Create an empty grid frame
        grid_frame = np.zeros((output_resolution[1], output_resolution[0], 3), dtype=np.uint8)
        
        # Process each video
        for i, cap in enumerate(video_captures):
            # Calculate position in the grid
            row = i // grid_size[1]
            col = i % grid_size[1]
            
            # Calculate the position in the output frame
            y_start = row * grid_height
            y_end = y_start + grid_height
            x_start = col * grid_width
            x_end = x_start + grid_width
            
            # Get the appropriate frame
            if frame_idx < frame_counts[i]:
                # For active videos, read the next frame
                if frame_idx > 0:  # If not the first frame
                    # Position the video at the correct frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = cap.read()
                if ret:
                    # Store this as the last valid frame
                    last_frames[i] = frame.copy()
                elif last_frames[i] is not None:
                    # Use the last valid frame if read fails
                    frame = last_frames[i]
                else:
                    # Create a blank frame if no valid frame is available
                    frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            else:
                # For finished videos, use the last frame
                if last_frames[i] is not None:
                    frame = last_frames[i]
                else:
                    # Create a blank frame if no valid frame is available
                    frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Resize the frame to fit the grid cell
            frame = cv2.resize(frame, (grid_width, grid_height))
            
            # Place the frame in the grid
            grid_frame[y_start:y_end, x_start:x_end] = frame
        
        # Write the grid frame to the output video
        out.write(grid_frame)
        
        # Print progress every 30 frames
        if frame_idx % 30 == 0 or frame_idx == max_frames - 1:
            print(f"Progress: {frame_idx+1}/{max_frames} frames ({(frame_idx+1)/max_frames*100:.1f}%)")
    
    # Release resources
    out.release()
    for cap in video_captures:
        cap.release()
    
    print(f"Combined video saved to: {output_path}")

def group_and_combine_videos():
    """Group videos by pattern a-*-c.mp4 and combine them into a-c.mp4 grid videos"""
    # Input directory
    input_dir = "output/animations"
    # Output directory for combined videos
    output_dir = "output/animations/combined_for_comparison"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all mp4 files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    # Group videos by pattern a-*-c.mp4
    video_groups = defaultdict(list)
    
    for video_file in video_files:
        # Extract parts from filename using regex
        match = re.match(r'(.+)-(.+)-(.+)\.mp4', video_file)
        if match:
            a, b, c = match.groups()
            
            # Group by a and c
            key = f"{a}-{c}"
            video_groups[key].append(os.path.join(input_dir, video_file))
    
    # Process each group that has exactly 4 videos
    for key, video_paths in video_groups.items():
        if len(video_paths) == 4:
            # Sort to ensure consistent ordering
            video_paths.sort()
            
            # Create output path
            output_path = os.path.join(output_dir, f"{key}.mp4")
            
            print(f"\nProcessing group: {key}")
            print(f"Videos to combine:")
            for path in video_paths:
                print(f"  - {os.path.basename(path)}")
            
            # Combine videos in grid
            combine_videos_in_grid(video_paths, output_path)
            print(f"Completed: {key}.mp4")
        else:
            print(f"Skipping group {key} with {len(video_paths)} videos (not exactly 4)")

if __name__ == "__main__":
    group_and_combine_videos()