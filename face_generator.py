import json
import math
import os
from typing import List, Tuple

# Type definitions
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]

def generate_smiley_face_segments(
    canvas_size: Tuple[int, int] = (1080, 1080),
    num_segments: int = 30,
    margin_percent: float = 0.1
) -> List[LineSegment]:
    """Generate line segments that form a smiley face within the canvas bounds"""
    segments = []
    width, height = canvas_size
    
    # Calculate margins
    margin_x = width * margin_percent
    margin_y = height * margin_percent
    
    # Calculate usable canvas area
    usable_width = width - 2 * margin_x
    usable_height = height - 2 * margin_y
    
    # Calculate center point and radius of the face
    center_x = width / 2
    center_y = height / 2
    radius = min(usable_width, usable_height) / 2
    
    # Parameters for various features
    eye_radius = radius * 0.15
    eye_offset_x = radius * 0.4
    eye_offset_y = radius * 0.2
    mouth_width = radius * 1.2
    mouth_height = radius * 0.6
    mouth_offset_y = radius * 0.1
    
    # Generate the face outline
    face_segments = generate_circle_segments(
        center=(center_x, center_y),
        radius=radius,
        num_segments=16
    )
    segments.extend(face_segments)
    
    # Generate left eye
    left_eye_center = (center_x - eye_offset_x, center_y - eye_offset_y)
    left_eye_segments = generate_circle_segments(
        center=left_eye_center,
        radius=eye_radius,
        num_segments=4
    )
    segments.extend(left_eye_segments)
    
    # Generate right eye
    right_eye_center = (center_x + eye_offset_x, center_y - eye_offset_y)
    right_eye_segments = generate_circle_segments(
        center=right_eye_center,
        radius=eye_radius,
        num_segments=4
    )
    segments.extend(right_eye_segments)
    
    # Generate smile (bottom half of an ellipse)
    smile_segments = generate_smile_segments(
        center=(center_x, center_y + mouth_offset_y),
        width=mouth_width,
        height=mouth_height,
        num_segments=6
    )
    segments.extend(smile_segments)
    
    return segments

def generate_circle_segments(
    center: Point,
    radius: float,
    num_segments: int
) -> List[LineSegment]:
    """Generate line segments forming a circle"""
    segments = []
    center_x, center_y = center
    
    for i in range(num_segments):
        angle1 = 2 * math.pi * i / num_segments
        angle2 = 2 * math.pi * (i + 1) / num_segments
        
        x1 = center_x + radius * math.cos(angle1)
        y1 = center_y + radius * math.sin(angle1)
        x2 = center_x + radius * math.cos(angle2)
        y2 = center_y + radius * math.sin(angle2)
        
        segments.append(((x1, y1), (x2, y2)))
    
    return segments

def generate_smile_segments(
    center: Point,
    width: float,
    height: float,
    num_segments: int
) -> List[LineSegment]:
    """Generate line segments forming a smile (bottom half of an ellipse)"""
    segments = []
    center_x, center_y = center
    
    for i in range(num_segments):
        # Only use the bottom half of the ellipse (0 to pi)
        angle1 = math.pi * i / (num_segments - 1)
        angle2 = math.pi * (i + 1) / (num_segments - 1)
        
        x1 = center_x + (width / 2) * math.cos(angle1)
        y1 = center_y + (height / 2) * math.sin(angle1)
        x2 = center_x + (width / 2) * math.cos(angle2)
        y2 = center_y + (height / 2) * math.sin(angle2)
        
        segments.append(((x1, y1), (x2, y2)))
    
    return segments

def save_segments_to_file(segments: List[LineSegment], filename: str):
    """Save segments to a JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(segments, f, indent=2)

def main():
    """Generate a smiley face segments file when run directly"""
    canvas_size = (1080, 1080)
    
    # Create input directory if it doesn't exist
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    # Generate smiley face segments
    segments = generate_smiley_face_segments(canvas_size=canvas_size)
    
    # Save to file
    filename = f"{input_dir}/smiley_face_segments.json"
    save_segments_to_file(segments, filename)
    
    print(f"Generated {len(segments)} line segments forming a smiley face and saved to {filename}")

if __name__ == "__main__":
    main()