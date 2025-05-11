import json
import math
import random
import os
from typing import List, Tuple

# Type definitions
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]

def generate_random_segments(
    num_segments: int = 50, 
    canvas_size: Tuple[int, int] = (1080, 1080),
    min_length: float = 50,
    max_length: float = 300
) -> List[LineSegment]:
    """Generate random line segments within the canvas bounds using only x,y coordinates"""
    segments = []
    width, height = canvas_size
    
    for _ in range(num_segments):
        # Generate random start point
        start_x = random.uniform(0, width)
        start_y = random.uniform(0, height)
        start_point = (start_x, start_y)
        
        # Generate random end point directly
        end_x = random.uniform(0, width)
        end_y = random.uniform(0, height)
        
        # Check if the length is within desired range, if not, adjust end point
        length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        if length < min_length or length > max_length:
            # Calculate unit vector from start to end
            dx = end_x - start_x
            dy = end_y - start_y
            
            # Normalize and scale to desired length
            target_length = random.uniform(min_length, max_length)
            scale = target_length / length if length > 0 else 0
            
            end_x = start_x + dx * scale
            end_y = start_y + dy * scale
            
            # Ensure end point is within canvas
            end_x = max(0, min(end_x, width))
            end_y = max(0, min(end_y, height))
        
        end_point = (end_x, end_y)
        segments.append((start_point, end_point))
    
    return segments


def save_segments_to_file(segments: List[LineSegment], filename: str):
    """Save segments to a JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(segments, f, indent=2)


def main():
    """Generate a random segments file when run directly"""
    canvas_size = (1080, 1080)
    
    # Create input directory if it doesn't exist
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    # Generate random segments
    segments = generate_random_segments(num_segments=30, canvas_size=canvas_size)
    
    # Save to file
    filename = f"{input_dir}/random_segments.json"
    save_segments_to_file(segments, filename)
    
    print(f"Generated {len(segments)} random line segments and saved to {filename}")


if __name__ == "__main__":
    main()
