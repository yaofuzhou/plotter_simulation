import json
import math
import random
import os
from typing import List, Tuple

# Type definitions
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]

def generate_edge_to_edge_segments(
    num_segments: int = 50,
    canvas_size: Tuple[int, int] = (1080, 1080),
    margin_percent: float = 0.05,  # 5% margin from edges
    edge_zone_percent: float = 0.15  # Points within 15% of edges
) -> List[LineSegment]:
    """Generate line segments with endpoints near the edges of the canvas"""
    segments = []
    width, height = canvas_size
    
    # Calculate margins
    margin_x = width * margin_percent
    margin_y = height * margin_percent
    
    # Calculate edge zones (where endpoints will be placed)
    edge_zone_width = width * edge_zone_percent
    edge_zone_height = height * edge_zone_percent
    
    # Define the four edge zones (with margins)
    edge_zones = [
        # Left edge zone
        (margin_x, margin_y, margin_x + edge_zone_width, height - margin_y),
        # Right edge zone
        (width - margin_x - edge_zone_width, margin_y, width - margin_x, height - margin_y),
        # Top edge zone
        (margin_x, margin_y, width - margin_x, margin_y + edge_zone_height),
        # Bottom edge zone
        (margin_x, height - margin_y - edge_zone_height, width - margin_x, height - margin_y)
    ]
    
    for _ in range(num_segments):
        # Select two different random edge zones for start and end points
        start_zone_index = random.randint(0, 3)
        end_zone_index = (start_zone_index + random.randint(1, 3)) % 4  # Ensure different zone
        
        # Get the selected zones
        start_zone = edge_zones[start_zone_index]
        end_zone = edge_zones[end_zone_index]
        
        # Generate random points within the selected zones
        start_x = random.uniform(start_zone[0], start_zone[2])
        start_y = random.uniform(start_zone[1], start_zone[3])
        start_point = (start_x, start_y)
        
        end_x = random.uniform(end_zone[0], end_zone[2])
        end_y = random.uniform(end_zone[1], end_zone[3])
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
    """Generate edge-to-edge segments file when run directly"""
    canvas_size = (1080, 1080)
    
    # Create input directory if it doesn't exist
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    
    # Generate segments with endpoints near edges
    segments = generate_edge_to_edge_segments(num_segments=30, canvas_size=canvas_size)
    
    # Save to file
    filename = f"{input_dir}/edge_segments.json"
    save_segments_to_file(segments, filename)
    
    print(f"Generated {len(segments)} line segments with endpoints near edges and saved to {filename}")

if __name__ == "__main__":
    main()