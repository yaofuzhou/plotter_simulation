import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
import json
import math
import os
from typing import List, Tuple, Dict, Any

# Import algorithm classes from the dedicated algorithms file
from algorithms import (
    PlotterAlgorithm, 
    BruteForceAlgorithm, 
    GreedyAlgorithm,
    KDTreeNearestNeighborAlgorithm,
    RTreeNearestNeighborAlgorithm
    # NearestInsertionAlgorithm,
    # TwoOptAlgorithm,
    # SimulatedAnnealingAlgorithm
)

# Type definitions for clarity
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]
Instruction = Dict[str, Any]  # Will contain type and points

def load_segments_from_file(filename: str) -> List[LineSegment]:
    """Load segments from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def save_instructions_to_file(instructions: List[Instruction], algorithm_name: str, input_filename: str):
    """Save instructions to a JSON file named after the algorithm and input file"""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Get base name of input file without extension
    base_input_name = os.path.splitext(os.path.basename(input_filename))[0]
    
    # Create output filename
    output_filename = f"output/{base_input_name}-{algorithm_name}-instructions.json"
    
    with open(output_filename, 'w') as f:
        json.dump(instructions, f, indent=2)
    
    return output_filename


def generate_report(segments: List[LineSegment], instructions: List[Instruction], algorithm_name: str):
    """Generate a report about the plotting process with expanded distance metrics"""
    total_segments = len(segments)
    total_instructions = len(instructions)
    
    # Calculate Euclidean distances
    euclidean_drawing_distance = sum(inst["euclidean_distance"] for inst in instructions if inst["type"] == "PEN_DOWN")
    euclidean_travel_distance = sum(inst["euclidean_distance"] for inst in instructions if inst["type"] == "PEN_UP")
    euclidean_total_distance = euclidean_drawing_distance + euclidean_travel_distance
    euclidean_efficiency = euclidean_drawing_distance / euclidean_total_distance if euclidean_total_distance > 0 else 0
    
    # Calculate Manhattan distances
    manhattan_drawing_distance = sum(inst["manhattan_distance"] for inst in instructions if inst["type"] == "PEN_DOWN")
    manhattan_travel_distance = sum(inst["manhattan_distance"] for inst in instructions if inst["type"] == "PEN_UP")
    manhattan_total_distance = manhattan_drawing_distance + manhattan_travel_distance
    manhattan_efficiency = manhattan_drawing_distance / manhattan_total_distance if manhattan_total_distance > 0 else 0
    
    print(f"\n===== PLOTTER ALGORITHM REPORT: {algorithm_name.upper()} =====")
    print(f"Number of segments: {total_segments}")
    print(f"Number of instructions: {total_instructions}")
    
    print("\n--- Euclidean (L2) Distance Metrics ---")
    print(f"Drawing distance: {euclidean_drawing_distance:.2f} units")
    print(f"Travel distance: {euclidean_travel_distance:.2f} units")
    print(f"Total distance: {euclidean_total_distance:.2f} units")
    print(f"Drawing efficiency: {euclidean_efficiency:.2%}")
    
    print("\n--- Manhattan (L1) Distance Metrics ---")
    print(f"Drawing distance: {manhattan_drawing_distance:.2f} units")
    print(f"Travel distance: {manhattan_travel_distance:.2f} units")
    print(f"Total distance: {manhattan_total_distance:.2f} units")
    print(f"Drawing efficiency: {manhattan_efficiency:.2%}")
    
    print("====================================\n")
    
    return {
        "algorithm": algorithm_name,
        "segments": total_segments,
        "instructions": total_instructions,
        "euclidean": {
            "drawing_distance": euclidean_drawing_distance,
            "travel_distance": euclidean_travel_distance,
            "total_distance": euclidean_total_distance,
            "efficiency": euclidean_efficiency
        },
        "manhattan": {
            "drawing_distance": manhattan_drawing_distance,
            "travel_distance": manhattan_travel_distance,
            "total_distance": manhattan_total_distance,
            "efficiency": manhattan_efficiency
        }
    }


def process_with_algorithm(input_filename: str, algorithm):
    """Process an input file with the given algorithm and generate a report"""
    segments = load_segments_from_file(input_filename)
    instructions = algorithm.process_segments(segments)
    output_filename = save_instructions_to_file(instructions, algorithm.name, input_filename)
    report = generate_report(segments, instructions, algorithm.name)
    
    # Generate visualization
    vis_filename = visualize_algorithm(segments, instructions, algorithm.name, algorithm.canvas_size)
    
    print(f"Instructions saved to {output_filename}")
    print(f"Visualization saved to {vis_filename}")
    
    return instructions, report


def visualize_algorithm(segments: List[LineSegment], instructions: List[Instruction], algorithm_name: str, canvas_size: Tuple[int, int]):
    """
    Create a visualization of the algorithm's path planning and save as PNG.
    
    Args:
        segments: Original line segments to draw
        instructions: Plotter instructions generated by the algorithm
        algorithm_name: Name of the algorithm for the filename
        canvas_size: Size of the canvas (width, height)
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set limits based on canvas size
    ax.set_xlim(0, canvas_size[0])
    ax.set_ylim(0, canvas_size[1])
    
    # Invert y-axis to match the canvas coordinates (0,0 at top-left)
    ax.invert_yaxis()
    
    # Draw the original segments
    segments_lines = []
    for segment in segments:
        segments_lines.append(segment)
    
    # Plot original segments in light gray
    lc_segments = LineCollection(segments_lines, colors='lightgray', linewidths=2, alpha=0.5, label='Original Segments')
    ax.add_collection(lc_segments)
    
    # Extract pen movements
    pen_up_lines = []
    pen_down_lines = []
    
    for instruction in instructions:
        from_point = instruction["from"]
        to_point = instruction["to"]
        if instruction["type"] == "PEN_UP":
            pen_up_lines.append([from_point, to_point])
        else:  # PEN_DOWN
            pen_down_lines.append([from_point, to_point])
    
    # Plot pen movements
    if pen_up_lines:
        lc_pen_up = LineCollection(pen_up_lines, colors='red', linewidths=1, linestyles='dashed', label='Pen Up (Travel)')
        ax.add_collection(lc_pen_up)
    
    if pen_down_lines:
        lc_pen_down = LineCollection(pen_down_lines, colors='blue', linewidths=2, label='Pen Down (Drawing)')
        ax.add_collection(lc_pen_down)
    
    # Mark start point (0,0) with a green dot
    ax.plot(0, 0, 'go', markersize=10, label='Start Point (0,0)')
    
    # Add numbered markers for segment ordering
    for i, segment in enumerate(segments):
        midpoint = ((segment[0][0] + segment[1][0])/2, (segment[0][1] + segment[1][1])/2)
        ax.text(midpoint[0], midpoint[1], str(i+1), fontsize=8, 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add movement sequence numbers
    for i, instruction in enumerate(instructions):
        if i % 5 == 0:  # Only label every 5th move to avoid clutter
            midpoint = ((instruction["from"][0] + instruction["to"][0])/2, 
                        (instruction["from"][1] + instruction["to"][1])/2)
            ax.text(midpoint[0], midpoint[1], f"{i}", fontsize=6, color='purple',
                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # Add legend, title and grid
    ax.legend(loc='upper right')
    ax.set_title(f'Plotter Path Planning: {algorithm_name.upper()}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Save figure
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/{algorithm_name}_visualization.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization saved to {filepath}")
    
    return filepath


def visualize_comparison(results: List[Dict]):
    """Create a bar chart comparing the algorithms"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data
        names = [r['name'].replace('_', ' ').title() for r in results]
        efficiencies = [r['euclidean_efficiency'] * 100 for r in results]
        travel_distances = [r['euclidean_travel'] for r in results]
        times = [r['execution_time'] for r in results]
        
        # Set up figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot efficiency comparison
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, efficiencies, width, label='Efficiency (%)')
        ax1.bar(x + width/2, [t/max(times)*100 for t in times], width, 
                label='Relative Time (%)', alpha=0.7)
        
        ax1.set_ylabel('Percentage')
        ax1.set_title('Efficiency & Computing Time')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot travel distance comparison
        ax2.bar(x, travel_distances, color='red', alpha=0.7)
        ax2.set_ylabel('Travel Distance (units)')
        ax2.set_title('Pen-Up Travel Distance (Lower is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(travel_distances):
            ax2.text(i, v + 50, f'{v:.0f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        filepath = f"{output_dir}/algorithm_comparison.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Comparison visualization saved to {filepath}")
        
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")



def visualize_all_paths(segments: List[LineSegment], algorithm_results: Dict[str, List[Instruction]], canvas_size: Tuple[int, int]):
    """
    Create a visualization showing all algorithms' paths side by side.
    
    Args:
        segments: Original line segments
        algorithm_results: Dictionary mapping algorithm names to their instructions
        canvas_size: Size of the canvas (width, height)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    
    # Calculate grid size for subplots
    n_algorithms = len(algorithm_results)
    cols = min(2, n_algorithms)
    rows = (n_algorithms + cols - 1) // cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])  # Ensure axes is always an array
    axes = axes.flatten()
    
    # Original segments
    segments_lines = segments.copy()
    
    # For each algorithm
    for i, (alg_name, instructions) in enumerate(algorithm_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Set limits based on canvas size
        ax.set_xlim(0, canvas_size[0])
        ax.set_ylim(0, canvas_size[1])
        
        # Invert y-axis to match the canvas coordinates (0,0 at top-left)
        ax.invert_yaxis()
        
        # Plot original segments in light gray
        lc_segments = LineCollection(segments_lines, colors='lightgray', linewidths=2, alpha=0.3, label='Segments')
        ax.add_collection(lc_segments)
        
        # Extract pen movements
        pen_up_lines = []
        pen_down_lines = []
        
        for instruction in instructions:
            from_point = instruction["from"]
            to_point = instruction["to"]
            if instruction["type"] == "PEN_UP":
                pen_up_lines.append([from_point, to_point])
            else:  # PEN_DOWN
                pen_down_lines.append([from_point, to_point])
        
        # Plot pen movements
        if pen_up_lines:
            lc_pen_up = LineCollection(pen_up_lines, colors='red', linewidths=1, linestyles='dashed', 
                                      label=f'Pen Up ({len(pen_up_lines)} moves)')
            ax.add_collection(lc_pen_up)
        
        if pen_down_lines:
            lc_pen_down = LineCollection(pen_down_lines, colors='blue', linewidths=1.5, 
                                        label=f'Pen Down ({len(pen_down_lines)} moves)')
            ax.add_collection(lc_pen_down)
        
        # Mark start point (0,0) with a green dot
        ax.plot(0, 0, 'go', markersize=6, label='Start Point (0,0)')
        
        # Add title and grid
        ax.set_title(f'{alg_name.replace("_", " ").title()}')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', fontsize='small')
    
    # Hide empty subplots
    for i in range(n_algorithms, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    plt.suptitle('Algorithm Path Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust for overall title
    
    # Save figure
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/algorithm_path_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Path comparison visualization saved to {filepath}")


def main():
    """Main function to test algorithms independently"""
    import line_generator
    
    canvas_size = (1080, 1080)
    
    # Generate random segments file if it doesn't exist
    input_dir = "input"
    os.makedirs(input_dir, exist_ok=True)
    input_filename = f"{input_dir}/random_segments.json"
    # input_filename = f"{input_dir}/smiley_face_segments.json"
    # input_filename = f"{input_dir}/edge_segments.json"
    
    if not os.path.exists(input_filename):
        print(f"Generating random line segments file: {input_filename}")
        segments = line_generator.generate_random_segments(num_segments=20, canvas_size=canvas_size)
        line_generator.save_segments_to_file(segments, input_filename)
    
    # Load segments once
    segments = load_segments_from_file(input_filename)
    
    # List of algorithms to test
    algorithms = [
        BruteForceAlgorithm(canvas_size),
        GreedyAlgorithm(canvas_size),
        KDTreeNearestNeighborAlgorithm(canvas_size),
        RTreeNearestNeighborAlgorithm(canvas_size)
        # NearestInsertionAlgorithm(canvas_size),
        # TwoOptAlgorithm(canvas_size),
        # SimulatedAnnealingAlgorithm(canvas_size)
    ]
    
    # Process with each algorithm
    results = []
    all_instructions = {}
    
    for algorithm in algorithms:
        try:
            print(f"\nProcessing with {algorithm.name.replace('_', ' ').title()} algorithm...")
            
            # Time the execution
            import time
            start_time = time.time()
            
            # Process segments
            instructions, report = process_with_algorithm(input_filename, algorithm)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Store results for comparison
            results.append({
                'name': algorithm.name,
                'euclidean_efficiency': report['euclidean']['efficiency'],
                'euclidean_travel': report['euclidean']['travel_distance'],
                'manhattan_efficiency': report['manhattan']['efficiency'],
                'execution_time': execution_time
            })
            
            # Store instructions for later visualization
            all_instructions[algorithm.name] = instructions
            
        except Exception as e:
            print(f"Error processing with {algorithm.name} algorithm: {str(e)}")
    
    if not results:
        print("No algorithms completed successfully.")
        return
    
    # Print comparison table
    print("\n===== ALGORITHM COMPARISON =====")
    print(f"{'Algorithm':<25} {'Efficiency':<12} {'Travel Dist':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for result in results:
        name = result['name'].replace('_', ' ').title()
        efficiency = f"{result['euclidean_efficiency']:.2%}"
        travel = f"{result['euclidean_travel']:.2f}"
        exec_time = f"{result['execution_time']:.3f}"
        
        print(f"{name:<25} {efficiency:<12} {travel:<12} {exec_time:<10}")
    
    print("=" * 60)
    
    # Create comparison visualization
    try:
        visualize_comparison(results)
    except Exception as e:
        print(f"Error creating comparison visualization: {str(e)}")
    
    # Visualize all paths
    try:
        if all_instructions:
            visualize_all_paths(segments, all_instructions, canvas_size)
    except Exception as e:
        print(f"Error creating path visualization: {str(e)}")
    
    print("All algorithms completed.")


if __name__ == "__main__":
    main()
