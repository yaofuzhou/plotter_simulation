import json
import os
import math
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Type definitions
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]
Instruction = Dict[str, Any]

class PlotterSimulator:
    """Simulates a plotter executing drawing instructions with realistic timing"""
    
    def __init__(self, canvas_size_cm: Tuple[float, float] = (20.0, 20.0),
                 pen_speed_cm_per_sec: float = 60.0,
                 dt_seconds: float = 0.01,
                 mode: str = "independent"):
        """
        Initialize the plotter simulator.
        
        Args:
            canvas_size_cm: Size of the canvas in centimeters (width, height)
            pen_speed_cm_per_sec: Maximum pen speed in cm/sec
            dt_seconds: Time step for simulation in seconds
            mode: "independent" for independent x/y motion or "direct" for direct motion
        """
        self.canvas_size_cm = canvas_size_cm
        self.pen_speed_cm_per_sec = pen_speed_cm_per_sec
        self.dt_seconds = dt_seconds
        self.mode = mode
        
        # Convert canvas size to plotter units (assuming input uses plotter units)
        self.plotter_to_cm_ratio = min(canvas_size_cm) / 1080.0
        
        # State variables
        self.reset()
        
    def reset(self):
        """Reset the simulator to initial state"""
        self.current_position = (0, 0)  # In plotter units
        self.pen_down = False
        self.drawn_segments = []  # Segments drawn so far (in plotter units)
        
        # Time tracking
        self.elapsed_time = 0.0  # Total elapsed time in seconds
        
        # For animation
        self.position_history = []  # List of (time, x, y, pen_down) tuples
        
    def _convert_to_cm(self, point: Point) -> Point:
        """Convert plotter units to centimeters"""
        return (point[0] * self.plotter_to_cm_ratio, point[1] * self.plotter_to_cm_ratio)
        
    def _calculate_movement_time(self, from_point: Point, to_point: Point) -> float:
        """
        Calculate the time required to move between two points based on the motion mode.
        
        Returns:
            Time in seconds
        """
        # Convert points to cm for time calculation
        from_cm = self._convert_to_cm(from_point)
        to_cm = self._convert_to_cm(to_point)
        
        # Calculate distance in cm
        dx = to_cm[0] - from_cm[0]
        dy = to_cm[1] - from_cm[1]
        
        if self.mode == "independent":
            # Independent x and y motion: time determined by the axis requiring more time
            time_x = abs(dx) / self.pen_speed_cm_per_sec if dx != 0 else 0
            time_y = abs(dy) / self.pen_speed_cm_per_sec if dy != 0 else 0
            return max(time_x, time_y)
        else:  # "direct" mode
            # Direct motion: time determined by straight-line distance
            direct_distance = math.sqrt(dx*dx + dy*dy)
            return direct_distance / self.pen_speed_cm_per_sec
    
    def _interpolate_position(self, from_point: Point, to_point: Point, fraction: float) -> Point:
        """
        Interpolate between two points based on the motion mode.
        
        Args:
            from_point: Starting point in plotter units
            to_point: Ending point in plotter units
            fraction: Value between 0.0 and 1.0 representing progress
        
        Returns:
            Interpolated position in plotter units
        """
        if self.mode == "independent":
            # Independent x and y motion: both axes move at constant speeds
            # but one axis might finish before the other
            x = from_point[0] + (to_point[0] - from_point[0]) * fraction
            y = from_point[1] + (to_point[1] - from_point[1]) * fraction
            return (x, y)
        else:  # "direct" mode
            # Direct motion: move in a straight line at constant speed
            return (
                from_point[0] + (to_point[0] - from_point[0]) * fraction,
                from_point[1] + (to_point[1] - from_point[1]) * fraction
            )
    
    def simulate_instruction(self, instruction: Instruction) -> List[Tuple[float, Point, bool]]:
        """
        Simulate a single plotter instruction with time steps.
        
        Returns:
            List of (time, position, pen_down) tuples for each time step
        """
        from_point = instruction["from"]
        to_point = instruction["to"]
        pen_down = instruction["type"] == "PEN_DOWN"
        
        # Calculate total time for this movement
        total_time = self._calculate_movement_time(from_point, to_point)
        
        # Create a list of time steps
        time_steps = []
        current_time = self.elapsed_time
        
        # Add initial position if this is a pen state change
        if pen_down != self.pen_down:
            time_steps.append((current_time, from_point, pen_down))
            
        # Generate position at each time step
        num_steps = max(1, math.ceil(total_time / self.dt_seconds))
        
        for i in range(1, num_steps + 1):
            # Calculate fraction of movement completed
            fraction = min(1.0, i * self.dt_seconds / total_time if total_time > 0 else 1.0)
            
            # Update time
            current_time = self.elapsed_time + fraction * total_time
            
            # Calculate interpolated position
            position = self._interpolate_position(from_point, to_point, fraction)
            
            # Add to time steps
            time_steps.append((current_time, position, pen_down))
        
        # Update state
        self.current_position = to_point
        self.pen_down = pen_down
        self.elapsed_time += total_time
        
        # If drawing, add to drawn segments
        if pen_down:
            self.drawn_segments.append((from_point, to_point))
        
        return time_steps
    
    def execute_instructions(self, instructions: List[Instruction]) -> List[Tuple[float, Point, bool]]:
        """Execute a list of plotter instructions and record the time-based position history"""
        self.reset()
        all_time_steps = []
        
        for i, instruction in enumerate(instructions):
            # Simulate this instruction
            time_steps = self.simulate_instruction(instruction)
            all_time_steps.extend(time_steps)
            
            # Print progress
            if (i+1) % 10 == 0 or i+1 == len(instructions):
                print(f"Simulated {i+1}/{len(instructions)} instructions, elapsed time: {self.elapsed_time:.2f} seconds")
        
        # Return the complete time history
        return all_time_steps
    
    def load_instructions(self, filename: str) -> List[Instruction]:
        """Load instructions from a JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)


    def create_animation(self, time_steps: List[Tuple[float, Point, bool]], output_file: str, algorithm_name: str = ""):
        """
        Create an animation of the plotter simulation and save as MP4.
        The animation playback time will match the actual simulation time.
        
        Args:
            time_steps: List of (time, position, pen_down) tuples
            output_file: Output file path
            algorithm_name: Name of the algorithm used (for the title)
        """
        if not time_steps:
            print("No time steps to animate!")
            return
                
        # Extract data
        times = [t[0] for t in time_steps]
        positions = [t[1] for t in time_steps]
        pen_states = [t[2] for t in time_steps]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 10 * self.canvas_size_cm[1] / self.canvas_size_cm[0]))
        
        # Set up limits
        ax.set_xlim(0, 1080)  # Plotter units
        ax.set_ylim(1080, 0)  # Invert y-axis to match canvas coordinates
        
        # Set up elements for efficient line drawing
        line_collection = LineCollection([], colors='blue', linewidths=2)
        ax.add_collection(line_collection)
        
        pen_marker, = ax.plot([], [], 'go', markersize=8)
        time_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12)
        
        # Format algorithm name for title - convert from snake_case to Title Case
        formatted_algorithm_name = algorithm_name.replace('_', ' ').title() if algorithm_name else "Unknown"
        
        # Add title and grid
        ax.set_title(f"{formatted_algorithm_name} Algorithm - {self.mode.capitalize()} Motion Mode")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('X Position (plotter units)')
        ax.set_ylabel('Y Position (plotter units)')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Drawn Path'),
            Line2D([0], [0], marker='o', color='g', markersize=8, label='Pen Up', linestyle='None'),
            Line2D([0], [0], marker='o', color='r', markersize=8, label='Pen Down', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add axis scales showing cm
        ax_cm_x = ax.twiny()
        ax_cm_x.set_xlim(0, self.canvas_size_cm[0])
        ax_cm_x.set_xlabel('X Position (cm)')
        
        ax_cm_y = ax.twinx()
        ax_cm_y.set_ylim(self.canvas_size_cm[1], 0)
        ax_cm_y.set_ylabel('Y Position (cm)')
        
        # Detect direction changes in the pen's path
        direction_change_frames = set([0, len(times) - 1])  # Start and end are critical
        
        # Find frames where pen changes direction significantly (when pen is down)
        for i in range(2, len(times)):
            if pen_states[i] and pen_states[i-1] and pen_states[i-2]:  # Pen is down for 3 consecutive frames
                # Get vectors for previous and current movements
                prev_vector = (positions[i-1][0] - positions[i-2][0], positions[i-1][1] - positions[i-2][1])
                curr_vector = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
                
                # Compute magnitudes
                prev_mag = math.sqrt(prev_vector[0]**2 + prev_vector[1]**2)
                curr_mag = math.sqrt(curr_vector[0]**2 + curr_vector[1]**2)
                
                # Avoid division by zero
                if prev_mag > 1e-6 and curr_mag > 1e-6:
                    # Calculate dot product and angle between vectors
                    dot_product = prev_vector[0] * curr_vector[0] + prev_vector[1] * curr_vector[1]
                    cos_angle = dot_product / (prev_mag * curr_mag)
                    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
                    angle = math.acos(cos_angle)
                    
                    # If angle is significant (> 10 degrees), mark as direction change
                    if angle > math.radians(10):
                        direction_change_frames.add(i-1)  # The middle frame is the turning point
        
        # Always include pen state transitions
        for i in range(1, len(pen_states)):
            if pen_states[i] != pen_states[i-1]:
                direction_change_frames.add(i)
        
        # Define target duration for animation in seconds
        # This will match the actual simulation time
        target_duration = times[-1]  # Total simulation time
        
        # Define target FPS for the animation (standard video frame rate)
        target_fps = 30
        
        # Calculate total number of frames needed based on duration and FPS
        total_frames_needed = int(target_duration * target_fps)
        
        # Use evenly spaced frames from the simulation time
        # This ensures the animation matches real-time
        if len(times) > total_frames_needed:
            # If we have more simulation steps than needed frames,
            # select evenly spaced frames but always include critical ones
            critical_frames = sorted(list(direction_change_frames))
            
            # Create evenly spaced timestamps that we want to show
            target_timestamps = [i * (target_duration / total_frames_needed) for i in range(total_frames_needed)]
            
            # Find the closest time step for each target timestamp
            frame_indices = []
            for target_time in target_timestamps:
                # Find the index of the closest time step
                closest_idx = min(range(len(times)), key=lambda i: abs(times[i] - target_time))
                frame_indices.append(closest_idx)
            
            # Ensure critical frames are included
            for idx in critical_frames:
                if idx not in frame_indices:
                    # Find the nearest target frame to insert this critical frame
                    nearest_target_idx = min(range(len(target_timestamps)), 
                                            key=lambda i: abs(target_timestamps[i] - times[idx]))
                    frame_indices[nearest_target_idx] = idx
            
            # Remove duplicates and sort
            frame_indices = sorted(list(set(frame_indices)))
        else:
            # If we have fewer simulation steps than needed frames,
            # use all simulation frames and adjust FPS
            frame_indices = list(range(len(times)))
            
            # Recalculate FPS to match duration
            if times[-1] > 0:
                target_fps = len(frame_indices) / times[-1]
                print(f"Adjusting animation FPS to {target_fps:.2f} to match simulation time")
        
        # Ensure the last frame is included
        if frame_indices[-1] != len(times) - 1:
            frame_indices.append(len(times) - 1)
            frame_indices = sorted(frame_indices)
        
        print(f"Selected {len(frame_indices)} frames including {len(direction_change_frames)} critical points")
        print(f"Animation will play at {target_fps:.2f} FPS with duration of {target_duration:.2f} seconds")
        
        # Generate continuous drawing data
        # We'll build the complete drawing state for each frame
        complete_drawing_states = []
        
        # Process all frames sequentially for accurate state building
        all_segments = []  # Completed segments
        current_segment = None  # Active segment being drawn
        
        for i in range(len(times)):
            is_pen_down = pen_states[i]
            current_pos = positions[i]
            
            # Handle pen state changes
            if i > 0 and is_pen_down != pen_states[i-1]:
                # Pen just went down - start new segment
                if is_pen_down:
                    current_segment = [current_pos]
                # Pen just went up - finalize segment
                else:
                    if current_segment and len(current_segment) >= 2:
                        all_segments.append(current_segment)
                    current_segment = None
            # Continue current segment
            elif is_pen_down:
                if current_segment is None:
                    current_segment = [current_pos]
                else:
                    # Only add point if it's different from the last one
                    if current_segment[-1] != current_pos:
                        current_segment.append(current_pos)
            
            # If this is a selected frame, save the current state
            if i in frame_indices:
                # Create a copy of the current drawing state
                frame_segments = all_segments.copy()
                # Include the active segment if it exists and has at least 2 points
                if current_segment and len(current_segment) >= 2:
                    frame_segments.append(current_segment.copy())
                    
                # Convert to line segments for LineCollection
                line_segments = []
                for segment in frame_segments:
                    if len(segment) >= 2:
                        for j in range(len(segment) - 1):
                            line_segments.append([segment[j], segment[j+1]])
                
                complete_drawing_states.append((line_segments, current_pos, is_pen_down))

        # Define initialization function
        def init():
            line_collection.set_segments([])
            pen_marker.set_data([], [])
            time_text.set_text('')
            return line_collection, pen_marker, time_text
        
        # Define animation function
        def animate(i):
            frame_idx = frame_indices[i]
            current_time = times[frame_idx]
            
            # Get precomputed state
            line_segments, marker_pos, is_pen_down = complete_drawing_states[i]
            
            # Update line collection
            line_collection.set_segments(line_segments)
            
            # Update pen marker
            pen_marker.set_data([marker_pos[0]], [marker_pos[1]])
            pen_marker.set_color('red' if is_pen_down else 'green')
            
            # Update time text with progress indicator
            time_text.set_text(f'Time: {current_time:.2f} s / {times[-1]:.2f} s ({int(100*current_time/times[-1])}%)')
            
            return line_collection, pen_marker, time_text
        
        # Create animation
        print(f"Creating animation with {len(frame_indices)} frames from {len(times)} time steps...")
        
        anim = animation.FuncAnimation(
            fig, animate, frames=len(frame_indices),
            init_func=init, blit=True, interval=20)
        
        # Save animation with higher quality
        print(f"Saving animation to {output_file}...")
        writer = animation.FFMpegWriter(fps=30, bitrate=5000, 
                                      extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
        anim.save(output_file, writer=writer)
        
        # Close figure to free memory
        plt.close(fig)
        
        print(f"Animation saved to {output_file}")


    def generate_report(self) -> Dict[str, Any]:
        """Generate a report with statistics about the simulation"""
        return {
            "mode": self.mode,
            "canvas_size_cm": self.canvas_size_cm,
            "pen_speed_cm_per_sec": self.pen_speed_cm_per_sec,
            "total_time_seconds": self.elapsed_time,
            "drawn_segments": len(self.drawn_segments),
            "pen_down_distance_cm": sum(
                math.sqrt(
                    (seg[1][0] - seg[0][0])**2 + 
                    (seg[1][1] - seg[0][1])**2
                ) * self.plotter_to_cm_ratio
                for seg in self.drawn_segments
            ),
            "average_speed_cm_per_sec": sum(
                math.sqrt(
                    (seg[1][0] - seg[0][0])**2 + 
                    (seg[1][1] - seg[0][1])**2
                ) * self.plotter_to_cm_ratio
                for seg in self.drawn_segments
            ) / self.elapsed_time if self.elapsed_time > 0 else 0
        }

def process_instruction_file(file_path: str, mode: str, canvas_size_cm: Tuple[float, float],
                           pen_speed_cm_per_sec: float, dt_seconds: float) -> Dict[str, Any]:
    """
    Process a single instruction file with the given parameters.
    
    Args:
        file_path: Path to instruction file
        mode: Motion mode ("independent" or "direct")
        canvas_size_cm: Canvas size in cm (width, height)
        pen_speed_cm_per_sec: Pen speed in cm/sec
        dt_seconds: Time step in seconds
    
    Returns:
        Dictionary with results and paths
    """
    try:
        # Create simulator
        simulator = PlotterSimulator(
            canvas_size_cm=canvas_size_cm,
            pen_speed_cm_per_sec=pen_speed_cm_per_sec,
            dt_seconds=dt_seconds,
            mode=mode
        )
        
        # Get base name and create output directory
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Extract segment name and algorithm name from the file name
        # File name format is typically: "{segment_name}-{algorithm_name}-instructions.json"
        segment_name = "unknown"
        algorithm_name = "unknown"
        
        if "-" in base_name:
            parts = base_name.split("-")
            if len(parts) >= 3 and parts[-1] == "instructions":
                segment_name = parts[0]  # First part is segment name
                algorithm_name = parts[1]  # Second part is algorithm name
                
                # Handle multi-part segment names (if they contain dashes)
                if len(parts) > 3:
                    segment_name = "-".join(parts[:-2])
                    algorithm_name = parts[-2]
        
        output_dir = os.path.join("output", "animations")
        os.makedirs(output_dir, exist_ok=True)

        # Use format: segment-algorithm-mode.mp4
        segment_name_clean = segment_name.replace(' ', '_')
        algorithm_name_clean = algorithm_name.replace(' ', '_')
        output_file = os.path.join(output_dir, f"{segment_name_clean}-{algorithm_name_clean}-{mode}.mp4")
        
        # Load and execute instructions
        print(f"Processing {file_path} with {mode} motion mode...")
        instructions = simulator.load_instructions(file_path)
        time_steps = simulator.execute_instructions(instructions)
        
        # Create animation with algorithm name
        simulator.create_animation(time_steps, output_file, algorithm_name)
        
        # Generate report
        report = simulator.generate_report()
        report["file_name"] = file_name
        report["output_file"] = output_file
        report["algorithm_name"] = algorithm_name
        report["segment_name"] = segment_name
        
        return report
    
    except Exception as e:
        print(f"Error processing {file_path} with {mode} mode: {str(e)}")
        return {
            "file_name": os.path.basename(file_path),
            "mode": mode,
            "error": str(e)
        }

def main():
    """Run the plotter simulation with realistic timing and output videos"""
    # Check if there are instruction files in the output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        print("Output directory not found. Please run algorithm_processor.py first.")
        return
    
    # Find instruction files
    instruction_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                        if f.endswith("-instructions.json")]
    
    if not instruction_files:
        print("No instruction files found. Please run algorithm_processor.py first.")
        return
    
    # Configuration
    canvas_size_cm = (20.0, 20.0)
    pen_speed_cm_per_sec = 30.0  # 5 cm/sec (adjust as needed)
    dt_seconds = 0.01  # 10ms time step (100Hz)
    modes = ["independent", "direct"]
    
    print(f"Found {len(instruction_files)} instruction files.")
    print(f"Canvas size: {canvas_size_cm[0]}×{canvas_size_cm[1]} cm")
    print(f"Pen speed: {pen_speed_cm_per_sec} cm/sec")
    print(f"Time step: {dt_seconds} seconds")
    print(f"Motion modes: {', '.join(modes)}")
    
    # Prepare tasks for parallel processing
    tasks = []
    for file_path in instruction_files:
        for mode in modes:
            tasks.append((file_path, mode, canvas_size_cm, pen_speed_cm_per_sec, dt_seconds))
    
    # Use ProcessPoolExecutor for parallel processing
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    all_reports = []
    
    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        import concurrent.futures
        
        # Submit all tasks
        future_to_task = {
            executor.submit(process_instruction_file, *task): task 
            for task in tasks
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                report = future.result()
                all_reports.append(report)
                print(f"Completed: {os.path.basename(task[0])} with {task[1]} mode")
            except Exception as e:
                print(f"Task failed: {os.path.basename(task[0])} with {task[1]} mode - {str(e)}")
    
    # Print final reports
    print("\n===== SIMULATION REPORTS =====")
    
    for report in sorted(all_reports, key=lambda r: (r.get("file_name", ""), r.get("mode", ""))):
        if "error" in report:
            print(f"\nFile: {report.get('file_name', 'Unknown')}, Mode: {report.get('mode', 'Unknown')}")
            print(f"  ERROR: {report['error']}")
        else:
            print(f"\nFile: {report['file_name']}, Mode: {report['mode']}")
            print(f"  Total time: {report['total_time_seconds']:.2f} seconds")
            print(f"  Drawn segments: {report['drawn_segments']}")
            print(f"  Pen-down distance: {report['pen_down_distance_cm']:.2f} cm")
            print(f"  Average speed: {report['average_speed_cm_per_sec']:.2f} cm/sec")
            print(f"  Output: {os.path.basename(report['output_file'])}")
    
    print("\nAll simulations completed!")

if __name__ == "__main__":
    main()
