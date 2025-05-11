# Plotter Simulation Project

This project simulates a line segment plotter with two main components:
1. Algorithm component (Part 1) - Converts line segments to plotter instructions using various path optimization algorithms
2. Simulation component (Part 2) - Executes the instructions with realistic physics-based motion and generates visualizations

## Environment Setup

This project requires several Python packages. To avoid dependency issues, it's recommended to create a dedicated environment.

### Option 1: Using Conda (Recommended)

```bash
# Create a new environment named 'plotter_env'
conda create -n plotter_env python=3.9

# Activate the environment
conda activate plotter_env

# Install required packages
conda install -c conda-forge numpy matplotlib ffmpeg
conda install -c conda-forge scipy  # For KD-Tree implementation

# Verify installation
python -c "import numpy; import matplotlib; print('Packages installed successfully')"
```

### Option 2: Using pip and venv

```bash
# Create a new virtual environment
python -m venv plotter_venv

# Activate the environment
# On Windows:
plotter_venv\Scripts\activate
# On macOS/Linux:
source plotter_venv/bin/activate

# Install required packages
pip install numpy matplotlib scipy ffmpeg-python

# Verify installation
python -c "import numpy; import matplotlib; print('Packages installed successfully')"
```

### Troubleshooting

If you encounter issues with package installation:

1. **NumPy dependency errors** (especially on macOS):
   ```bash
   conda install -c conda-forge libgfortran
   conda install -c conda-forge numpy matplotlib
   ```

2. **FFmpeg issues**:
   ```bash
   # Install FFmpeg system-wide
   # On Ubuntu/Debian:
   sudo apt-get install ffmpeg
   # On macOS:
   brew install ffmpeg
   # On Windows:
   # Download from https://ffmpeg.org/download.html
   ```

3. **Complete reinstallation**:
   ```bash
   conda remove --name plotter_env --all
   conda create -n plotter_env python=3.9
   conda activate plotter_env
   conda install -c conda-forge numpy matplotlib ffmpeg scipy
   ```

Remember to activate the environment before running any scripts from this project:
```bash
conda activate plotter_env  # or source plotter_venv/bin/activate for pip/venv
```

## Project Structure

```
plotter_simulation/
├── input/                     # Directory for input files
│   └── README.md              # Instructions for input files
├── output/                    # Directory for output files
│   ├── animations/            # Animation videos from simulator
│   └── README.md              # Instructions for output files
├── line_generator.py          # Utility to generate random line segments
├── algorithms.py              # Path planning algorithms (core algorithms only)
├── algorithm_processor.py     # Processing and utility functions
├── plotter_simulator.py       # Physics-based simulator with video output
└── README.md                  # This file
```

## Generating Simulated User Input

There are several ways to generate input for the system:

### 1. Using the line_generator.py utility

The simplest way to generate random line segments:

```bash
python line_generator.py
```

This will:
- Create a file `input/random_segments.json` with 20 random line segments
- Each segment will have random start and end points within the 1080x1080 canvas
- Line lengths will be between 50 and 300 units

### 2. Customizing the random generation

You can modify `line_generator.py` to customize the random generation parameters:
- Change `num_segments` to generate more or fewer lines
- Adjust `min_length` and `max_length` to control line segment sizes
- Modify the canvas size for different dimensions

### 3. Creating your own input files

You can create your own input JSON files with the following format:

```json
[
  [[x1_start, y1_start], [x1_end, y1_end]],
  [[x2_start, y2_start], [x2_end, y2_end]],
  ...
]
```

Each line segment is represented as a pair of points, where each point is a pair of coordinates. Save these files in the `input/` directory with a `.json` extension.

## Running the Plotter Algorithms (Part 1)

To process line segments with path planning algorithms:

```bash
python algorithm_processor.py
```

This will:
1. Generate random input if it doesn't exist
2. Process the line segments with multiple optimization algorithms:
   - Brute Force (baseline)
   - Grid-based Greedy
   - KD-Tree Nearest Neighbor
   - R-Tree Nearest Neighbor
   - Nearest Insertion TSP
   - 2-Opt Local Search
   - Simulated Annealing
3. Save the resulting instructions to the output directory
4. Generate comparison charts and visualizations
5. Print detailed performance reports

## Running the Plotter Simulator (Part 2)

To create realistic physics-based simulations with videos:

```bash
python plotter_simulator.py
```

This will:
1. Load all instruction files from the output directory
2. Simulate the plotter's execution with two motion modes:
   - "Independent" - Independent x/y motion with shared speed limit
   - "Direct" - Single direct motion with overall speed limit
3. Generate MP4 animations of the drawing process
4. Print detailed timing and performance reports
5. Utilize all available CPU cores for parallel processing

### Simulator Features

The simulator provides:
- Realistic physics-based motion with configurable speed (default: 5 cm/sec)
- Canvas size of 20×20 cm (configurable)
- Video output showing the drawing progress with timer
- Independent vs. direct motion mode comparison
- Pen position tracking (green for pen-up, red for pen-down)
- Detailed performance metrics

## Comparing Algorithms

The algorithm comparison features:
1. **Efficiency Metrics**: Drawing ratio, travel distance, execution time
2. **Visualization**: Comparative charts of all algorithms
3. **Path Visualization**: Side-by-side comparison of different path plans

The most efficient algorithms in terms of minimizing pen travel distance are typically:
1. Simulated Annealing (highest quality but slowest)
2. 2-Opt Local Search (good balance of quality and speed)
3. Nearest Insertion (faster with good results)
4. KD-Tree Nearest Neighbor (efficient for larger datasets)

## Adding New Algorithms

To add a new algorithm:

1. Create a new class that inherits from `PlotterAlgorithm` in `algorithms.py`
2. Implement the `process_segments` method
3. Set the `name` attribute in the constructor
4. Add your new algorithm to the list in `algorithm_processor.py`

Example:
```python
class MyNewAlgorithm(PlotterAlgorithm):
    def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080)):
        super().__init__(canvas_size)
        self.name = "my_new_algorithm"
    
    def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
        # Your path planning logic here
        instructions = []
        # ...
        return instructions
```

## Example Custom Input File

Here's an example of a simple custom input file with 3 line segments forming a triangle:

```json
[
  [[0, 0], [100, 100]],
  [[100, 100], [200, 0]],
  [[200, 0], [0, 0]]
]
```

Save this as `input/triangle.json` to create a simple triangle drawing.