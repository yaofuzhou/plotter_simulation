import math
from typing import List, Tuple, Dict, Any, Optional

# Type definitions for clarity
Point = Tuple[float, float]
LineSegment = Tuple[Point, Point]
Instruction = Dict[str, Any]  # Will contain type and points

class PlotterAlgorithm:
    """Base class for plotter path planning algorithms"""
    
    def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080)):
        self.canvas_size = canvas_size
        self.current_position = (0, 0)  # Starting at upper left corner (0,0)
        self.pen_down = False
        self.name = "base"  # Algorithm name for file naming
    
    def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
        """Process line segments into plotter instructions"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def move_to(self, point: Point, pen_down: bool) -> Instruction:
        """Create a move instruction to the given point"""
        euclidean_distance = self._calculate_euclidean_distance(self.current_position, point)
        manhattan_distance = self._calculate_manhattan_distance(self.current_position, point)
        
        instruction = {
            "type": "PEN_DOWN" if pen_down else "PEN_UP",
            "from": self.current_position,
            "to": point,
            "euclidean_distance": euclidean_distance,
            "manhattan_distance": manhattan_distance
        }
        self.current_position = point
        self.pen_down = pen_down
        return instruction

    def _calculate_euclidean_distance(self, point1: Point, point2: Point) -> float:
        """Calculate Euclidean (L2) distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def _calculate_manhattan_distance(self, point1: Point, point2: Point) -> float:
        """Calculate Manhattan (L1) distance between two points"""
        return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])


class BruteForceAlgorithm(PlotterAlgorithm):
    """Naive algorithm that simply processes segments in the order they appear"""
    
    def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080)):
        super().__init__(canvas_size)
        self.name = "brute_force"
    
    def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
        instructions = []
        
        # Start with pen up if not already at the origin (0,0)
        if self.current_position != (0, 0):
            instructions.append(self.move_to((0, 0), False))
        
        for segment in segments:
            start_point, end_point = segment
            
            # Move to the start of the segment (pen up)
            instructions.append(self.move_to(start_point, False))
            
            # Draw the segment (pen down)
            instructions.append(self.move_to(end_point, True))
        
        # End with pen up
        if self.pen_down:
            instructions.append(self.move_to(self.current_position, False))
            
        return instructions

# Additional algorithms can be added here

class GreedyAlgorithm(PlotterAlgorithm):
    """
    Greedy algorithm that always moves to the nearest unprocessed segment.
    Uses grid-based spatial partitioning to optimize nearest neighbor search.
    Considers both endpoints of segments and can draw segments in reverse direction.
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080), grid_size: int = 10):
        super().__init__(canvas_size)
        self.name = "greedy"
        self.grid_size = grid_size
        
    def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
        instructions = []
        
        # Start with pen up at the origin (0,0)
        if self.current_position != (0, 0):
            instructions.append(self.move_to((0, 0), False))
        
        # Create a copy of segments to work with
        unprocessed_segments = segments.copy()
        
        # Create a grid-based spatial index for both start and end points
        grid_map = self._build_grid_index(unprocessed_segments)
        
        # Process segments until all are drawn
        while unprocessed_segments:
            # Find the nearest unprocessed segment endpoint
            nearest_data = self._find_nearest_endpoint(
                self.current_position, unprocessed_segments, grid_map)
            
            if nearest_data is None:
                # This should not happen but just in case
                break
                
            segment_idx, is_start = nearest_data
                
            # Get the nearest segment
            segment = unprocessed_segments.pop(segment_idx)
            start_point, end_point = segment
            
            # Remove this segment from the grid index
            self._remove_from_grid(segment, grid_map, segment_idx)
            
            # Draw the segment in the appropriate direction
            if is_start:
                # Draw from start to end
                instructions.append(self.move_to(start_point, False))
                instructions.append(self.move_to(end_point, True))
                self.current_position = end_point
            else:
                # Draw from end to start (reverse direction)
                instructions.append(self.move_to(end_point, False))
                instructions.append(self.move_to(start_point, True))
                self.current_position = start_point
        
        # End with pen up
        if self.pen_down:
            instructions.append(self.move_to(self.current_position, False))
            
        return instructions
    
    def _build_grid_index(self, segments: List[LineSegment]) -> Dict[Tuple[int, int], List[Tuple[int, bool, Point]]]:
        """
        Build a grid-based spatial index for faster nearest neighbor search.
        Indexes both start and end points of each segment.
        
        Returns:
            Dict mapping grid cell (row, col) to list of (segment_idx, is_start, point) tuples
        """
        grid_map = {}
        width, height = self.canvas_size
        cell_width = width / self.grid_size
        cell_height = height / self.grid_size
        
        for i, segment in enumerate(segments):
            # Add start point to the grid
            start_point = segment[0]
            start_row = min(int(start_point[1] / cell_height), self.grid_size - 1)
            start_col = min(int(start_point[0] / cell_width), self.grid_size - 1)
            start_cell = (start_row, start_col)
            
            if start_cell not in grid_map:
                grid_map[start_cell] = []
            
            grid_map[start_cell].append((i, True, start_point))
            
            # Add end point to the grid
            end_point = segment[1]
            end_row = min(int(end_point[1] / cell_height), self.grid_size - 1)
            end_col = min(int(end_point[0] / cell_width), self.grid_size - 1)
            end_cell = (end_row, end_col)
            
            if end_cell not in grid_map:
                grid_map[end_cell] = []
            
            grid_map[end_cell].append((i, False, end_point))
            
        return grid_map
    
    def _remove_from_grid(self, segment: LineSegment, 
                         grid_map: Dict[Tuple[int, int], List[Tuple[int, bool, Point]]], 
                         removed_idx: int):
        """
        Remove a segment from the grid index and update remaining indices.
        
        When a segment is removed, all segments with higher indices need to be decremented.
        Both start and end points of the segment are removed from the grid.
        """
        width, height = self.canvas_size
        cell_width = width / self.grid_size
        cell_height = height / self.grid_size
        
        # Get start and end points
        start_point, end_point = segment
        
        # Calculate grid cells for both points
        points_cells = []
        
        # Start point grid cell
        start_row = min(int(start_point[1] / cell_height), self.grid_size - 1)
        start_col = min(int(start_point[0] / cell_width), self.grid_size - 1)
        points_cells.append(((start_row, start_col), True))
        
        # End point grid cell
        end_row = min(int(end_point[1] / cell_height), self.grid_size - 1)
        end_col = min(int(end_point[0] / cell_width), self.grid_size - 1)
        points_cells.append(((end_row, end_col), False))
        
        # Update all grid cells
        for cell, points in list(grid_map.items()):
            updated_points = []
            for idx, is_start, point in points:
                if idx == removed_idx:
                    # Skip this point as it's being removed
                    continue
                elif idx > removed_idx:
                    # Decrement indices for points after the removed one
                    updated_points.append((idx - 1, is_start, point))
                else:
                    # Keep points with lower indices as is
                    updated_points.append((idx, is_start, point))
            
            if updated_points:
                grid_map[cell] = updated_points
            else:
                # Remove empty cells
                del grid_map[cell]
    
    def _find_nearest_endpoint(self, 
                             current_pos: Point, 
                             segments: List[LineSegment], 
                             grid_map: Dict[Tuple[int, int], List[Tuple[int, bool, Point]]]) -> Optional[Tuple[int, bool]]:
        """
        Find the nearest unprocessed segment endpoint from the current position.
        Searches neighboring grid cells first for efficiency.
        
        Returns:
            Tuple of (segment_index, is_start_point) or None if none found
        """
        if not segments:
            return None
        
        width, height = self.canvas_size
        cell_width = width / self.grid_size
        cell_height = height / self.grid_size
        
        # Find the grid cell containing the current position
        current_row = min(int(current_pos[1] / cell_height), self.grid_size - 1)
        current_col = min(int(current_pos[0] / cell_width), self.grid_size - 1)
        
        # Search in expanding rings of grid cells around the current cell
        max_radius = max(self.grid_size, self.grid_size)  # Ensure we cover the entire grid
        nearest_idx = None
        nearest_is_start = None
        nearest_distance = float('inf')
        
        for radius in range(max_radius + 1):
            # If we've found a point in an inner ring and the manhattan distance to the
            # next ring is greater than our current best, we can stop searching
            if nearest_idx is not None and nearest_distance < radius * min(cell_width, cell_height):
                break
            
            # Generate cells at this radius from current_row, current_col
            cells_to_check = self._get_cells_at_radius(current_row, current_col, radius)
            
            # Check each cell for the nearest point
            for cell_row, cell_col in cells_to_check:
                # Skip invalid cells
                if cell_row < 0 or cell_row >= self.grid_size or cell_col < 0 or cell_col >= self.grid_size:
                    continue
                
                cell = (cell_row, cell_col)
                if cell in grid_map:
                    # Check all points in this cell
                    for segment_idx, is_start, point in grid_map[cell]:
                        # Verify index is in range (defensive programming)
                        if segment_idx < 0 or segment_idx >= len(segments):
                            continue
                            
                        distance = self._calculate_euclidean_distance(current_pos, point)
                        if distance < nearest_distance:
                            nearest_idx = segment_idx
                            nearest_is_start = is_start
                            nearest_distance = distance
        
        if nearest_idx is None:
            return None
        return (nearest_idx, nearest_is_start)
    
    def _get_cells_at_radius(self, center_row: int, center_col: int, radius: int) -> List[Tuple[int, int]]:
        """Get all grid cells at a given Manhattan distance (radius) from the center cell"""
        if radius == 0:
            return [(center_row, center_col)]
        
        cells = []
        
        # Top and bottom edges
        for dc in range(-radius, radius + 1):
            cells.append((center_row - radius, center_col + dc))
            cells.append((center_row + radius, center_col + dc))
        
        # Left and right edges (excluding corners already covered)
        for dr in range(-radius + 1, radius):
            cells.append((center_row + dr, center_col - radius))
            cells.append((center_row + dr, center_col + radius))
        
        return cells



class KDTreeNearestNeighborAlgorithm(PlotterAlgorithm):
    """
    KD-Tree based nearest neighbor algorithm for optimizing plotter paths.
    
    This algorithm:
    1. Builds a KD-Tree to efficiently find nearest neighbors in 2D space
    2. Uses the tree to quickly identify the closest unprocessed segment endpoint
    3. Can draw segments in reverse direction to minimize travel
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080)):
        super().__init__(canvas_size)
        self.name = "KD Tree Nearest Neighbor"
    
    class KDNode:
        """Node in a KD-Tree"""
        def __init__(self, point: Point, segment_idx: int, is_start: bool, axis: int = 0):
            self.point = point               # The point in 2D space
            self.segment_idx = segment_idx   # Index of the segment this point belongs to
            self.is_start = is_start         # Whether this is the start (True) or end (False) point
            self.axis = axis                 # Split axis (0 for x, 1 for y)
            self.left = None                 # Left child (points with smaller value on axis)
            self.right = None                # Right child (points with larger value on axis)
    
    def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
        if not segments:
            return []
            
        instructions = []
        
        # Start with pen up at the origin (0,0)
        if self.current_position != (0, 0):
            instructions.append(self.move_to((0, 0), False))
        
        # Build KD-Tree from both start and end points of segments
        points = []
        for i, segment in enumerate(segments):
            points.append((segment[0], i, True))   # Start point
            points.append((segment[1], i, False))  # End point
            
        root = self._build_kdtree(points)
        
        # Track which segments have been processed
        processed = [False] * len(segments)
        current_pos = (0, 0)
        
        # Process segments in nearest-neighbor order
        for _ in range(len(segments)):
            # Find the nearest unprocessed segment endpoint
            nearest_data = self._find_nearest(root, current_pos, processed)
            
            if nearest_data is None:
                break
                
            nearest_idx, is_start_point = nearest_data
            
            # Mark segment as processed
            processed[nearest_idx] = True
            
            # Get the segment
            segment = segments[nearest_idx]
            
            # Draw the segment in appropriate direction
            if is_start_point:
                # Draw from start to end (normal direction)
                start_point, end_point = segment
            else:
                # Draw from end to start (reversed direction)
                end_point, start_point = segment
                
            # Move to the first endpoint (pen up)
            instructions.append(self.move_to(start_point, False))
            
            # Draw to the second endpoint (pen down)
            instructions.append(self.move_to(end_point, True))
            
            # Update current position
            current_pos = end_point
        
        # End with pen up
        if self.pen_down:
            instructions.append(self.move_to(self.current_position, False))
            
        return instructions
    
    def _build_kdtree(self, points: List[Tuple[Point, int, bool]], depth: int = 0) -> Optional[KDNode]:
        """
        Recursively build a KD-Tree from a list of points.
        
        Args:
            points: List of (point, segment_idx, is_start) tuples
            depth: Current depth in the tree
        
        Returns:
            Root node of the KD-Tree
        """
        if not points:
            return None
            
        # Select axis based on depth (cycle through x and y)
        axis = depth % 2
        
        # Sort points by the axis value
        points.sort(key=lambda x: x[0][axis])
        
        # Find median point
        median_idx = len(points) // 2
        
        # Create node and construct subtrees
        point, seg_idx, is_start = points[median_idx]
        node = self.KDNode(point, seg_idx, is_start, axis)
        node.left = self._build_kdtree(points[:median_idx], depth + 1)
        node.right = self._build_kdtree(points[median_idx + 1:], depth + 1)
        
        return node
    
    def _find_nearest(self, root: KDNode, query_point: Point, processed: List[bool]) -> Optional[Tuple[int, bool]]:
        """
        Find the nearest unprocessed segment endpoint in the KD-Tree.
        
        Args:
            root: Root node of the KD-Tree
            query_point: Point to find the nearest neighbor for
            processed: Boolean array indicating which segments have been processed
        
        Returns:
            Tuple of (segment_index, is_start_point) or None if none found
        """
        # Use a best-bin-first approach for searching the KD-Tree
        best_idx = None
        best_is_start = None
        best_dist = float('inf')
        
        # Use a priority queue for best-bin-first traversal
        import heapq
        counter = 0
        queue = [(0, counter, root)]  # (distance_bound, unique_id, node)
        
        while queue and (best_dist == float('inf') or queue[0][0] < best_dist):
            _, _, node = heapq.heappop(queue)
            
            if node is None:
                continue
                
            # Check if this node's segment is unprocessed
            if not processed[node.segment_idx]:
                dist = self._calculate_euclidean_distance(query_point, node.point)
                if dist < best_dist:
                    best_idx = node.segment_idx
                    best_is_start = node.is_start
                    best_dist = dist
            
            # Determine which child to visit first (the one more likely to contain nearest neighbor)
            axis = node.axis
            if query_point[axis] < node.point[axis]:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
            
            # Distance bound for second branch
            axis_dist = abs(query_point[axis] - node.point[axis])
            
            # Add children to queue with appropriate bounds
            if first is not None:
                counter += 1
                heapq.heappush(queue, (0, counter, first))  # First branch could have points with zero distance
            
            if second is not None:
                counter += 1
                heapq.heappush(queue, (axis_dist, counter, second))  # Second branch has minimum distance of axis_dist
        
        if best_idx is None:
            return None
        return (best_idx, best_is_start)


class RTreeNearestNeighborAlgorithm(PlotterAlgorithm):
    """
    R-Tree based nearest neighbor algorithm for optimizing plotter paths.
    
    This algorithm:
    1. Builds an R-Tree to efficiently find nearest neighbors in 2D space
    2. Considers both endpoints of each segment when finding the nearest neighbor
    3. Can reverse segment direction to minimize total travel distance
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080), max_entries: int = 5):
        super().__init__(canvas_size)
        self.name = "R Tree Nearest Neighbor"
        self.max_entries = max_entries
    
    class RTreeNode:
        """Node in an R-Tree"""
        def __init__(self, is_leaf: bool = True):
            self.is_leaf = is_leaf          # Whether this is a leaf node
            self.entries = []                # Entries (children or data points)
            self.mbr = None                  # Minimum Bounding Rectangle (x_min, y_min, x_max, y_max)
        
        def update_mbr(self):
            """Update the minimum bounding rectangle to contain all entries"""
            if not self.entries:
                self.mbr = None
                return
                
            if self.is_leaf:
                # Explicitly unpack each entry to avoid indexing confusion
                x_coords = []
                y_coords = []
                for entry in self.entries:
                    point, _, _ = entry  # Unpack each entry (point, segment_idx, is_start)
                    x_coords.append(point[0])  # x-coordinate
                    y_coords.append(point[1])  # y-coordinate
            else:
                # Non-leaf entries are child nodes
                x_mins = [child.mbr[0] for child in self.entries]
                y_mins = [child.mbr[1] for child in self.entries]
                x_maxs = [child.mbr[2] for child in self.entries]
                y_maxs = [child.mbr[3] for child in self.entries]
                x_coords = x_mins + x_maxs
                y_coords = y_mins + y_maxs
            
            self.mbr = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
        if not segments:
            return []
            
        instructions = []
        
        # Start with pen up at the origin (0,0)
        if self.current_position != (0, 0):
            instructions.append(self.move_to((0, 0), False))
        
        # Build R-Tree from both start and end points of segments
        points = []
        for i, segment in enumerate(segments):
            points.append((segment[0], i, True))   # Start point
            points.append((segment[1], i, False))  # End point
            
        root = self._build_rtree(points)
        
        # Track which segments have been processed
        processed = [False] * len(segments)
        current_pos = (0, 0)
        
        # Process segments in nearest-neighbor order
        for _ in range(len(segments)):
            # Find the nearest unprocessed segment endpoint
            nearest_data = self._find_nearest(root, current_pos, processed)
            
            if nearest_data is None:
                break
                
            nearest_idx, is_start_point = nearest_data
            
            # Mark segment as processed
            processed[nearest_idx] = True
            
            # Get the segment
            segment = segments[nearest_idx]
            
            # Draw the segment in appropriate direction
            if is_start_point:
                # Draw from start to end (normal direction)
                start_point, end_point = segment
            else:
                # Draw from end to start (reversed direction)
                end_point, start_point = segment
            
            # Move to the first endpoint (pen up)
            instructions.append(self.move_to(start_point, False))
            
            # Draw to the second endpoint (pen down)
            instructions.append(self.move_to(end_point, True))
            
            # Update current position
            current_pos = end_point
        
        # End with pen up
        if self.pen_down:
            instructions.append(self.move_to(self.current_position, False))
            
        return instructions
    
    def _build_rtree(self, points: List[Tuple[Point, int, bool]]) -> RTreeNode:
        """
        Build an R-Tree from a list of points.
        
        Args:
            points: List of (point, segment_idx, is_start) tuples
        
        Returns:
            Root node of the R-Tree
        """
        # Start with a leaf node containing all points
        leaf = self.RTreeNode(is_leaf=True)
        leaf.entries = points
        leaf.update_mbr()
        
        # If the node is small enough, we're done
        if len(points) <= self.max_entries:
            return leaf
        
        # Otherwise, split the node and create a non-leaf parent
        root = self.RTreeNode(is_leaf=False)
        root.entries = self._split_node(leaf)
        root.update_mbr()
        
        return root
    
    def _split_node(self, node: RTreeNode) -> List[RTreeNode]:
        """
        Split a node that has too many entries into multiple nodes.
        Uses a simple quadratic split algorithm.
        
        Returns:
            List of new nodes after splitting
        """
        entries = node.entries
        
        # For simplicity, just split into equal parts based on x-coordinate
        if node.is_leaf:
            # Sort leaf entries by x-coordinate of point
            entries.sort(key=lambda entry: entry[0][0])
        else:
            # Sort non-leaf entries by center x-coordinate of MBR
            entries.sort(key=lambda child: (child.mbr[0] + child.mbr[2]) / 2)
        
        # Create new nodes with roughly equal numbers of entries
        result = []
        chunk_size = max(1, len(entries) // self.max_entries)
        
        for i in range(0, len(entries), chunk_size):
            new_node = self.RTreeNode(is_leaf=node.is_leaf)
            new_node.entries = entries[i:i+chunk_size]
            new_node.update_mbr()
            
            # If this is a non-leaf node and has too many entries, recursively split
            if not new_node.is_leaf and len(new_node.entries) > self.max_entries:
                result.extend(self._split_node(new_node))
            else:
                result.append(new_node)
        
        return result
    
    def _find_nearest(self, root: RTreeNode, query_point: Point, processed: List[bool]) -> Optional[Tuple[int, bool]]:
        """
        Find the nearest unprocessed segment endpoint in the R-Tree.
        
        Args:
            root: Root node of the R-Tree
            query_point: Point to find the nearest neighbor for
            processed: Boolean array indicating which segments have been processed
        
        Returns:
            Tuple of (segment_index, is_start_point) or None if none found
        """
        # Priority queue for branch-and-bound search
        import heapq
        counter = 0
        queue = [(self._min_dist_to_rectangle(query_point, root.mbr), counter, root)]
        
        best_idx = None
        best_is_start = None
        best_dist = float('inf')
        
        while queue and (best_dist == float('inf') or queue[0][0] < best_dist):
            _, _, node = heapq.heappop(queue)
            
            if node.is_leaf:
                # Check all points in this leaf
                for entry in node.entries:
                    point, segment_idx, is_start = entry  # Correctly unpack the entry
                    if not processed[segment_idx]:
                        dist = self._calculate_euclidean_distance(query_point, point)
                        if dist < best_dist:
                            best_idx = segment_idx
                            best_is_start = is_start
                            best_dist = dist
            else:
                # Add child nodes to queue
                for child in node.entries:
                    # Calculate minimum possible distance to this child's MBR
                    min_dist = self._min_dist_to_rectangle(query_point, child.mbr)
                    
                    # Only add to queue if it could contain a closer point
                    if min_dist < best_dist:
                        counter += 1
                        heapq.heappush(queue, (min_dist, counter, child))
        
        if best_idx is None:
            return None
        return (best_idx, best_is_start)
    
    def _min_dist_to_rectangle(self, point: Point, rectangle: Tuple[float, float, float, float]) -> float:
        """
        Calculate the minimum distance from a point to a rectangle.
        
        Args:
            point: Query point (x, y)
            rectangle: Rectangle as (min_x, min_y, max_x, max_y)
        
        Returns:
            Minimum possible distance
        """
        if rectangle is None:
            return float('inf')
            
        x, y = point
        min_x, min_y, max_x, max_y = rectangle
        
        # Find closest x-coordinate
        if x < min_x:
            dx = min_x - x
        elif x > max_x:
            dx = x - max_x
        else:
            dx = 0
        
        # Find closest y-coordinate
        if y < min_y:
            dy = min_y - y
        elif y > max_y:
            dy = y - max_y
        else:
            dy = 0
        
        # Euclidean distance
        return math.sqrt(dx*dx + dy*dy)



####### Backup algorithms #######


# class NearestInsertionAlgorithm(PlotterAlgorithm):
#     """
#     Nearest Insertion algorithm for optimizing plotter paths.
    
#     This algorithm treats the problem as a modified Traveling Salesman Problem by:
#     1. Starting with a minimal tour (just a single segment)
#     2. Iteratively inserting each remaining segment at the position that minimizes total distance
#     3. Considering both the travel distance and the orientation of segments
#     """
    
#     def __init__(self, canvas_size: Tuple[int, int] = (1080, 1080)):
#         super().__init__(canvas_size)
#         self.name = "nearest_insertion"
    
#     def process_segments(self, segments: List[LineSegment]) -> List[Instruction]:
#         if not segments:
#             return []
            
#         instructions = []
        
#         # Start with pen up at the origin (0,0)
#         if self.current_position != (0, 0):
#             instructions.append(self.move_to((0, 0), False))
        
#         # Make a copy of segments to work with
#         remaining_segments = segments.copy()
        
#         # Start with the segment closest to current position (0,0)
#         start_idx, start_dist = self._find_nearest_segment_start(self.current_position, remaining_segments)
#         current_tour = [remaining_segments.pop(start_idx)]
        
#         # Build tour by nearest insertion
#         while remaining_segments:
#             best_segment_idx, best_insertion_pos, best_cost = self._find_best_insertion(
#                 current_tour, remaining_segments)
            
#             if best_segment_idx is None:
#                 break
                
#             # Insert the segment at the best position
#             current_tour.insert(best_insertion_pos, remaining_segments.pop(best_segment_idx))
        
#         # Convert the optimized tour to plotter instructions
#         for segment in current_tour:
#             start_point, end_point = segment
            
#             # Move to the start of the segment (pen up)
#             instructions.append(self.move_to(start_point, False))
            
#             # Draw the segment (pen down)
#             instructions.append(self.move_to(end_point, True))
        
#         # End with pen up
#         if self.pen_down:
#             instructions.append(self.move_to(self.current_position, False))
            
#         return instructions
    
#     def _find_nearest_segment_start(self, point: Point, segments: List[LineSegment]) -> Tuple[Optional[int], float]:
#         """Find the segment with the start point closest to the given point"""
#         if not segments:
#             return None, float('inf')
            
#         best_idx = None
#         best_dist = float('inf')
        
#         for i, segment in enumerate(segments):
#             start_point = segment[0]
#             dist = self._calculate_euclidean_distance(point, start_point)
#             if dist < best_dist:
#                 best_idx = i
#                 best_dist = dist
                
#         return best_idx, best_dist
    
#     def _find_best_insertion(self, 
#                            tour: List[LineSegment], 
#                            segments: List[LineSegment]) -> Tuple[Optional[int], Optional[int], float]:
#         """
#         Find the best segment and position to insert into the tour.
        
#         Returns:
#             Tuple of (segment_index, insertion_position, insertion_cost)
#         """
#         if not segments or not tour:
#             return None, None, float('inf')
            
#         best_segment_idx = None
#         best_insertion_pos = None
#         best_cost = float('inf')
        
#         # For each unvisited segment
#         for i, segment in enumerate(segments):
#             # Try inserting at each position in the tour
#             for j in range(len(tour) + 1):
#                 # Calculate the cost of this insertion
#                 cost = self._calculate_insertion_cost(tour, j, segment)
#                 if cost < best_cost:
#                     best_segment_idx = i
#                     best_insertion_pos = j
#                     best_cost = cost
        
#         return best_segment_idx, best_insertion_pos, best_cost
    
#     def _calculate_insertion_cost(self, tour: List[LineSegment], pos: int, segment: LineSegment) -> float:
#         """
#         Calculate the cost of inserting a segment at a specific position in the tour.
        
#         Cost considers both travel distance and the orientation of segments.
#         """
#         # Handle edge cases
#         if not tour:
#             # If tour is empty, cost is just distance from origin to segment start
#             return self._calculate_euclidean_distance((0, 0), segment[0])
        
#         # Get points before and after insertion
#         if pos == 0:
#             # Inserting at beginning
#             next_segment = tour[0]
#             prev_point = (0, 0) if self.current_position == (0, 0) else self.current_position
#             next_point = next_segment[0]
#         elif pos == len(tour):
#             # Inserting at end
#             prev_segment = tour[-1]
#             prev_point = prev_segment[1]  # End of last segment
#             next_point = (0, 0)  # Return to origin
#         else:
#             # Inserting in middle
#             prev_segment = tour[pos-1]
#             next_segment = tour[pos]
#             prev_point = prev_segment[1]  # End of previous segment
#             next_point = next_segment[0]  # Start of next segment
        
#         # Calculate cost:
#         # 1. Distance from previous point to segment start
#         # 2. Distance from segment end to next point
#         # 3. Minus the original distance from prev to next (which we're replacing)
#         original_dist = self._calculate_euclidean_distance(prev_point, next_point)
#         new_dist = (self._calculate_euclidean_distance(prev_point, segment[0]) + 
#                    self._calculate_euclidean_distance(segment[1], next_point))
        
#         # Cost is the extra travel distance added by this insertion
#         return new_dist - original_dist
