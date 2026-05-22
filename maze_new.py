"""
=============================================================
Maze Graph Algorithms - Final Group Project
=============================================================
Students: Ilaha Habibova, Ngoc Bao Tram Tran, Kamalesh Ashokkumar Kowsalya
Student IDs: 231ADB231, 231ADB294, 221ADB216
Group: [Your Group Number]
Language: Python 3.8+

Instructions for running:
  1. Save the required maze as maze_10x10_A.txt in the same folder.
  2. Run the full required solution with all cost-model extension:
       python maze.py maze_10x10_A.txt --all-cost-models --output maze_10x10_A_output.txt
  3. Optional JSON export:
       python maze.py maze_10x10_A.txt --all-cost-models --export-json
  4. Optional GraphML export:
       python maze.py maze_10x10_A.txt --export-graphml
  5. Optional benchmark on generated mazes:
       python maze.py maze_10x10_A.txt --benchmark
       python maze.py maze_10x10_A.txt --benchmark 50x50
       python maze.py maze_10x10_A.txt --benchmark 10,20,50,100

Notes:
  - The program supports mazes up to 100x100.
  - S and G have numeric value 0.
  - For 8-directional movement, diagonal moves count as one move.
  - Diagonal movement is allowed if the destination cell is valid and passable.
    No additional corner-cutting restriction is applied.
=============================================================
"""

import argparse
import heapq
import json
import random
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


Position = Tuple[int, int]
Grid = List[str]
PathType = List[Position]

ALLOWED_CHARS = set("SGX0123456789")
COST_MODEL_NAMES = {
    1: "entering (cost = value(v))",
    2: "leaving  (cost = value(u))",
    3: "combined (cost = value(u) + value(v))",
}


##############################
# MAZE PARSING AND VALIDATION
##############################
def parse_maze(filepath: str) -> Tuple[Grid, int, int, Position, Position]:
    """
    Read and validate a maze from a plain text file.

    Returns:
      grid  - list of strings
      rows  - number of rows
      cols  - number of columns
      start - coordinates of S as (row, column)
      goal  - coordinates of G as (row, column)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Maze file not found: {filepath}")

    with path.open("r", encoding="utf-8") as file:
        # Empty lines are ignored. Characters inside non-empty rows are preserved.
        grid = [line.rstrip("\n") for line in file if line.rstrip("\n") != ""]

    return validate_maze(grid)


def validate_maze(grid: Grid) -> Tuple[Grid, int, int, Position, Position]:
    """
    Validate the required input rules:
      - maze is not empty
      - all rows have the same length
      - size is at most 100x100
      - characters are only S, G, X, or digits 0-9
      - exactly one S and exactly one G exist
    """
    if not grid:
        raise ValueError("Maze is empty.")

    rows = len(grid)
    cols = len(grid[0])

    if cols == 0:
        raise ValueError("Maze has an empty first row.")

    if rows > 100 or cols > 100:
        raise ValueError(f"Maze size is {rows}x{cols}, but the limit is 100x100.")

    start_positions: List[Position] = []
    goal_positions: List[Position] = []

    for r, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError(
                f"All rows must have the same length. Row 0 has length {cols}, "
                f"but row {r} has length {len(row)}."
            )

        for c, ch in enumerate(row):
            if ch not in ALLOWED_CHARS:
                raise ValueError(
                    f"Invalid character '{ch}' at position ({r},{c}). "
                    "Allowed characters are S, G, X, and digits 0-9."
                )
            if ch == "S":
                start_positions.append((r, c))
            elif ch == "G":
                goal_positions.append((r, c))

    if len(start_positions) != 1:
        raise ValueError(f"Maze must contain exactly one S. Found {len(start_positions)}.")
    if len(goal_positions) != 1:
        raise ValueError(f"Maze must contain exactly one G. Found {len(goal_positions)}.")

    return grid, rows, cols, start_positions[0], goal_positions[0]


def cell_value(grid: Grid, r: int, c: int) -> Optional[int]:
    """
    Return the numeric value of a cell.

    Rules:
      S -> 0
      G -> 0
      digit -> integer value
      X -> None because it is a wall
    """
    ch = grid[r][c]
    if ch in {"S", "G"}:
        return 0
    if ch == "X":
        return None
    return int(ch)


def is_passable(grid: Grid, rows: int, cols: int, r: int, c: int) -> bool:
    """Return True if the cell is inside the maze and is not a wall."""
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != "X"


def neighbors(grid: Grid, rows: int, cols: int, r: int, c: int, mode: str = "4") -> Iterable[Position]:
    """
    Generate valid neighboring cells.

    mode = '4': up, down, left, right
    mode = '8': four-directional moves plus diagonals
    """
    directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    directions_8 = directions_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions = directions_8 if mode == "8" else directions_4

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if is_passable(grid, rows, cols, nr, nc):
            yield nr, nc


def count_vertices_edges(grid: Grid, rows: int, cols: int, mode: str) -> Tuple[int, int]:
    """Return number of passable vertices and directed neighbor relations."""
    vertices = 0
    edges = 0
    for r in range(rows):
        for c in range(cols):
            if is_passable(grid, rows, cols, r, c):
                vertices += 1
                edges += sum(1 for _ in neighbors(grid, rows, cols, r, c, mode))
    return vertices, edges


def path_to_str(path: PathType) -> str:
    """Convert a path list into the required readable format."""
    if not path:
        return "UNREACHABLE"
    return "->".join(f"({r},{c})" for r, c in path)


def format_maze_with_path(grid: Grid, path: PathType) -> List[str]:
    """Return a maze visualization where path cells are marked with '*'."""
    path_set = set(path)
    formatted: List[str] = []
    for r, row in enumerate(grid):
        new_row = []
        for c, ch in enumerate(row):
            new_row.append("*" if (r, c) in path_set else ch)
        formatted.append("".join(new_row))
    return formatted


##############################
# PATH COST HELPERS
##############################

def edge_cost(grid: Grid, u: Position, v: Position, model: int) -> int:
    """
    Return the cost of moving from u to v.

    Cost Model 1: entering cost  -> cost(u,v) = value(v)
    Cost Model 2: leaving cost   -> cost(u,v) = value(u)
    Cost Model 3: combined cost  -> cost(u,v) = value(u) + value(v)
    """
    ur, uc = u
    vr, vc = v
    u_value = cell_value(grid, ur, uc)
    v_value = cell_value(grid, vr, vc)

    if u_value is None or v_value is None:
        raise ValueError("edge_cost() received a wall cell, which should not happen.")

    if model == 1:
        return v_value
    if model == 2:
        return u_value
    if model == 3:
        return u_value + v_value

    raise ValueError("Cost model must be 1, 2, or 3.")


def total_path_cost(grid: Grid, path: PathType, model: int) -> Optional[int]:
    """Calculate total cost of a path under the selected cost model."""
    if not path:
        return None
    total = 0
    for i in range(len(path) - 1):
        total += edge_cost(grid, path[i], path[i + 1], model)
    return total


##############################
# SUBTASK A - SHORTEST PATH BY NUMBER OF MOVES
##############################

def subtask_a(grid: Grid, rows: int, cols: int, start: Position, goal: Position, mode: str = "4") -> Tuple[Optional[int], PathType]:
    """
    Find the path from S to G with the smallest number of moves using BFS.

    BFS is suitable because every legal move has equal cost 1.

    Time complexity:  O(V + E)
    Space complexity: O(V)
    where V is the number of passable cells and E is the number of graph edges.
    """
    visited: Set[Position] = {start}
    parent: Dict[Position, Optional[Position]] = {start: None}
    queue: deque[Position] = deque([start])

    while queue:
        current = queue.popleft()

        if current == goal:
            path: PathType = []
            node: Optional[Position] = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return len(path) - 1, path

        r, c = current
        for nxt in neighbors(grid, rows, cols, r, c, mode):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = current
                queue.append(nxt)

    return None, []


##############################
# SUBTASK B - MINIMUM-COST PATH
##############################

def subtask_b(
    grid: Grid,
    rows: int,
    cols: int,
    start: Position,
    goal: Position,
    mode: str = "4",
    cost_model: int = 1,
) -> Tuple[Optional[int], PathType]:
    """
    Find the minimum-cost path from S to G using Dijkstra's algorithm.

    Dijkstra is suitable because all edge costs are non-negative.

    Time complexity:  O((V + E) log V)
    Space complexity: O(V + E) in general because the priority queue may contain
                      multiple entries. In a grid maze, E is proportional to V,
                      so this is often simplified to O(V).
    """
    distance: Dict[Position, int] = {start: 0}
    parent: Dict[Position, Optional[Position]] = {start: None}
    heap: List[Tuple[int, Position]] = [(0, start)]

    while heap:
        current_cost, current = heapq.heappop(heap)

        if current_cost > distance.get(current, float("inf")):
            continue

        if current == goal:
            path: PathType = []
            node: Optional[Position] = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return current_cost, path

        r, c = current
        for nxt in neighbors(grid, rows, cols, r, c, mode):
            new_cost = current_cost + edge_cost(grid, current, nxt, cost_model)
            if new_cost < distance.get(nxt, float("inf")):
                distance[nxt] = new_cost
                parent[nxt] = current
                heapq.heappush(heap, (new_cost, nxt))

    return None, []


##############################
# SUBTASK C - MOVEMENT MODE COMPARISON
##############################

def subtask_c(grid: Grid, rows: int, cols: int, start: Position, goal: Position) -> Dict[str, Dict[str, object]]:
    """
    Compare 4-directional and 8-directional movement for:
      - shortest path by number of moves
      - minimum-cost path using Cost Model 1, entering cost
    """
    results: Dict[str, Dict[str, object]] = {}

    for mode in ("4", "8"):
        moves, path_moves = subtask_a(grid, rows, cols, start, goal, mode)
        min_cost, path_cost = subtask_b(grid, rows, cols, start, goal, mode, cost_model=1)

        results[mode] = {
            "shortest_moves": moves,
            "path_moves": path_moves,
            "shortest_path_entering_cost": total_path_cost(grid, path_moves, 1),
            "min_cost": min_cost,
            "path_cost": path_cost,
            "min_cost_path_moves": len(path_cost) - 1 if path_cost else None,
        }

    return results


##############################
# SUBTASK D - MAXIMUM FLOW FROM G TO S
##############################

def build_flow_graph(
    grid: Grid,
    rows: int,
    cols: int,
    start: Position,
    goal: Position,
    mode: str = "4",
) -> Tuple[DefaultDict[Position, Set[Position]], DefaultDict[Position, DefaultDict[Position, int]]]:
    """
    Build a directed flow network.

    Vertices: all non-wall cells.
    Directed edges: u -> v for every allowed movement from u to v.
    Capacity rule: capacity(u,v) = value(v).
    Special cases: capacity into S = 100, capacity into G = 100.
    """
    special_capacity = 100
    adjacency: DefaultDict[Position, Set[Position]] = defaultdict(set)
    capacity: DefaultDict[Position, DefaultDict[Position, int]] = defaultdict(lambda: defaultdict(int))

    for r in range(rows):
        for c in range(cols):
            if not is_passable(grid, rows, cols, r, c):
                continue

            u = (r, c)
            for v in neighbors(grid, rows, cols, r, c, mode):
                if v == start or v == goal:
                    cap = special_capacity
                else:
                    v_value = cell_value(grid, v[0], v[1])
                    if v_value is None:
                        continue
                    cap = v_value

                if cap > 0:
                    capacity[u][v] += cap
                    capacity[v][u] += 0  # ensure reverse residual edge exists
                    adjacency[u].add(v)
                    adjacency[v].add(u)

    return adjacency, capacity


def bfs_augmenting_path(
    source: Position,
    sink: Position,
    adjacency: DefaultDict[Position, Set[Position]],
    residual: DefaultDict[Position, DefaultDict[Position, int]],
) -> Optional[Dict[Position, Optional[Position]]]:
    """Find an augmenting path in the residual graph using BFS."""
    visited: Set[Position] = {source}
    parent: Dict[Position, Optional[Position]] = {source: None}
    queue: deque[Position] = deque([source])

    while queue:
        u = queue.popleft()
        if u == sink:
            return parent

        for v in adjacency[u]:
            if v not in visited and residual[u][v] > 0:
                visited.add(v)
                parent[v] = u
                queue.append(v)

    return None


def subtask_d(
    grid: Grid,
    rows: int,
    cols: int,
    start: Position,
    goal: Position,
    mode: str = "4",
) -> Tuple[int, List[Tuple[Position, Position, int, int]]]:
    """
    Compute maximum flow from G to S using Edmonds-Karp.

    Source = G
    Sink   = S

    Time complexity:  O(V * E^2)
    Space complexity: O(V + E)
    """
    source = goal
    sink = start

    adjacency, capacity = build_flow_graph(grid, rows, cols, start, goal, mode)

    residual: DefaultDict[Position, DefaultDict[Position, int]] = defaultdict(lambda: defaultdict(int))
    for u in capacity:
        for v in capacity[u]:
            residual[u][v] = capacity[u][v]

    original_capacity: Dict[Position, Dict[Position, int]] = {u: dict(v_map) for u, v_map in capacity.items()}
    max_flow = 0

    while True:
        parent = bfs_augmenting_path(source, sink, adjacency, residual)
        if parent is None:
            break

        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            if u is None:
                raise RuntimeError("Broken parent chain in augmenting path.")
            path_flow = min(path_flow, residual[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            if u is None:
                raise RuntimeError("Broken parent chain in augmenting path.")
            residual[u][v] -= int(path_flow)
            residual[v][u] += int(path_flow)
            v = u

        max_flow += int(path_flow)

    positive_flow_edges: List[Tuple[Position, Position, int, int]] = []
    for u in original_capacity:
        for v, cap in original_capacity[u].items():
            flow = cap - residual[u][v]
            if flow > 0:
                positive_flow_edges.append((u, v, flow, cap))

    return max_flow, positive_flow_edges


##############################
# SUBTASK E - MINIMUM SPANNING TREE
##############################
def subtask_e(
    grid: Grid,
    rows: int,
    cols: int,
    start: Position,
    goal: Position,
    mode: str = "4",
) -> Tuple[int, int, int, List[Tuple[Position, Position, int]], bool]:
    """
    Compute the minimum spanning tree for the connected component containing S.

    The maze graph is treated as undirected.
    Edge weight between u and v is value(u) + value(v).

    Prim's algorithm avoids cycles by adding only vertices not already in the tree.

    Time complexity:  O((V + E) log V)
    Space complexity: O(V + E)
    """
    in_tree: Set[Position] = set()
    best_key: Dict[Position, int] = {start: 0}
    heap: List[Tuple[int, Position, Optional[Position]]] = [(0, start, None)]
    mst_edges: List[Tuple[Position, Position, int]] = []
    total_weight = 0

    while heap:
        weight, u, parent = heapq.heappop(heap)

        if u in in_tree:
            continue

        in_tree.add(u)

        if parent is not None:
            mst_edges.append((parent, u, weight))
            total_weight += weight

        for v in neighbors(grid, rows, cols, u[0], u[1], mode):
            if v in in_tree:
                continue

            u_value = cell_value(grid, u[0], u[1])
            v_value = cell_value(grid, v[0], v[1])
            if u_value is None or v_value is None:
                continue

            edge_weight = u_value + v_value
            if edge_weight < best_key.get(v, float("inf")):
                best_key[v] = edge_weight
                heapq.heappush(heap, (edge_weight, v, u))

    goal_reachable = goal in in_tree
    return total_weight, len(in_tree), len(mst_edges), mst_edges, goal_reachable


##############################
# EXTENSION - RANDOM MAZE GENERATOR
##############################

def generate_maze(rows: int, cols: int, wall_probability: float = 0.25, seed: Optional[int] = None) -> Grid:
    """
    Generate a random maze.

    A guaranteed path is carved along the top row and then down the right column.
    """
    rng = random.Random(seed)
    grid_as_lists: List[List[str]] = []

    for _ in range(rows):
        row = []
        for _ in range(cols):
            if rng.random() < wall_probability:
                row.append("X")
            else:
                row.append(str(rng.randint(1, 9)))
        grid_as_lists.append(row)

    grid_as_lists[0][0] = "S"
    grid_as_lists[rows - 1][cols - 1] = "G"

    # Carve a simple guaranteed path from S to G.
    for c in range(cols):
        if grid_as_lists[0][c] == "X":
            grid_as_lists[0][c] = str(rng.randint(1, 9))
    grid_as_lists[0][0] = "S"

    for r in range(rows):
        if grid_as_lists[r][cols - 1] == "X":
            grid_as_lists[r][cols - 1] = str(rng.randint(1, 9))
    grid_as_lists[rows - 1][cols - 1] = "G"

    return ["".join(row) for row in grid_as_lists]


def save_maze(grid: Grid, filepath: str) -> None:
    """Save a generated maze to a text file."""
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(grid) + "\n")


##############################
# EXTENSION - JSON AND GRAPHML EXPORT
##############################

def export_json(
    grid: Grid,
    rows: int,
    cols: int,
    start: Position,
    goal: Position,
    mode: str,
    path_a: PathType,
    path_b: PathType,
    flow_edges: List[Tuple[Position, Position, int, int]],
    mst_edges: List[Tuple[Position, Position, int]],
    filepath: str,
) -> None:
    """Export graph and selected results to JSON."""
    vertices = []
    for r in range(rows):
        for c in range(cols):
            if is_passable(grid, rows, cols, r, c):
                vertices.append(
                    {
                        "id": f"{r},{c}",
                        "row": r,
                        "col": c,
                        "value": cell_value(grid, r, c),
                        "type": grid[r][c],
                    }
                )

    edges = []
    seen: Set[Tuple[Position, Position]] = set()
    for r in range(rows):
        for c in range(cols):
            if not is_passable(grid, rows, cols, r, c):
                continue
            for nr, nc in neighbors(grid, rows, cols, r, c, mode):
                key = tuple(sorted([(r, c), (nr, nc)]))  # type: ignore[arg-type]
                if key not in seen:
                    seen.add(key)
                    weight = cell_value(grid, r, c) + cell_value(grid, nr, nc)  # type: ignore[operator]
                    edges.append({"u": f"{r},{c}", "v": f"{nr},{nc}", "weight": weight})

    data = {
        "maze": {"rows": rows, "cols": cols, "start": list(start), "goal": list(goal)},
        "movement": f"{mode}-directional",
        "vertices": vertices,
        "edges": edges,
        "subtask_a_path": [list(p) for p in path_a],
        "subtask_b_path": [list(p) for p in path_b],
        "positive_flow_edges": [[list(u), list(v), flow, cap] for u, v, flow, cap in flow_edges],
        "mst_edges": [[list(u), list(v), weight] for u, v, weight in mst_edges],
    }

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"[export] JSON written to {filepath}")


def export_graphml(grid: Grid, rows: int, cols: int, mode: str, filepath: str) -> None:
    """Export the maze graph to GraphML for tools such as Gephi or yEd."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
        '  <key id="val" for="node" attr.name="value" attr.type="int"/>',
        '  <key id="wt" for="edge" attr.name="weight" attr.type="int"/>',
        '  <graph id="maze" edgedefault="undirected">',
    ]

    for r in range(rows):
        for c in range(cols):
            if is_passable(grid, rows, cols, r, c):
                value = cell_value(grid, r, c)
                lines.append(f'    <node id="{r}_{c}"><data key="val">{value}</data></node>')

    edge_id = 0
    seen: Set[Tuple[Position, Position]] = set()
    for r in range(rows):
        for c in range(cols):
            if not is_passable(grid, rows, cols, r, c):
                continue
            for nr, nc in neighbors(grid, rows, cols, r, c, mode):
                key = tuple(sorted([(r, c), (nr, nc)]))  # type: ignore[arg-type]
                if key not in seen:
                    seen.add(key)
                    weight = cell_value(grid, r, c) + cell_value(grid, nr, nc)  # type: ignore[operator]
                    lines.append(
                        f'    <edge id="e{edge_id}" source="{r}_{c}" target="{nr}_{nc}">'
                        f'<data key="wt">{weight}</data></edge>'
                    )
                    edge_id += 1

    lines.extend(["  </graph>", "</graphml>"])

    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"[export] GraphML written to {filepath}")


##############################
# EXTENSION - BENCHMARKING
##############################

def parse_benchmark_sizes(size_text: str) -> List[int]:
    """
    Parse benchmark sizes.

    Accepted examples:
      '50x50'
      '50'
      '10,20,50,100'
    """
    sizes: List[int] = []
    parts = [part.strip().lower() for part in size_text.split(",") if part.strip()]

    for part in parts:
        if "x" in part:
            left, right = part.split("x", 1)
            rows = int(left)
            cols = int(right)
            if rows != cols:
                raise ValueError("Benchmark currently supports square sizes only, such as 50x50.")
            size = rows
        else:
            size = int(part)

        if size < 2 or size > 100:
            raise ValueError("Benchmark sizes must be between 2 and 100.")
        sizes.append(size)

    if not sizes:
        raise ValueError("No valid benchmark sizes provided.")

    return sizes


def benchmark(sizes: List[int]) -> None:
    """Run all main algorithms on randomly generated square mazes and print timing results."""
    print("\n" + "=" * 70)
    print("BENCHMARK ON GENERATED MAZES")
    print("=" * 70)
    print(f"{'Size':>8}  {'A BFS(ms)':>10}  {'B Dijkstra(ms)':>15}  {'D Flow(ms)':>11}  {'E MST(ms)':>10}")
    print("-" * 70)

    for n in sizes:
        generated_grid = generate_maze(n, n, seed=42)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(generated_grid))
            tmp_path = tmp.name

        try:
            grid, rows, cols, start, goal = parse_maze(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        t0 = time.perf_counter()
        subtask_a(grid, rows, cols, start, goal, mode="4")
        time_a = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        subtask_b(grid, rows, cols, start, goal, mode="4", cost_model=1)
        time_b = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        subtask_d(grid, rows, cols, start, goal, mode="4")
        time_d = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        subtask_e(grid, rows, cols, start, goal, mode="4")
        time_e = (time.perf_counter() - t0) * 1000

        print(f"{n:>4}x{n:<3}  {time_a:>10.2f}  {time_b:>15.2f}  {time_d:>11.2f}  {time_e:>10.2f}")

    print("=" * 70)


##############################
# OUTPUT COLLECTION
##############################

class OutputWriter:
    """Collect output lines and print them at the same time."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def write(self, text: str = "") -> None:
        self.lines.append(text)
        print(text)

    def section(self, title: str) -> None:
        line = "=" * 70
        self.write("\n" + line)
        self.write(title)
        self.write(line)

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("\n".join(self.lines) + "\n")


##############################
# MAIN REPORTING LOGIC
##############################

def report_subtask_a(writer: OutputWriter, grid: Grid, rows: int, cols: int, start: Position, goal: Position) -> None:
    writer.section("Subtask A: Shortest Path by Number of Moves")

    for mode in ("4", "8"):
        moves, path = subtask_a(grid, rows, cols, start, goal, mode)
        writer.write(f"movement = {mode}-directional")
        if path:
            writer.write(f"minimum_moves = {moves}")
            writer.write(f"path = {path_to_str(path)}")
            writer.write("path_visualization:")
            for line in format_maze_with_path(grid, path):
                writer.write(line)
        else:
            writer.write("result = UNREACHABLE")
        writer.write()

    writer.write("Approach: Breadth-First Search (BFS).")
    writer.write("Reason: Every legal move has equal cost 1, so BFS guarantees the minimum number of moves.")
    writer.write("Time complexity: O(V + E).")
    writer.write("Space complexity: O(V).")


def report_subtask_b(
    writer: OutputWriter,
    grid: Grid,
    rows: int,
    cols: int,
    start: Position,
    goal: Position,
    selected_mode: str,
    all_cost_models: bool,
) -> None:
    writer.section("Subtask B: Minimum-Cost Path")

    cost_models = [1, 2, 3] if all_cost_models else [1]
    movement_modes = ["4", "8"] if all_cost_models else [selected_mode]

    for mode in movement_modes:
        for cost_model in cost_models:
            cost, path = subtask_b(grid, rows, cols, start, goal, mode, cost_model)
            writer.write(f"movement = {mode}-directional")
            writer.write(f"cost_model = {cost_model} - {COST_MODEL_NAMES[cost_model]}")
            if path:
                writer.write(f"minimum_cost = {cost}")
                writer.write(f"path = {path_to_str(path)}")
            else:
                writer.write("result = UNREACHABLE")
            writer.write()

    writer.write("Approach: Dijkstra's algorithm with a binary heap priority queue.")
    writer.write("Reason: All cell values and edge costs are non-negative, so Dijkstra correctly finds the minimum-cost path.")
    writer.write("Time complexity: O((V + E) log V).")
    writer.write("Space complexity: O(V + E) in general; for grid graphs this is usually simplified to O(V) because E is proportional to V.")


def report_subtask_c(writer: OutputWriter, grid: Grid, rows: int, cols: int, start: Position, goal: Position) -> None:
    writer.section("Subtask C: Movement Mode Comparison")

    results = subtask_c(grid, rows, cols, start, goal)

    for mode in ("4", "8"):
        data = results[mode]
        writer.write(f"movement = {mode}-directional")
        writer.write(f"shortest_moves = {data['shortest_moves']}")
        writer.write(f"shortest_path = {path_to_str(data['path_moves'])}")
        writer.write(f"cost_of_shortest_path_using_entering_model = {data['shortest_path_entering_cost']}")
        writer.write(f"minimum_cost_entering_model = {data['min_cost']}")
        writer.write(f"minimum_cost_path_moves = {data['min_cost_path_moves']}")
        writer.write(f"minimum_cost_path = {path_to_str(data['path_cost'])}")
        writer.write()

    moves_4 = results["4"]["shortest_moves"]
    moves_8 = results["8"]["shortest_moves"]
    cost_4 = results["4"]["min_cost"]
    cost_8 = results["8"]["min_cost"]
    same_path_4 = results["4"]["path_moves"] == results["4"]["path_cost"]

    writer.write(f"Does allowing diagonal movement change the shortest path length? {'Yes' if moves_4 != moves_8 else 'No'}.")
    writer.write(f"4-directional shortest moves = {moves_4}; 8-directional shortest moves = {moves_8}.")
    writer.write(f"Does allowing diagonal movement change the cheapest path cost? {'Yes' if cost_4 != cost_8 else 'No'}.")
    writer.write(f"4-directional minimum cost = {cost_4}; 8-directional minimum cost = {cost_8}.")
    writer.write(f"Is the fewest-moves path the same as the cheapest path in 4-directional mode? {'Yes' if same_path_4 else 'No'}.")
    writer.write(
        "Explanation: BFS minimizes the number of moves, while Dijkstra minimizes the total cell cost. "
        "Therefore, a path with more moves can still be cheaper if it passes through lower-value cells."
    )
    writer.write("Time complexity: BFS is O(V + E); Dijkstra is O((V + E) log V).")
    writer.write("Space complexity: O(V) for BFS and O(V + E) for Dijkstra.")


def report_subtask_d(writer: OutputWriter, grid: Grid, rows: int, cols: int, start: Position, goal: Position) -> None:
    writer.section("Subtask D: Maximum Flow from G to S")

    for mode in ("4", "8"):
        max_flow, flow_edges = subtask_d(grid, rows, cols, start, goal, mode)
        writer.write(f"movement = {mode}-directional")
        writer.write(f"max_flow_G_to_S = {max_flow}")
        writer.write(f"positive_flow_edges_count = {len(flow_edges)}")
        writer.write("positive_flow_edges:")
        for u, v, flow, cap in sorted(flow_edges):
            writer.write(f"({u[0]},{u[1]})->({v[0]},{v[1]}): {flow}/{cap}")
        writer.write()

    writer.write("Approach: Edmonds-Karp algorithm, which is BFS-based Ford-Fulkerson.")
    writer.write("Vertices: all non-wall cells.")
    writer.write("Edges: directed edges between neighboring cells according to the selected movement mode.")
    writer.write("Capacity: capacity(u,v) = value(v), except capacity into S and G is 100.")
    writer.write("Source: G. Sink: S.")
    writer.write("Time complexity: O(V * E^2).")
    writer.write("Space complexity: O(V + E).")


def report_subtask_e(writer: OutputWriter, grid: Grid, rows: int, cols: int, start: Position, goal: Position) -> None:
    writer.section("Subtask E: Minimum Spanning Tree of the Maze Graph")

    for mode in ("4", "8"):
        total_weight, vertex_count, edge_count, mst_edges, goal_reachable = subtask_e(grid, rows, cols, start, goal, mode)
        writer.write(f"movement = {mode}-directional")
        writer.write(f"mst_total_weight = {total_weight}")
        writer.write(f"vertices_in_component = {vertex_count}")
        writer.write(f"mst_edges_count = {edge_count}")
        writer.write(f"goal_reachable_from_S = {goal_reachable}")
        writer.write("mst_edges:")
        for u, v, weight in sorted(mst_edges):
            writer.write(f"({u[0]},{u[1]})-({v[0]},{v[1]}): weight {weight}")
        writer.write()

    writer.write("Approach: Prim's algorithm starting from S.")
    writer.write("Graph interpretation: undirected weighted graph of all non-wall cells in the component containing S.")
    writer.write("Edge weight: weight(u,v) = value(u) + value(v).")
    writer.write("Cycle avoidance: a cell is added only once, using the in_tree set.")
    writer.write("Time complexity: O((V + E) log V).")
    writer.write("Space complexity: O(V + E).")


def report_assumptions(writer: OutputWriter) -> None:
    writer.section("Assumptions and Notes")
    writer.write("1. S and G are treated as passable cells with numeric value 0.")
    writer.write("2. X cells are walls and cannot be entered.")
    writer.write("3. For 4-directional movement, the allowed moves are up, down, left, and right.")
    writer.write("4. For 8-directional movement, diagonal moves are also allowed and count as one move.")
    writer.write("5. Diagonal movement is allowed if the destination cell is inside the maze and not a wall.")
    writer.write("6. No additional corner-cutting restriction is applied for diagonal moves.")
    writer.write("7. V means the number of passable cells; E means the number of graph edges.")
    writer.write("8. The program validates that the maze has exactly one S, exactly one G, valid characters, equal row lengths, and size at most 100x100.")


def run_all_reports(args: argparse.Namespace) -> None:
    grid, rows, cols, start, goal = parse_maze(args.maze)
    writer = OutputWriter()

    vertices_4, edges_4 = count_vertices_edges(grid, rows, cols, "4")
    vertices_8, edges_8 = count_vertices_edges(grid, rows, cols, "8")

    writer.write("=" * 70)
    writer.write("Maze Graph Algorithms - Final Group Project Output")
    writer.write("=" * 70)
    writer.write(f"maze_file = {args.maze}")
    writer.write(f"maze_size = {rows}x{cols}")
    writer.write(f"start = ({start[0]},{start[1]})")
    writer.write(f"goal = ({goal[0]},{goal[1]})")
    writer.write(f"vertices_4_directional = {vertices_4}")
    writer.write(f"directed_neighbor_edges_4_directional = {edges_4}")
    writer.write(f"vertices_8_directional = {vertices_8}")
    writer.write(f"directed_neighbor_edges_8_directional = {edges_8}")

    report_assumptions(writer)
    report_subtask_a(writer, grid, rows, cols, start, goal)
    report_subtask_b(writer, grid, rows, cols, start, goal, args.mode, args.all_cost_models)
    report_subtask_c(writer, grid, rows, cols, start, goal)
    report_subtask_d(writer, grid, rows, cols, start, goal)
    report_subtask_e(writer, grid, rows, cols, start, goal)

    output_file = args.output or str(Path(args.maze).with_suffix("")) + "_output.txt"
    writer.save(output_file)
    print(f"\n[done] Results written to {output_file}")

    if args.export_json:
        _, path_a = subtask_a(grid, rows, cols, start, goal, args.mode)
        _, path_b = subtask_b(grid, rows, cols, start, goal, args.mode, 1)
        _, flow_edges = subtask_d(grid, rows, cols, start, goal, args.mode)
        _, _, _, mst_edges, _ = subtask_e(grid, rows, cols, start, goal, args.mode)
        json_file = str(Path(args.maze).with_suffix("")) + "_graph.json"
        export_json(grid, rows, cols, start, goal, args.mode, path_a, path_b, flow_edges, mst_edges, json_file)

    if args.export_graphml:
        graphml_file = str(Path(args.maze).with_suffix("")) + "_graph.graphml"
        export_graphml(grid, rows, cols, args.mode, graphml_file)

    if args.benchmark is not None:
        sizes = parse_benchmark_sizes(args.benchmark)
        benchmark(sizes)


##############################
# COMMAND-LINE INTERFACE
##############################
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maze Graph Algorithms Solver")

    parser.add_argument("maze", help="Path to the maze .txt file")
    parser.add_argument(
        "--mode",
        default="4",
        choices=["4", "8"],
        help="Movement mode used for single-mode options and exports. Default: 4.",
    )
    parser.add_argument(
        "--all-cost-models",
        action="store_true",
        help="Run Subtask B for all three cost models and both movement modes.",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export the graph and selected results to JSON.",
    )
    parser.add_argument(
        "--export-graphml",
        action="store_true",
        help="Export the maze graph to GraphML.",
    )
    parser.add_argument(
        "--benchmark",
        nargs="?",
        const="10,20,50,100",
        default=None,
        help="Run benchmark. Examples: --benchmark, --benchmark 50x50, --benchmark 10,20,50,100.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the main result output to this file.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        run_all_reports(args)
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()