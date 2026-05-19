"""
=============================================================
Maze Graph Algorithms — Final Group Project
=============================================================
Student(s): [Your Name(s)]
Student ID(s): [Your ID(s)]
Group: [Your Group Number]
Language: Python 3.8+
Run:  python maze_solver.py maze_10x10_A.txt
      python maze_solver.py maze_10x10_A.txt --all-cost-models
      python maze_solver.py maze_10x10_A.txt --export-json
      python maze_solver.py maze_10x10_A.txt --benchmark 50x50
=============================================================
"""

import sys
import json
import time
import random
import heapq
import argparse
from collections import deque, defaultdict
from copy import deepcopy


# ─────────────────────────────────────────────────────────────
# MAZE PARSING
# ─────────────────────────────────────────────────────────────

def parse_maze(filepath):
    """
    Read maze from text file.  Returns:
      grid  – list of strings (rows)
      rows  – number of rows
      cols  – number of columns
      start – (row, col) of 'S'
      goal  – (row, col) of 'G'
    """
    with open(filepath) as f:
        lines = [line.rstrip('\n') for line in f if line.strip()]
    grid = lines
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            ch = grid[r][c]
            if ch == 'S':
                start = (r, c)
            elif ch == 'G':
                goal = (r, c)
    assert start is not None, "No 'S' found in maze"
    assert goal is not None,  "No 'G' found in maze"
    return grid, rows, cols, start, goal


def cell_value(grid, r, c):
    """
    Numeric value of a cell.
    S→0, G→0, digit→int, X→wall (None).
    """
    ch = grid[r][c]
    if ch == 'S' or ch == 'G':
        return 0
    if ch == 'X':
        return None          # wall
    return int(ch)


def is_passable(grid, rows, cols, r, c):
    """True when (r,c) is inside the maze and not a wall."""
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X'


def neighbors(grid, rows, cols, r, c, mode='4'):
    """
    Yield valid neighbor positions.
    mode='4'  → up/down/left/right
    mode='8'  → + diagonals
    """
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    dirs8 = dirs4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    directions = dirs8 if mode == '8' else dirs4
    for dr, dc in directions:
        nr, nc = r+dr, c+dc
        if is_passable(grid, rows, cols, nr, nc):
            yield nr, nc


def path_to_str(path):
    return '->'.join(f'({r},{c})' for r, c in path)


# ─────────────────────────────────────────────────────────────
# SUBTASK A — Shortest path by number of moves (BFS)
# ─────────────────────────────────────────────────────────────

def subtask_a(grid, rows, cols, start, goal, mode='4'):
    """
    BFS from start to goal.  Each legal move has cost 1.

    Time complexity:  O(V + E)  where V = passable cells, E = edges.
    Space complexity: O(V)      for visited set + queue + parent map.
    """
    visited = {start}
    parent  = {start: None}
    queue   = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            # reconstruct path
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return len(path) - 1, path   # moves = edges in path

        for nr, nc in neighbors(grid, rows, cols, r, c, mode):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return None, []   # unreachable


# ─────────────────────────────────────────────────────────────
# SUBTASK B — Minimum-cost path (Dijkstra)
# ─────────────────────────────────────────────────────────────

def edge_cost(grid, ur, uc, vr, vc, model):
    """
    Cost of moving from (ur,uc) to (vr,vc).
    model 1 → entering cost  (value of v)
    model 2 → leaving  cost  (value of u)
    model 3 → combined cost  (value of u + value of v)
    """
    uval = cell_value(grid, ur, uc)
    vval = cell_value(grid, vr, vc)
    if model == 1:
        return vval
    if model == 2:
        return uval
    return uval + vval   # model 3


def subtask_b(grid, rows, cols, start, goal, mode='4', cost_model=1):
    """
    Dijkstra's algorithm for minimum-cost path.

    Time complexity:  O((V + E) log V)   using a binary heap.
    Space complexity: O(V)               for dist map, heap, parent map.
    """
    dist   = {start: 0}
    parent = {start: None}
    heap   = [(0, start)]   # (cost, node)

    while heap:
        cost, (r, c) = heapq.heappop(heap)
        if cost > dist.get((r, c), float('inf')):
            continue   # stale entry
        if (r, c) == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return cost, path

        for nr, nc in neighbors(grid, rows, cols, r, c, mode):
            new_cost = cost + edge_cost(grid, r, c, nr, nc, cost_model)
            if new_cost < dist.get((nr, nc), float('inf')):
                dist[(nr, nc)]   = new_cost
                parent[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, (nr, nc)))

    return None, []


# ─────────────────────────────────────────────────────────────
# SUBTASK C — Movement mode comparison
# ─────────────────────────────────────────────────────────────

def subtask_c(grid, rows, cols, start, goal):
    """
    Run subtask A and B under both movement modes and compare.
    """
    results = {}
    for mode in ('4', '8'):
        moves_a, path_a  = subtask_a(grid, rows, cols, start, goal, mode)
        cost_b,  path_b  = subtask_b(grid, rows, cols, start, goal, mode, cost_model=1)
        results[mode] = {
            'shortest_moves': moves_a,
            'path_moves':     path_a,
            'min_cost':       cost_b,
            'path_cost':      path_b,
        }
    return results


# ─────────────────────────────────────────────────────────────
# SUBTASK D — Maximum flow G → S  (Edmonds-Karp / BFS augmenting)
# ─────────────────────────────────────────────────────────────

def build_flow_graph(grid, rows, cols, start, goal, mode='4'):
    """
    Directed graph for max-flow.
    capacity(u→v) = value(v), except:
      capacity into S = 100
      capacity into G = 100
    Returns adjacency dict and capacity dict.
    """
    SPECIAL = 100

    # Build adjacency list and capacity map
    capacity = defaultdict(lambda: defaultdict(int))
    adj      = defaultdict(set)

    for r in range(rows):
        for c in range(cols):
            if not is_passable(grid, rows, cols, r, c):
                continue
            for nr, nc in neighbors(grid, rows, cols, r, c, mode):
                # capacity of directed edge (r,c)→(nr,nc) = value(nr,nc)
                dest = (nr, nc)
                if dest == start or dest == goal:
                    cap = SPECIAL
                else:
                    cap = cell_value(grid, nr, nc)
                if cap > 0:
                    capacity[(r,c)][dest]   += cap
                    # reverse edge (for residual graph)
                    capacity[dest][(r,c)]   += 0   # ensure key exists
                    adj[(r,c)].add(dest)
                    adj[dest].add((r,c))

    return adj, capacity


def bfs_find_path(source, sink, adj, residual):
    """BFS on residual graph.  Returns parent dict or None if unreachable."""
    visited = {source}
    parent  = {source: None}
    queue   = deque([source])
    while queue:
        u = queue.popleft()
        if u == sink:
            return parent
        for v in adj[u]:
            if v not in visited and residual[u][v] > 0:
                visited.add(v)
                parent[v] = u
                queue.append(v)
    return None


def subtask_d(grid, rows, cols, start, goal, mode='4'):
    """
    Edmonds-Karp algorithm (BFS-based Ford-Fulkerson).
    Source = goal cell (G), Sink = start cell (S).

    Time complexity:  O(V * E^2)  Edmonds-Karp worst case.
    Space complexity: O(V + E)    for residual graph.
    """
    source = goal    # flow FROM G
    sink   = start   # flow TO   S

    adj, cap = build_flow_graph(grid, rows, cols, start, goal, mode)

    # residual capacity (deep copy so we can modify)
    residual = defaultdict(lambda: defaultdict(int))
    for u in cap:
        for v in cap[u]:
            residual[u][v] = cap[u][v]

    # Also record original capacities for output
    original_cap = {u: dict(v_map) for u, v_map in cap.items()}

    max_flow = 0

    while True:
        parent = bfs_find_path(source, sink, adj, residual)
        if parent is None:
            break
        # find bottleneck
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u
        # augment
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u
        max_flow += path_flow

    # collect edges with positive flow
    flow_edges = []
    for u in original_cap:
        for v, orig in original_cap[u].items():
            flow = orig - residual[u][v]
            if flow > 0:
                flow_edges.append((u, v, flow, orig))

    return max_flow, flow_edges


# ─────────────────────────────────────────────────────────────
# SUBTASK E — Minimum Spanning Tree (Prim's algorithm)
# ─────────────────────────────────────────────────────────────

def subtask_e(grid, rows, cols, start, goal, mode='4'):
    """
    Prim's algorithm on the undirected weighted graph.
    Edge weight(u,v) = value(u) + value(v).
    Operates only on the connected component containing S.

    Time complexity:  O((V + E) log V)   with a binary heap.
    Space complexity: O(V + E)           for adjacency representation.
    """
    in_tree = set()
    key     = {start: 0}    # minimum weight to attach vertex
    parent  = {start: None}
    heap    = [(0, start, None)]   # (weight, node, from_node)
    mst_edges   = []
    total_weight = 0

    while heap:
        w, u, from_u = heapq.heappop(heap)
        if u in in_tree:
            continue
        in_tree.add(u)
        if from_u is not None:
            mst_edges.append((from_u, u, w))
            total_weight += w

        for nr, nc in neighbors(grid, rows, cols, *u, mode):
            v = (nr, nc)
            if v in in_tree:
                continue
            edge_w = cell_value(grid, *u) + cell_value(grid, *v)
            if edge_w < key.get(v, float('inf')):
                key[v]    = edge_w
                parent[v] = u
                heapq.heappush(heap, (edge_w, v, u))

    goal_reachable = goal in in_tree
    return total_weight, len(in_tree), len(mst_edges), mst_edges, goal_reachable


# ─────────────────────────────────────────────────────────────
# EXTENSION — Random maze generator
# ─────────────────────────────────────────────────────────────

def generate_maze(rows, cols, wall_prob=0.25, seed=None):
    """
    Generate a random maze ensuring S and G are placed,
    with a guaranteed path via a simple carving step.
    """
    rng = random.Random(seed)
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if rng.random() < wall_prob:
                row.append('X')
            else:
                row.append(str(rng.randint(1, 9)))
        grid.append(row)
    grid[0][0] = 'S'
    grid[rows-1][cols-1] = 'G'
    # Carve a guaranteed path along the top row then down the right column
    for c in range(cols):
        if grid[0][c] == 'X':
            grid[0][c] = str(rng.randint(1,9))
    for r in range(rows):
        if grid[r][cols-1] == 'X':
            grid[r][cols-1] = str(rng.randint(1,9))
    return [''.join(row) for row in grid]


def save_maze(grid, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in grid:
            f.write(row + '\n')


# ─────────────────────────────────────────────────────────────
# EXTENSION — GraphML / JSON export
# ─────────────────────────────────────────────────────────────

def export_json(grid, rows, cols, start, goal, mode,
                path_a, path_b, flow_edges, mst_edges, filepath):
    """Export maze graph, paths, flow, and MST to JSON."""
    vertices = []
    for r in range(rows):
        for c in range(cols):
            if is_passable(grid, rows, cols, r, c):
                vertices.append({'id': f'{r},{c}', 'row': r, 'col': c,
                                  'value': cell_value(grid, r, c),
                                  'type': grid[r][c]})
    edges = []
    seen  = set()
    for r in range(rows):
        for c in range(cols):
            if not is_passable(grid, rows, cols, r, c):
                continue
            for nr, nc in neighbors(grid, rows, cols, r, c, mode):
                key = tuple(sorted([(r,c),(nr,nc)]))
                if key not in seen:
                    seen.add(key)
                    w = cell_value(grid, r, c) + cell_value(grid, nr, nc)
                    edges.append({'u': f'{r},{c}', 'v': f'{nr},{nc}', 'weight': w})

    data = {
        'maze':       {'rows': rows, 'cols': cols,
                       'start': list(start), 'goal': list(goal)},
        'movement':   mode,
        'vertices':   vertices,
        'edges':      edges,
        'path_a':     [list(p) for p in path_a],
        'path_b':     [list(p) for p in path_b],
        'flow_edges': [[list(u), list(v), f, c]
                       for u, v, f, c in flow_edges],
        'mst_edges':  [[list(u), list(v), w]
                       for u, v, w in mst_edges],
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[export] JSON written to {filepath}")


def export_graphml(grid, rows, cols, start, goal, mode, filepath):
    """Export maze as GraphML for Gephi / yEd."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
        '  <key id="val" for="node" attr.name="value" attr.type="int"/>',
        '  <key id="wt"  for="edge" attr.name="weight" attr.type="int"/>',
        '  <graph id="maze" edgedefault="undirected">',
    ]
    for r in range(rows):
        for c in range(cols):
            if is_passable(grid, rows, cols, r, c):
                v = cell_value(grid, r, c)
                lines.append(f'    <node id="{r}_{c}"><data key="val">{v}</data></node>')
    eid  = 0
    seen = set()
    for r in range(rows):
        for c in range(cols):
            if not is_passable(grid, rows, cols, r, c):
                continue
            for nr, nc in neighbors(grid, rows, cols, r, c, mode):
                key = tuple(sorted([(r,c),(nr,nc)]))
                if key not in seen:
                    seen.add(key)
                    w = cell_value(grid, r, c) + cell_value(grid, nr, nc)
                    lines.append(f'    <edge id="e{eid}" source="{r}_{c}" target="{nr}_{nc}">'
                                 f'<data key="wt">{w}</data></edge>')
                    eid += 1
    lines += ['  </graph>', '</graphml>']
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[export] GraphML written to {filepath}")


# ─────────────────────────────────────────────────────────────
# EXTENSION — Benchmarking
# ─────────────────────────────────────────────────────────────

def benchmark(sizes=None):
    """Run all subtasks on randomly generated mazes and report timing."""
    if sizes is None:
        sizes = [10, 20, 50, 100]
    print("\n" + "="*60)
    print("BENCHMARK")
    print("="*60)
    print(f"{'Size':>8}  {'A(ms)':>8}  {'B(ms)':>8}  {'D(ms)':>8}  {'E(ms)':>8}")
    print("-"*60)
    for n in sizes:
        maze_grid = generate_maze(n, n, seed=42)
        import tempfile, os
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        tf.write('\n'.join(maze_grid))
        tf.close()
        grid, rows, cols, start, goal = parse_maze(tf.name)
        os.unlink(tf.name)

        t0 = time.perf_counter(); subtask_a(grid, rows, cols, start, goal); ta = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter(); subtask_b(grid, rows, cols, start, goal); tb = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter(); subtask_d(grid, rows, cols, start, goal); td = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter(); subtask_e(grid, rows, cols, start, goal); te = (time.perf_counter()-t0)*1000
        print(f"{n:>4}x{n:<4}  {ta:>8.2f}  {tb:>8.2f}  {td:>8.2f}  {te:>8.2f}")
    print("="*60)


# ─────────────────────────────────────────────────────────────
# OUTPUT HELPERS
# ─────────────────────────────────────────────────────────────

def print_maze_with_path(grid, rows, cols, path, label=""):
    """Print maze with path highlighted."""
    path_set = set(path)
    print(f"\n  Maze ({label}):")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if (r, c) in path_set:
                row_str += "*"
            else:
                row_str += grid[r][c]
        print("  " + row_str)


def sep(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─"*pad + f" {title} " + "─"*(w-pad-len(title)-2))
    else:
        print("\n" + "─"*w)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Maze Graph Algorithms Solver")
    parser.add_argument('maze',              help="Path to maze .txt file")
    parser.add_argument('--mode',            default='4', choices=['4','8'],
                        help="Movement mode (default 4)")
    parser.add_argument('--all-cost-models', action='store_true',
                        help="Run subtask B with all three cost models")
    parser.add_argument('--export-json',     action='store_true',
                        help="Export graph + results to JSON")
    parser.add_argument('--export-graphml',  action='store_true',
                        help="Export graph to GraphML")
    parser.add_argument('--benchmark',       action='store_true',
                        help="Run benchmark on generated mazes")
    parser.add_argument('--output',          default=None,
                        help="Write results to this file instead of stdout")
    args = parser.parse_args()

    grid, rows, cols, start, goal = parse_maze(args.maze)

    lines = []   # collect output

    def out(*a, **kw):
        text = ' '.join(str(x) for x in a)
        lines.append(text)
        print(text, **kw)

    out("="*60)
    out(f"MAZE: {args.maze}  ({rows}x{cols})")
    out(f"Start: {start}   Goal: {goal}")
    out("="*60)

    # ── Subtask A ────────────────────────────────────────────
    sep("SUBTASK A — Shortest Path (moves)")
    for mode in ('4', '8'):
        moves, path = subtask_a(grid, rows, cols, start, goal, mode)
        out(f"\nmovement         = {mode}-directional")
        if path:
            out(f"minimum_moves    = {moves}")
            out(f"path             = {path_to_str(path)}")
        else:
            out("result           = UNREACHABLE")
        print_maze_with_path(grid, rows, cols, path, f"{mode}-dir, {moves} moves")

    out("\nApproach: BFS — guarantees shortest path when all edge costs are equal.")
    out("Time:  O(V + E)   V=passable cells, E=edges")
    out("Space: O(V)       visited set + queue + parent map")

    # ── Subtask B ────────────────────────────────────────────
    sep("SUBTASK B — Minimum-Cost Path")
    cost_models = [1, 2, 3] if args.all_cost_models else [1]
    model_names = {1: "entering (cost=value(v))",
                   2: "leaving  (cost=value(u))",
                   3: "combined (cost=value(u)+value(v))"}
    for mode in ('4', '8') if args.all_cost_models else (args.mode,):
        for cm in cost_models:
            cost, path = subtask_b(grid, rows, cols, start, goal, mode, cm)
            out(f"\nmovement         = {mode}-directional")
            out(f"cost_model       = {cm} — {model_names[cm]}")
            if path:
                out(f"minimum_cost     = {cost}")
                out(f"path             = {path_to_str(path)}")
            else:
                out("result           = UNREACHABLE")

    out("\nApproach: Dijkstra with binary heap.")
    out("Time:  O((V+E) log V)")
    out("Space: O(V)")

    # ── Subtask C ────────────────────────────────────────────
    sep("SUBTASK C — Movement Mode Comparison")
    res = subtask_c(grid, rows, cols, start, goal)
    for mode in ('4', '8'):
        r = res[mode]
        out(f"\n  {mode}-directional:")
        out(f"    shortest_moves = {r['shortest_moves']}")
        out(f"    path_moves     = {path_to_str(r['path_moves'])}")
        out(f"    min_cost       = {r['min_cost']}")
        out(f"    path_cost      = {path_to_str(r['path_cost'])}")

    m4, m8 = res['4']['shortest_moves'], res['8']['shortest_moves']
    c4, c8 = res['4']['min_cost'],       res['8']['min_cost']
    out(f"\n  Diagonal changes move count? {'Yes — ' + str(m8) + ' vs ' + str(m4) if m8 != m4 else 'No (' + str(m4) + ')'}")
    out(f"  Diagonal changes cost?       {'Yes — ' + str(c8) + ' vs ' + str(c4) if c8 != c4 else 'No (' + str(c4) + ')'}")
    same_path = (res['4']['path_moves'] == res['4']['path_cost'])
    out(f"  Fewest-moves path = cheapest path (4-dir)? {'Yes' if same_path else 'No'}")
    out("\n  Discussion: 8-directional movement can reduce the number of steps by")
    out("  cutting corners diagonally. However the cheapest path may still prefer")
    out("  non-diagonal routes when diagonal cells carry higher values.")

    # ── Subtask D ────────────────────────────────────────────
    sep("SUBTASK D — Maximum Flow G → S")
    for mode in ('4', '8'):
        mf, fe = subtask_d(grid, rows, cols, start, goal, mode)
        out(f"\nmovement         = {mode}-directional")
        out(f"max_flow_G_to_S  = {mf}")
        out(f"positive_flow_edges ({len(fe)} total):")
        for (ur,uc),(vr,vc),flow,cap in sorted(fe):
            out(f"  ({ur},{uc})->({vr},{vc}): {flow}/{cap}")

    out("\nApproach: Edmonds-Karp (BFS Ford-Fulkerson).")
    out("  Vertices: all non-wall cells.")
    out("  Edges:    directed, capacity(u→v) = value(v); into S/G = 100.")
    out("  Source=G, Sink=S.")
    out("Time:  O(V * E²)  Edmonds-Karp worst case")
    out("Space: O(V + E)   residual graph")

    # ── Subtask E ────────────────────────────────────────────
    sep("SUBTASK E — Minimum Spanning Tree")
    for mode in ('4', '8'):
        tw, nv, ne, edges, g_reach = subtask_e(grid, rows, cols, start, goal, mode)
        out(f"\nmovement           = {mode}-directional")
        out(f"mst_total_weight   = {tw}")
        out(f"vertices_in_component = {nv}")
        out(f"mst_edges_count    = {ne}")
        out(f"goal_reachable     = {g_reach}")
        out(f"mst_edges:")
        for (ur,uc),(vr,vc),w in sorted(edges):
            out(f"  ({ur},{uc})-({vr},{vc}): weight {w}")

    out("\nApproach: Prim's algorithm starting from S.")
    out("  Undirected edge weight(u,v) = value(u)+value(v).")
    out("  Cycles avoided by in_tree set; cheapest crossing edge selected via heap.")
    out("Time:  O((V+E) log V)")
    out("Space: O(V+E)")

    # ── Extensions ───────────────────────────────────────────
    if args.export_json or args.export_graphml:
        sep("EXTENSIONS — Export")
        _, path_a   = subtask_a(grid, rows, cols, start, goal)
        _, path_b   = subtask_b(grid, rows, cols, start, goal)
        _, flow_edges = subtask_d(grid, rows, cols, start, goal)
        _, _, _, mst_edges, _ = subtask_e(grid, rows, cols, start, goal)

    if args.export_json:
        _, path_a     = subtask_a(grid, rows, cols, start, goal)
        _, path_b     = subtask_b(grid, rows, cols, start, goal)
        _, flow_edges = subtask_d(grid, rows, cols, start, goal)
        _, _, _, mst_edges, _ = subtask_e(grid, rows, cols, start, goal)
        export_json(grid, rows, cols, start, goal, args.mode,
                    path_a, path_b, flow_edges, mst_edges,
                    args.maze.replace('.txt', '_graph.json'))

    if args.export_graphml:
        export_graphml(grid, rows, cols, start, goal, args.mode,
                       args.maze.replace('.txt', '_graph.graphml'))

    if args.benchmark:
        benchmark()

    # Write output file
    out_file = args.output or args.maze.replace('.txt', '_output.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[done] Results written to {out_file}")


if __name__ == '__main__':
    main()