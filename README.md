# Maze Graph Algorithms — Final Group Project

**Students:** Ilaha Habibova (231ADB231) · Ngoc Bao Tram Tran (231ADB294) · Kamalesh Ashokkumar Kowsalya (221ADB216)  
**Language:** Python 3.8+  
**Subtasks completed:** A · B · C · D · E (all five)

---

## Files in this submission

| File | Description |
|------|-------------|
| `maze_new.py` | Main solver — all 5 subtasks + extensions |
| `maze_10x10_A.txt` | Required test maze |
| `maze_10x10_A_output.txt` | Full output for all subtasks |
| `maze_visualizer.html` | Interactive browser visualizer |
| `report.docx` | Project report |
| `README.md` | This file |

---

## Requirements

- Python 3.8 or later
- No external packages needed — only standard library
- Open `maze_visualizer.html` in any modern browser (Chrome, Firefox, Edge)

---

## How to run

### Basic run — all subtasks, 4-directional, cost model 1
```
python maze_new.py maze_10x10_A.txt
```

### Full run — all cost models, both movement modes, save to file
```
python maze_new.py maze_10x10_A.txt --all-cost-models --output maze_10x10_A_output.txt
```

### With JSON export
```
python maze_new.py maze_10x10_A.txt --all-cost-models --export-json
```

### With GraphML export (for Gephi / yEd)
```
python maze_new.py maze_10x10_A.txt --export-graphml
```

### Run benchmark on generated mazes
```
python maze_new.py maze_10x10_A.txt --benchmark
```

### Benchmark on specific sizes
```
python maze_new.py maze_10x10_A.txt --benchmark 50x50
python maze_new.py maze_10x10_A.txt --benchmark 10,20,50,100
```

### Use 8-directional movement
```
python maze_new.py maze_10x10_A.txt --mode 8
```

---

## All command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `maze` | (required) | Path to maze .txt file |
| `--mode 4` or `--mode 8` | `4` | Movement mode |
| `--all-cost-models` | off | Run subtask B with all 3 cost models and both movement modes |
| `--export-json` | off | Export graph and results to JSON |
| `--export-graphml` | off | Export graph to GraphML |
| `--benchmark` | off | Run benchmark (default sizes: 10,20,50,100) |
| `--output FILE` | auto | Save output to specified file |

---

## How to test

### Test 1 — check basic output
```
python maze_new.py maze_10x10_A.txt
```
Expected: subtasks A through E printed to screen, output file created.

### Test 2 — check all cost models appear
```
python maze_new.py maze_10x10_A.txt --all-cost-models
```
Expected: subtask B shows 6 results (3 cost models × 2 movement modes).

### Test 3 — verify known results
Check these exact values in the output:

```
Subtask A:
  4-directional minimum_moves = 18
  8-directional minimum_moves = 11

Subtask B (cost model 1):
  4-directional minimum_cost = 26
  8-directional minimum_cost = 18

Subtask D:
  4-directional max_flow_G_to_S = 1
  8-directional max_flow_G_to_S = 4

Subtask E:
  4-directional mst_total_weight = 390
  8-directional mst_total_weight = 370
  vertices_in_component = 76 (both modes)
  mst_edges_count = 75 (both modes)
```

### Test 4 — benchmark runs without error
```
python maze_new.py maze_10x10_A.txt --benchmark 10,20,50,100
```
Expected: timing table printed for all 4 sizes.

### Test 5 — exports work
```
python maze_new.py maze_10x10_A.txt --export-json --export-graphml
```
Expected: `maze_10x10_A_graph.json` and `maze_10x10_A_graph.graphml` created.

### Test 6 — invalid maze gives clear error
Create a file `bad.txt` with content `SXXX` and run:
```
python maze_new.py bad.txt
```
Expected: error message saying no G was found.

---

## Visualization

Open `maze_visualizer.html` in a browser. No installation needed.

**Features:**
- Click tabs to switch between subtasks A / B / C / D / E
- Toggle between 4-directional and 8-directional movement
- For subtask B, switch between cost models 1, 2, 3
- Click **▶ animate path** to watch the path build step by step
- Benchmark chart shows actual timing results at the bottom

---

## Algorithm summary

| Subtask | Algorithm | Time | Space |
|---------|-----------|------|-------|
| A — fewest moves | BFS | O(V+E) | O(V) |
| B — min cost | Dijkstra + heap | O((V+E) log V) | O(V) |
| C — comparison | BFS + Dijkstra both modes | O((V+E) log V) | O(V) |
| D — max flow | Edmonds-Karp | O(V·E²) | O(V+E) |
| E — MST | Prim's + heap | O((V+E) log V) | O(V+E) |

V = passable cells, E = edges between neighboring cells.

---

## Extensions implemented

1. **All 3 cost models** for subtask B (`--all-cost-models`)
2. **JSON export** of full graph + all results (`--export-json`)
3. **GraphML export** for Gephi / yEd (`--export-graphml`)
4. **Random maze generator** — `generate_maze(rows, cols, seed=42)`
5. **Benchmark** on generated mazes up to 100×100 (`--benchmark`)
6. **Interactive HTML visualizer** with step-by-step animation

---

## Maze file format

```
S1234X6789
1X2X4X6X89
...
9876X4321G
```

- `S` = start (value 0)
- `G` = goal (value 0)
- `X` = wall (cannot enter)
- `0–9` = passable cell with that numeric value
- All rows must have the same length
- Maximum size: 100×100
- Exactly one S and one G required
