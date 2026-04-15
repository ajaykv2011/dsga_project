# Property-Based Testing for NetworkX Closeness Centrality

Team Member: Venkata Ajay Kolla
Course: E0 251o

## Overview

This project contains property-based tests for the `networkx.closeness_centrality` algorithm using the Hypothesis library. The tests verify mathematical properties of closeness centrality across randomly generated graphs of varying sizes, densities, topologies, and edge weight configurations.

## Requirements

- Python 3.10+
- networkx
- hypothesis
- pytest

Install dependencies:

```
pip install networkx hypothesis pytest
```

## How to Run

Run all tests with verbose output:

```
python -m pytest test_closeness_centrality.py -v
```

Run without verbose output:

```
python -m pytest test_closeness_centrality.py
```

## Tests

The file contains 7 property-based tests organized into 4 classes:

### TestClosenessCentralityFormula

1. **test_closeness_matches_definition** - Verifies that closeness centrality matches the formula C(v) = (n-1) / sum(d(v,u)) for every node on random connected unweighted graphs.

2. **test_closeness_centrality_bounds** - Checks that closeness centrality is always in the range (0, 1] for connected graphs.

### TestClosenessCentralitySpecialGraphs

3. **test_complete_graph_all_closeness_one** - Verifies that every node in a complete graph K_n has closeness centrality exactly 1.

### TestClosenessCentralityMetamorphic

4. **test_adding_edge_cannot_decrease_closeness** - Checks that adding an edge to a graph never decreases any node's closeness centrality.

5. **test_closeness_isomorphism_invariance** - Verifies that relabeling nodes does not change closeness centrality values.

### TestClosenessCentralityWeighted

6. **test_weighted_closeness_matches_definition** - Verifies the closeness formula on weighted graphs using Dijkstra-based shortest path distances.

7. **test_weighted_closeness_bounds** - Checks that weighted closeness centrality stays in (0, 1] when all edge weights are >= 1.

## Property Types Covered

- Invariants
- Postconditions 
- Metamorphic properties 
- Boundary conditions 

## Graph Generation

Tests use custom Hypothesis strategies that generate diverse graph structures:
- Connected unweighted graphs (2-25 nodes, varying density)
- Connected weighted graphs (2-20 nodes, integer weights 1-50)
- Special graph families (complete graphs K_n, n up to 30)
