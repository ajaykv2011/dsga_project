"""
#####
Property-Based Tests for NetworkX Closeness Centrality Implementation
#####

Team Member: Venkata Ajay Kolla
Course: E0 251o

Algorithms Tested:
    - networkx.closeness_centrality

This file contains property-based tests using the Hypothesis library to verify
fundamental mathematical properties of closeness centrality as implemented in
NetworkX. Tests cover invariants, postconditions, metamorphic properties and boundary conditions.


To run this test run : python -m pytest test_closeness_centrality.py -v
"""

import networkx as nx
from hypothesis import given, settings, assume, HealthCheck
import hypothesis.strategies as st



# Custom Graph Generator Strategies for Diverse Test Coverage

# These strategies produce diverse graph structures with varying :
# Sizes: 2-25 nodes
# Densities: sparse to dense graphs
# Topologies: random, complete, cyclic
# Edge types: weighted and unweighted
# Graph types: undirected and directed

@st.composite
def connected_graphs(draw, min_nodes=2, max_nodes=25):
    """
    Generate random connected undirected unweighted graphs.

    Strategy: Build a random tree (guarantees connectivity via n-1 edges),
    then add 0 to n random extra edges. This produces sparse and dense graphs.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(1, n):
        j = draw(st.integers(min_value=0, max_value=i - 1))
        G.add_edge(i, j)

    num_extra = draw(st.integers(min_value=0, max_value=n))
    for _ in range(num_extra):
        u = draw(st.integers(min_value=0, max_value=n - 1))
        v = draw(st.integers(min_value=0, max_value=n - 1))
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G


@st.composite
def connected_weighted_graphs(draw, min_nodes=2, max_nodes=20,
                              min_weight=1, max_weight=50):
    """
    Generate random connected undirected graphs with positive edge weights.

    Weights represent distances. Closeness centrality on weighted graphs uses edge weights as distances,
    so shortest-path computations consider weight sums rather than step counts.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(1, n):
        j = draw(st.integers(min_value=0, max_value=i - 1))
        w = draw(st.integers(min_value=min_weight, max_value=max_weight))
        G.add_edge(i, j, weight=w)

    num_extra = draw(st.integers(min_value=0, max_value=n))
    for _ in range(num_extra):
        u = draw(st.integers(min_value=0, max_value=n - 1))
        v = draw(st.integers(min_value=0, max_value=n - 1))
        if u != v and not G.has_edge(u, v):
            w = draw(st.integers(min_value=min_weight, max_value=max_weight))
            G.add_edge(u, v, weight=w)

    return G


###
### Property-Based Tests for Closeness Centrality
###
class TestClosenessCentralityFormula:
    #Tests verifying the mathematical definition and bounds.

    @given(G=connected_graphs(min_nodes=2, max_nodes=25))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_closeness_matches_definition(self, G):
        """
        Property: For a connected graph with n>=2 nodes, the closeness
        centrality of every node v equals (n-1)/sum(d(v,u) for all u!=v).

        Mathematical basis: Closeness centrality is defined as the reciprocal of
        the average shortest-path distance from v to all other nodes, normalized
        by (n-1). Formally: C(v)=(n-1)/sum_{u!=v}d(v,u). This measures how close a node is to all others. 

        Test strategy: Generate random connected graphs with 2-25 nodes. Compute
        closeness centrality via NetworkX and independently compute the formula
        using shortest_path_length and compare values.

        Assumptions: Graph is connected and n>=2.

        Why this matters: If the centrality deviates from the formula, either
        the distance computation or the normalization is incorrect. This is
        the most fundamental correctness check for closeness centrality.
        """
        n = G.number_of_nodes()
        cc = nx.closeness_centrality(G)

        for v in G.nodes():
            dist_sum = sum(
                nx.shortest_path_length(G, v, u) for u in G.nodes() if u != v
            )
            expected = (n - 1) / dist_sum
            assert abs(cc[v] - expected) < 1e-9, (
                f"Node {v}: closeness {cc[v]} != (n-1)/dist_sum = {expected}"
            )

    @given(G=connected_graphs(min_nodes=2, max_nodes=25))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_closeness_centrality_bounds(self, G):
        """
        Property: For a connected graph with n>=2, the closeness centrality of
        every node v satisfies 0<c(v)<= 1.

        Mathematical basis: In a connected graph every distance d(v,u)>= 1,
        so the denominator sum >= n-1, giving closeness <= (n-1)/(n-1) = 1.
        All distances are finite (graph is connected) and positive, so the
        denominator is a finite positive number, making closeness >0.
        The upper bound of 1 is achieved when v is adjacent to all other nodes
        (all distances are exactly 1).

        Test strategy: Generate random connected graphs. Check both bounds
        for every node. This covers dense graphs (closeness near 1) and sparse
        graphs (closeness near 0).

        Assumptions: Graph is connected with n>=2.

        Why this matters: A value outside (0, 1] would mean the incorrect graph traversal, or normalization bug.
        """
        cc = nx.closeness_centrality(G)

        for v in G.nodes():
            assert 0 < cc[v] <= 1.0, (
                f"Node {v} closeness {cc[v]} is outside (0, 1]"
            )


class TestClosenessCentralitySpecialGraphs:
    #Test on graph families with analytically known closeness values.

    @given(n=st.integers(min_value=2, max_value=30))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_complete_graph_all_closeness_one(self, n):
        """
        Property: In a complete graph K_n, every node has closeness centrality
        exactly 1.

        Mathematical basis: In K_n every pair of distinct nodes is directly
        connected, so d(u,v) = 1 for all u!=v. The sum of distances from
        any node v is (n-1)*1 = n-1. Therefore C(v) = (n-1)/(n-1) = 1. This is the 
        theoretical maximum of closeness centrality.

        Test strategy: Generate complete graphs for n in [2, 30] and verify
        every node has closeness exactly 1. This is a boundary check since complete
        graphs are the densest possible topology.

        Assumptions: n>=2.

        Why this matters: If this fails on the simplest maximum-closeness case,
        the algorithm cannot handle basic distance computations. Complete graphs
        are the upper-bound test.
        """
        G = nx.complete_graph(n)
        cc = nx.closeness_centrality(G)

        for v in G.nodes():
            assert abs(cc[v] - 1.0) < 1e-10, (
                f"In K_{n}, node {v} has closeness {cc[v]}, expected 1.0"
            )


class TestClosenessCentralityMetamorphic:
    #Tests for metamorphic relationships under graph transformations.

    @given(G=connected_graphs(min_nodes=3, max_nodes=20))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_adding_edge_cannot_decrease_closeness(self, G):
        """
        Property: Adding an edge to a connected
        graph can never decrease any node's closeness centrality.

        Mathematical basis: Adding an edge can only create new shortest paths
        (shorter or equal distances), it cannot make any distance longer.
        Since closeness = (n-1)/sum_of_distances, and the sum of distances
        can only decrease or stay the same when an edge is added (distances
        can only shrink), closeness can only increase or stay the same.

        Test strategy: Generate a connected graph, pick a non-adjacent pair
        (skip if complete), add the edge, and verify all closeness values
        are >= their original values.

        Assumptions: Graph is connected, a non-adjacent pair exists.

        Why this matters: If adding an edge decreases a node's closeness,
        the distance computation after edge addition is incorrectly yielding
        longer paths, a contradiction that would indicate a bug
        """
        cc_before = nx.closeness_centrality(G)

        non_adj = [(u, v) for u in G.nodes() for v in G.nodes()
                   if u < v and not G.has_edge(u, v)]
        assume(len(non_adj) > 0)

        u, v = non_adj[0]
        G_new = G.copy()
        G_new.add_edge(u, v)
        cc_after = nx.closeness_centrality(G_new)

        for node in G.nodes():
            assert cc_after[node] >= cc_before[node] - 1e-10, (
                f"Node {node}: closeness decreased from {cc_before[node]} to "
                f"{cc_after[node]} after adding edge ({u},{v})"
            )

    @given(G=connected_graphs(min_nodes=2, max_nodes=20))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_closeness_isomorphism_invariance(self, G):
        """
        Property: Closeness centrality after relabeling nodes produces identical centrality values
        (up to relabeling).

        Mathematical basis: Shortest-path distances depend only on graph
        structure, not on vertex labels. Closeness centrality is a structural invariant so relabeling nodes 
        should not change the centrality values, just permute them according to the relabeling.

        Test strategy: Generate a random connected graph G. Create a relabeled
        copy G' by reversing the node labels. Compute
        closeness on both and verify C_G(v) = C_G'(n-1-v) for all v.

        Assumptions: Graph is connected with n>=2.

        Why this matters: If relabeling changes the centrality values, the
        algorithm is sensitive to node IDs rather than structure, possibly
        due to an implementation bug in how it indexes or iterates over nodes.
        """
        n = G.number_of_nodes()
        cc_original = nx.closeness_centrality(G)

        # Relabel: node i -> node (n-1-i)
        mapping = {i: n-1-i for i in range(n)}
        G_relabeled = nx.relabel_nodes(G, mapping)
        cc_relabeled = nx.closeness_centrality(G_relabeled)

        for v in G.nodes():
            mapped_v = mapping[v]
            assert abs(cc_original[v] - cc_relabeled[mapped_v]) < 1e-10, (
                f"Node {v} (mapped to {mapped_v}): original closeness "
                f"{cc_original[v]} != relabeled closeness {cc_relabeled[mapped_v]}"
            )


class TestClosenessCentralityWeighted:
    #Tests on weighted graphs where edge weights represent distances.

    @given(G=connected_weighted_graphs(min_nodes=2, max_nodes=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_weighted_closeness_matches_definition(self, G):
        """
        Property: For a connected weighted graph, closeness centrality using
        the distance parameter equals (n-1)/sum of weighted shortest-path
        distances.

        Mathematical basis: When edge weights represent distances the closeness centrality generalizes to:
        C(v) = (n-1)/sum_{u!=v} d_w(v, u)
        where d_w(v, u) is the weighted shortest-path distance (sum of edge
        weights along the shortest path, computed via Dijkstra's algorithm).
        This differs from the unweighted case where d(v,u) counts hops.

        Test strategy: Generate random connected graphs with positive integer
        edge weights (1-50). Compute closeness centrality with distance='weight'
        and independently verify using Dijkstra-based shortest path lengths.
        Graphs range from sparse trees to denser structures with varying weights.

        Assumptions: All edge weights are positive. Graph is connected.

        Why this matters: If weighted closeness deviates from the definition,
        the algorithm incorrectly computes weighted shortest paths or applies
        wrong normalization. Many real-world networks are weighted, so this is
        a critical correctness check beyond the unweighted case.
        """
        n = G.number_of_nodes()
        cc = nx.closeness_centrality(G, distance="weight")

        for v in G.nodes():
            dist_sum = sum(
                nx.dijkstra_path_length(G, v, u, weight="weight")
                for u in G.nodes() if u != v
            )
            expected = (n - 1) / dist_sum
            assert abs(cc[v] - expected) < 1e-9, (
                f"Node {v}: weighted closeness {cc[v]} != expected {expected}"
            )

    @given(G=connected_weighted_graphs(min_nodes=2, max_nodes=20))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
    def test_weighted_closeness_bounds(self, G):
        """
        Property: Weighted closeness centrality on a connected graph satisfies
        0<C(v) for all nodes, but may exceed 1 when edge weights are < 1.
        With integer weights >= 1, it satisfies 0 <C(v)<= 1.

        Mathematical basis: With all weights >= 1, every weighted distance
        d_w(v,u) >= 1 (since paths traverse at least one edge of weight >= 1).
        So the denominator sum>= n-1, giving C(v) <= 1. Closeness is always
        positive for connected graphs since all distances are finite and
        positive.

        Test strategy: Generate connected weighted graphs with integer weights
        in [1,50]. Verify 0<C(v)<=1 for all nodes.

        Assumptions: Connected graph, all weights>= 1.

        Why this matters: If bounds are violated with weights >= 1, the
        weighted distance computation is wrong, possibly ignoring weights
        or summing them incorrectly.
        """
        cc = nx.closeness_centrality(G, distance="weight")

        for v in G.nodes():
            assert 0 < cc[v] <= 1.0, (
                f"Node {v}: weighted closeness {cc[v]} outside (0, 1]"
            )

