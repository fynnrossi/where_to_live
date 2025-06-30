import os
import sys
import networkx as nx
import pandas as pd
import pytest

# Allow imports from the backend directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from housing_tool_functions import compute_weekly_travel_time


def build_graph():
    G = nx.Graph()
    G.add_node(1, name="A")
    G.add_node(2, name="B")
    G.add_node(3, name="C")
    # edges with travel times stored in 'lines'
    G.add_edge(1, 2, lines=[{"travel_time": 10}])
    G.add_edge(2, 3, lines=[{"travel_time": 5}])
    G.add_edge(1, 3, lines=[{"travel_time": 20}])
    return G


def test_compute_weekly_travel_time_basic():
    G = build_graph()
    destinations = [("B", 2), ("C", 1)]
    total = compute_weekly_travel_time(G, pd.DataFrame(), "A", destinations)
    assert total == 35


def test_compute_weekly_travel_time_numeric_base():
    G = build_graph()
    destinations = [("A", 1), ("C", 3)]
    total = compute_weekly_travel_time(G, pd.DataFrame(), 2, destinations)
    assert total == 25


def test_unreachable_destination():
    G = build_graph()
    G.add_node(4, name="D")  # isolated node
    with pytest.raises(Exception, match="No connection from A to D"):
        compute_weekly_travel_time(G, pd.DataFrame(), "A", [("D", 1)])


def test_compute_weekly_travel_time_multiple_destinations():
    G = build_graph()
    destinations = [("B", 1), ("C", 1), ("B", 2)]
    total = compute_weekly_travel_time(G, pd.DataFrame(), "A", destinations)
    assert total == 45


def test_compute_weekly_travel_time_zero_visits():
    G = build_graph()
    destinations = [("B", 0), ("C", 2)]
    total = compute_weekly_travel_time(G, pd.DataFrame(), "A", destinations)
    assert total == 30
