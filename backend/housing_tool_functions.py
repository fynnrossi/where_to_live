# All of the functions for the housing tool, let's keep it tidy
import requests
import json
from shapely.geometry import LineString, Point
import networkx as nx
import pandas as pd
import folium
import time
from IPython.display import display
import branca.colormap as cm

def build_graph_from_csv(stations_file, lines_file, connections_file):
    """
    Reads station, line, and connection CSV files and builds a NetworkX graph.
    
    Stations CSV must have columns including:
      - id (unique station id)
      - latitude
      - longitude
      - name
      - display_name
      - zone
      - total_lines
      - rail

    Lines CSV must have columns including:
      - line (unique line id)
      - name
      - colour
      - stripe

    Connections CSV must have columns:
      - station1 (id of the first station)
      - station2 (id of the second station)
      - line (id of the line connecting the two stations)
      - time (travel time between the two stations)
    
    Returns:
        G (networkx.Graph): A graph representing the network.
    """
    # Read CSV files (assuming tab-delimited)
    stations_df = pd.read_csv(stations_file)
    lines_df = pd.read_csv(lines_file)
    connections_df = pd.read_csv(connections_file, dtype={'time': float})
    
    # Build dictionaries for fast lookup
    # For stations, we use the unique 'id' column as the key.
    stations_dict = stations_df.set_index("id").to_dict("index")
    # For lines, use the 'line' column as the key.
    lines_dict = lines_df.set_index("line").to_dict("index")
    
    # Create an undirected graph
    G = nx.Graph()
    
    # Add nodes (stations) with attributes from stations CSV.
    for station_id, info in stations_dict.items():
        G.add_node(station_id, 
                   latitude=info["latitude"],
                   longitude=info["longitude"],
                   name=info["name"],
                   display_name=info["display_name"],
                   zone=info["zone"],
                   total_lines=info["total_lines"],
                   rail=info["rail"])
    
    # Add edges from the connections CSV.
    # Each row has: station1, station2, line, time.
    for idx, row in connections_df.iterrows():
        s1 = row["station1"]
        s2 = row["station2"]
        line_id = row["line"]
        travel_time = row["time"]
        
        # Lookup the line information using lines_dict.
        line_info = lines_dict.get(line_id, {})
        line_name = line_info.get("name", str(line_id))
        line_colour = line_info.get("colour", None)
        line_stripe = line_info.get("stripe", None)
        
        # If an edge already exists (indicating multiple lines connect the same pair),
        # append the line info; otherwise, add a new edge.
        if G.has_edge(s1, s2):
            G[s1][s2]["lines"].append({
                "line_id": line_id,
                "line_name": line_name,
                "line_colour": line_colour,
                "line_stripe": line_stripe,
                "travel_time": travel_time
            })
        else:
            G.add_edge(s1, s2, lines=[{
                "line_id": line_id,
                "line_name": line_name,
                "line_colour": line_colour,
                "line_stripe": line_stripe,
                "travel_time": travel_time
            }])
    
    return G


# --- Helper Functions ---

def get_start_node(G, start):
    """
    Given a graph G and a start value (either a station code as int or station name as str),
    returns the corresponding node (station code as int) in G.
    """
    # First, try to convert start to an integer.
    try:
        candidate = int(start)
        if candidate in G.nodes:
            return candidate
    except (ValueError, TypeError):
        pass
    # Otherwise, search by station name (case-insensitive).
    for node, attr in G.nodes(data=True):
        if "name" in attr and attr["name"].lower() == start.lower():
            return node
    raise Exception(f"Station '{start}' not found in the graph.")

def find_maximal_routes(G, start, max_stops=8):
    """
    Performs a DFS from the starting station (given by code or name) to find all simple routes
    (i.e. no repeated stations) with at most max_stops nodes.
    Then filters out any route that is a prefix of a longer route.
    
    Returns:
        list of list of int: Each route is a list of station IDs.
    """
    start_node = get_start_node(G, start)
    all_routes = []
    
    def dfs(current, path):
        # If we reach the maximum allowed stops, record this path.
        if len(path) == max_stops:
            all_routes.append(path)
            return
        extended = False
        for neighbor in G.neighbors(current):
            if neighbor not in path:  # avoid cycles
                extended = True
                dfs(neighbor, path + [neighbor])
        # If we cannot extend the path further and path length > 1, record the path.
        if not extended and len(path) > 1:
            all_routes.append(path)
    
    dfs(start_node, [start_node])
    
    # Filter out routes that are prefixes of longer routes.
    maximal_routes = []
    for route in all_routes:
        if not any((len(other) > len(route)) and (other[:len(route)] == route) for other in all_routes):
            maximal_routes.append(route)
    return maximal_routes

def calculate_route_time(route, connections_df):
    """
    Given a route (list of station IDs) and a connections DataFrame,
    computes the total travel time for that route.
    
    For each consecutive pair of stations in the route, the function looks for a matching row in the 
    connections DataFrame where either (station1 == s1 and station2 == s2) or vice-versa.
    If multiple rows exist for the same connection, the minimum travel time is used.
    
    Args:
        route (list of int): A list of station IDs.
        connections_df (pd.DataFrame): DataFrame with columns 'station1', 'station2', and 'time'.
    
    Returns:
        float: The total travel time, or None if any connection is missing.
    """
    total_time = 0.0
    for i in range(len(route) - 1):
        s1 = route[i]
        s2 = route[i+1]
        # Look up the connection in the given order.
        matching = connections_df[(connections_df["station1"] == s1) & (connections_df["station2"] == s2)]
        if matching.empty:
            # If not found, try the reverse order.
            matching = connections_df[(connections_df["station1"] == s2) & (connections_df["station2"] == s1)]
        if matching.empty:
            print(f"Warning: No connection found between {s1} and {s2}.")
            return None
        total_time += matching["time"].min()
    return total_time

def convert_route_to_names(G, route):
    """
    Converts a route given as a list of station IDs into a list of station names,
    using the "name" attribute of each node in graph G.
    
    Args:
        G (networkx.Graph): The graph.
        route (list of int): A route as a list of station IDs.
    
    Returns:
        list of str: The corresponding station names.
    """
    return [G.nodes[node].get("name", str(node)) for node in route]

# --- Combined Function ---
def find_routes_by_criteria(G, connections_df, start, max_stops, max_travel_time):
    # Convert max_travel_time to float if it's a string
    if isinstance(max_travel_time, str):
        try:
            max_travel_time = float(max_travel_time)
        except ValueError:
            raise Exception("max_travel_time must be a numeric value.")

    # 1. Find all routes from the starting station.
    all_routes = find_maximal_routes(G, start, max_stops)
    
    # 2. Compute total travel times for each route.
    route_times = [calculate_route_time(route, connections_df) for route in all_routes]
    
    # Debug: Print route times and their types
    for route, t in zip(all_routes, route_times):
        print(f"DEBUG: Route {route} -> travel time: {t} (type: {type(t)})")
    print(f"DEBUG: max_travel_time: {max_travel_time} (type: {type(max_travel_time)})")
    
    # 4. Filter routes by travel time threshold.
    try:
        filtered_routes = [
            route for route, t in zip(all_routes, route_times)
            if t is not None and t <= max_travel_time
        ]
    except Exception as e:
        print("DEBUG: Error in filtering routes:")
        for route, t in zip(all_routes, route_times):
            print(f"  Route: {route}, travel time: {t} (type: {type(t)})")
        print(f"  max_travel_time: {max_travel_time} (type: {type(max_travel_time)})")
        raise e

    # 5. Compute the set of all unique stations reached.
    reachable_stations = get_all_reachable_stations(all_routes)
    
    # 6. Filter for routes with unique endpoints.
    unique_end_routes = filter_routes_by_endpoint_not_visited_elsewhere(all_routes)
    
    # (Assuming route_names is defined earlier; if not, add conversion for route_names as needed.)
    route_names = [convert_route_to_names(G, route) for route in all_routes]
    
    return all_routes, route_names, route_times, filtered_routes, reachable_stations, unique_end_routes

# --- Helper Functions ---

def get_start_node(G, start):
    """
    Given a graph G and a start value (either an int station code or a station name),
    returns the corresponding node (as an int) from G.
    """
    # Try converting to int:
    try:
        candidate = int(start)
        if candidate in G.nodes:
            return candidate
    except (ValueError, TypeError):
        pass
    # Otherwise, search by station name (case-insensitive)
    for node, attr in G.nodes(data=True):
        if "name" in attr and attr["name"].lower() == start.lower():
            return node
    raise Exception(f"Station '{start}' not found in the graph.")

def find_maximal_routes(G, start, max_stops=8):
    """
    Performs a DFS from the starting node (resolved from the given start value)
    to collect all simple routes (no cycles) with at most max_stops nodes.
    Then, filters out any route that is a prefix of a longer route.
    
    Args:
        G (networkx.Graph): The graph.
        start (int or str): The starting station (as station code or station name).
        max_stops (int): Maximum number of nodes allowed in a route.
        
    Returns:
        list of list of int: All maximal routes (each route is a list of station IDs).
    """
    start_node = get_start_node(G, start)
    all_routes = []
    
    def dfs(current, path):
        if len(path) == max_stops:
            all_routes.append(path)
            return
        extended = False
        for neighbor in G.neighbors(current):
            if neighbor not in path:  # avoid cycles
                extended = True
                dfs(neighbor, path + [neighbor])
        if not extended and len(path) > 1:
            all_routes.append(path)
    
    dfs(start_node, [start_node])
    
    # Filter out routes that are prefixes of longer routes.
    maximal_routes = []
    for route in all_routes:
        if not any((len(other) > len(route)) and (other[:len(route)] == route) for other in all_routes):
            maximal_routes.append(route)
    return maximal_routes

def calculate_route_time(route, connections_df):
    """
    Given a route (list of station IDs) and a connections DataFrame,
    computes the total travel time for the route by summing travel times
    for each consecutive pair. The function checks for the connection in either
    direction.
    
    Args:
        route (list of int): A route (list of station IDs).
        connections_df (pd.DataFrame): DataFrame with columns "station1", "station2", and "time".
        
    Returns:
        float: The total travel time, or None if any connection is missing.
    """
    total_time = 0.0
    for i in range(len(route) - 1):
        s1, s2 = route[i], route[i+1]
        # Look up connection in the given order.
        matching = connections_df[(connections_df["station1"] == s1) & (connections_df["station2"] == s2)]
        if matching.empty:
            # Try the reverse order.
            matching = connections_df[(connections_df["station1"] == s2) & (connections_df["station2"] == s1)]
        if matching.empty:
            print(f"Warning: No connection found between {s1} and {s2}.")
            return None
        total_time += matching["time"].min()
    return total_time

def convert_route_to_names(G, route):
    """
    Converts a route (list of station IDs) to a list of station names.
    
    Args:
        G (networkx.Graph): The graph.
        route (list of int): A route as station IDs.
        
    Returns:
        list of str: Station names corresponding to the route.
    """
    return [G.nodes[node].get("name", str(node)) for node in route]

def filter_routes_by_endpoint_not_visited_elsewhere(routes):
    """
    Given a list of routes (each a list of station IDs, ints), this function first
    groups routes by their endpoint (keeping the longest route for each endpoint) and then
    filters out any route whose endpoint appears as an intermediate station in any route.
    
    Args:
        routes (list of list of int): Each route is a list of station IDs.
        
    Returns:
        list of list of int: Filtered routes.
    """
    # Group routes by endpoint; keep the longest for each endpoint.
    endpoint_to_route = {}
    for route in routes:
        if not route:
            continue
        endpoint = route[-1]
        if endpoint not in endpoint_to_route or len(route) > len(endpoint_to_route[endpoint]):
            endpoint_to_route[endpoint] = route
    unique_routes = list(endpoint_to_route.values())
    
    # Build a set of intermediate stations across all routes.
    intermediate_stations = set()
    for route in routes:
        if len(route) > 2:
            intermediate_stations.update(route[1:-1])
    
    # Keep only routes where the endpoint is not in the intermediate set.
    filtered = [route for route in unique_routes if route[-1] not in intermediate_stations]
    return filtered

def get_all_reachable_stations(routes):
    """
    Given a list of routes (each a list of station IDs), returns the set of all unique stations
    that appear in any route.
    
    Args:
        routes (list of list of int): List of routes.
        
    Returns:
        set of int: Unique station IDs reached.
    """
    reachable = set()
    for route in routes:
        reachable.update(route)
    return reachable

def find_routes_by_criteria(G, connections_df, start, max_stops, max_travel_time):
    # 1. Find all routes from the starting station.
    all_routes = find_maximal_routes(G, start, max_stops)
    
    # 2. Compute total travel times for each route.
    route_times = [calculate_route_time(route, connections_df) for route in all_routes]
    
    # Debug: Print route times and their types
    for route, t in zip(all_routes, route_times):
        print(f"DEBUG: Route {route} -> travel time: {t} (type: {type(t)})")
        print(f"DEBUG: max_travel_time: {max_travel_time} (type: {type(max_travel_time)})")
    
    # 4. Filter routes by travel time threshold.
    try:
        filtered_routes = [
            route for route, t in zip(all_routes, route_times)
            if t is not None and t <= max_travel_time
        ]
    except Exception as e:
        print("DEBUG: Error in filtering routes:")
        for route, t in zip(all_routes, route_times):
            print(f"  Route: {route}, travel time: {t} (type: {type(t)})")
            print(f"  max_travel_time: {max_travel_time} (type: {type(max_travel_time)})")
        raise e

    # 5. Compute the set of all unique stations reached.
    reachable_stations = get_all_reachable_stations(all_routes)
    
    # 6. Filter for routes with unique endpoints.
    unique_end_routes = filter_routes_by_endpoint_not_visited_elsewhere(all_routes)
    
    return all_routes, route_names, route_times, filtered_routes, reachable_stations, unique_end_routes


# --- Weight Function ---
def edge_weight(u, v, d):
    """
    Given an edge's data dictionary d (which contains a "lines" list with travel times),
    returns the minimum travel time.
    """
    return min(line["travel_time"] for line in d["lines"])

# --- Function to Map Station Names to IDs ---
def create_name_to_id_map(G):
    """
    Creates a dictionary mapping station names to their IDs.
    
    Args:
        G (networkx.Graph): The graph representing the network.
    
    Returns:
        dict: A dictionary where keys are station names and values are station IDs.
    """
    return {data['name']: node for node, data in G.nodes(data=True)}

# --- Modified Optimal Home Station Function ---
def find_all_stations_sorted_by_travel_time(G, destinations, weight=None):
    """
    Given a weighted graph G and a list of destination tuples with station names,
    returns all stations sorted by total weekly travel time.
    
    Each destination is given as a tuple: (destination_station_name, visits_per_week).
    
    Args:
        G (networkx.Graph): The graph representing the network.
        destinations (list of tuple): Each tuple is (destination_station_name, visits_per_week).
        weight (callable, optional): Weight function for edges. Defaults to edge_weight.
    
    Returns:
        list: A list of tuples (station_name, total_weekly_travel_time) sorted by travel time.
    """
    if weight is None:
        weight = edge_weight

    # Create the name-to-ID mapping
    name_to_id = create_name_to_id_map(G)

    # Convert destination names to IDs
    try:
        destinations = [(name_to_id[dest], visits) for dest, visits in destinations]
    except KeyError as e:
        raise ValueError(f"Station name '{e.args[0]}' not found in the graph.")

    travel_times = []

    # Loop over every candidate in the graph.
    for candidate in G.nodes:
        try:
            # Compute shortest path lengths from candidate to all other nodes using the custom weight.
            distances = nx.single_source_dijkstra_path_length(G, candidate, weight=weight)
        except Exception:
            continue

        candidate_cost = 0.0
        valid = True
        for dest, visits in destinations:
            if candidate == dest:
                d = 0.0
            elif dest in distances:
                d = distances[dest]
            else:
                valid = False
                break
            candidate_cost += d * visits
        if valid:
            travel_times.append((G.nodes[candidate]['name'], candidate_cost))

        travel_times = sorted(travel_times, key = lambda x : x[1])


    return travel_times




def visualize_tube_routes(routes, station_data, line_colors, map_style="cartodbpositron"):
    """
    Visualizes animated London Underground routes on an Apple Maps-style map.
    Routes trace along TfL tube lines in their official colors.

    :param routes: List of routes (list of station ID lists)
    :param station_data: Dictionary mapping station IDs to (lat, lon, name)
    :param line_colors: Dictionary mapping station IDs to line HEX colors
    :param map_style: The map style to use (default: CartoDB Positron)
    """

    # Find the central station for centering the map
    start_station = routes[0][0]  # Take the first station of the first route
    start_lat, start_lon, _ = station_data[start_station]

    # Create a Folium map with a clean Apple Maps-style basemap
    tube_map = folium.Map(
        location=[start_lat, start_lon],
        zoom_start=12,
        tiles=map_style,
    )

    # Loop through each route to animate tracing the paths
    for route in routes:
        polyline_points = []
        line_color = line_colors.get(route[0], "#000000")  # Default to black if not found
        
        for i in range(len(route) - 1):
            station1 = route[i]
            station2 = route[i + 1]

            if station1 in station_data and station2 in station_data:
                lat1, lon1, name1 = station_data[station1]
                lat2, lon2, name2 = station_data[station2]

                polyline_points.append([lat1, lon1])
                polyline_points.append([lat2, lon2])

                # Add an animated line segment
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=line_color,
                    weight=5,
                    opacity=0.8,
                ).add_to(tube_map)

                # Add station markers with popups
                folium.CircleMarker(
                    location=[lat1, lon1],
                    radius=4,
                    color=line_color,
                    fill=True,
                    fill_color=line_color,
                    popup=name1,
                ).add_to(tube_map)

                folium.CircleMarker(
                    location=[lat2, lon2],
                    radius=4,
                    color=line_color,
                    fill=True,
                    fill_color=line_color,
                    popup=name2,
                ).add_to(tube_map)

                # Simulate animation delay (for effect)
                time.sleep(0.5)

    # Show the final map
    display(tube_map)


def compute_weekly_travel_time(G, connections_df, base_station, destinations_data, weight=None):
    if weight is None:
        weight = edge_weight
    try:
        base_node = get_start_node(G, base_station)
    except Exception as e:
        raise Exception(f"Base station error: {e}")
    total_weekly_time = 0.0
    for dest_name, visits in destinations_data:
        try:
            dest_node = get_start_node(G, dest_name)
        except Exception as e:
            raise Exception(f"Destination error: {e}")
        distances = nx.single_source_dijkstra_path_length(G, base_node, weight=weight)
        travel_time = distances.get(dest_node)
        
        # Debugging print statements:
        print(f"DEBUG: Destination: {dest_name}")
        print(f"       Raw travel_time: {travel_time} (type: {type(travel_time)})")
        print(f"       Visits: {visits} (type: {type(visits)})")
        
        if travel_time is None:
            raise Exception(f"No connection from {base_station} to {dest_name}.")
        try:
            travel_time = float(travel_time)
        except Exception as e:
            raise Exception(f"Error converting travel_time to float for {dest_name}: {travel_time}")
        total_weekly_time += travel_time * visits

    print(f"DEBUG: Total weekly time from {base_station} = {total_weekly_time} (type: {type(total_weekly_time)})")
    return total_weekly_time

def format_route_with_line_colors(G, route):
    """
    Given a route (list of station IDs), returns an HTML string where station names
    are separated by arrows styled in the color of the line connecting them.
    """
    route_str = ""
    for i in range(len(route)):
        station_name = G.nodes[route[i]].get("name", str(route[i]))
        route_str += station_name
        if i < len(route) - 1:
            edge_data = G.get_edge_data(route[i], route[i+1])
            if edge_data and "lines" in edge_data and edge_data["lines"]:
                # Get the line colour from the first line
                line_colour = edge_data["lines"][0].get("line_colour", "#000")
                # Prepend '#' if not present
                if not line_colour.startswith("#"):
                    line_colour = "#" + line_colour
            else:
                line_colour = "#000"
            route_str += f' <span style="color: {line_colour};">→</span> '
    return route_str



def create_name_to_id_map(G):
    """
    Given a NetworkX graph G, returns a dictionary mapping each station's name (from the 'name' attribute)
    to its corresponding node ID.
    """
    return {data.get("name"): node for node, data in G.nodes(data=True)}

def format_route_simplified(G, route):
    """
    Given a route (a list of station IDs), returns an HTML string that shows only:
      - The starting station,
      - Any station where the line (as defined by the first line in the edge data) changes,
      - The final station.
      
    Colored arrows (using the line colour) are inserted between these stations.
    """
    if not route:
        return ""
    
    # Start with the first station.
    simplified = [route[0]]
    
    # Get the line (colour) used on the first segment, if available.
    current_line_colour = None
    if len(route) > 1:
        edge_data = G.get_edge_data(route[0], route[1])
        if edge_data and "lines" in edge_data and edge_data["lines"]:
            current_line_colour = edge_data["lines"][0].get("line_colour", "#000")
            if not current_line_colour.startswith("#"):
                current_line_colour = "#" + current_line_colour

    # Iterate through the route to identify transfer points.
    for i in range(1, len(route)):
        edge_data = G.get_edge_data(route[i - 1], route[i])
        new_line_colour = None
        if edge_data and "lines" in edge_data and edge_data["lines"]:
            new_line_colour = edge_data["lines"][0].get("line_colour", "#000")
            if not new_line_colour.startswith("#"):
                new_line_colour = "#" + new_line_colour
        # If the line colour changes compared to the previous segment,
        # add the previous station as a transfer point.
        if new_line_colour != current_line_colour:
            if simplified[-1] != route[i - 1]:
                simplified.append(route[i - 1])
            current_line_colour = new_line_colour
    # Always include the final station.
    if simplified[-1] != route[-1]:
        simplified.append(route[-1])
    
    # Now, build the HTML string for the simplified route.
    route_str = ""
    for i, station_id in enumerate(simplified):
        station_name = G.nodes[station_id].get("name", str(station_id))
        route_str += station_name
        if i < len(simplified) - 1:
            # For the arrow, get the colour of the edge connecting this station to the next.
            edge_data = G.get_edge_data(simplified[i], simplified[i + 1])
            arrow_colour = "#000"
            if edge_data and "lines" in edge_data and edge_data["lines"]:
                arrow_colour = edge_data["lines"][0].get("line_colour", "#000")
                if not arrow_colour.startswith("#"):
                    arrow_colour = "#" + arrow_colour
            route_str += f' <span style="color: {arrow_colour};">→</span> '
    return route_str








