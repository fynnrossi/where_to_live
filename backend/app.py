from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
from housing_tool_functions import *


app = Flask(__name__)

# Load transport data into a NetworkX graph
G = build_graph_from_csv('london.stations.csv', 'london.lines.csv', 'london.connections.csv')
connections_df = pd.read_csv('london.connections.csv', dtype={'time': float})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get form inputs
        base_station = request.form.get('base_station', None)
        destinations = request.form.getlist('destination[]')
        visits = request.form.getlist('visits[]')
        
        # Process max travel time input (empty input becomes infinity)
        max_travel_time_str = request.form.get('max_travel_time', '').strip()
        if not max_travel_time_str:
            max_travel_time = float('inf')
        else:
            try:
                max_travel_time = float(max_travel_time_str)
            except ValueError:
                return jsonify({"error": f"Invalid max travel time: {max_travel_time_str}"})
        
        # Get the "Show Routes" toggle (checkbox returns "on" when checked)
        show_routes = request.form.get('show_routes', None)
        show_routes = True if show_routes and show_routes.lower() == "on" else False

        # Prepare destination data as tuples: (destination name, visits per week)
        destinations_data = list(zip(destinations, map(int, visits)))
        
        # Compute recommended stations (weighted by weekly travel time)
        recommended_stations = find_all_stations_sorted_by_travel_time(G, destinations_data)
        if max_travel_time is not None:
            recommended_stations = [
                station for station in recommended_stations if station[1] <= max_travel_time
            ]
        
        response_data = {}
        
        # If a base station is provided, compute its total weekly travel time
        if base_station:
            total_time = compute_weekly_travel_time(G, connections_df, base_station, destinations_data)
            response_data["comparable_station"] = base_station
            response_data["total_weekly_travel_time"] = total_time
            
            # If "Show Routes" is enabled, compute simplified routes from the base station.
            if show_routes:
                base_routes = {}
                base_node = get_start_node(G, base_station)
                for dest_name, _ in destinations_data:
                    dest_node = get_start_node(G, dest_name)
                    try:
                        route = nx.shortest_path(G, base_node, dest_node, weight=edge_weight)
                        formatted_route = format_route_simplified(G, route)
                    except Exception as e:
                        formatted_route = f"Error: {str(e)}"
                    base_routes[dest_name] = formatted_route
                response_data["base_routes"] = base_routes
        
        # Always include recommended stations
        response_data["recommended_stations"] = recommended_stations
        
        # If "Show Routes" is enabled, compute simplified routes for each recommended station.
        if show_routes:
            routes_data = {}
            name_to_id = create_name_to_id_map(G)
            for station in recommended_stations:
                station_name = station[0]
                if station_name in name_to_id:
                    candidate_id = name_to_id[station_name]
                    station_routes = {}
                    for dest_name, _ in destinations_data:
                        dest_node = get_start_node(G, dest_name)
                        try:
                            route = nx.shortest_path(G, candidate_id, dest_node, weight=edge_weight)
                            formatted_route = format_route_simplified(G, route)
                        except Exception as e:
                            formatted_route = f"Error: {str(e)}"
                        station_routes[dest_name] = formatted_route
                    routes_data[station_name] = station_routes
            response_data["routes"] = routes_data
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)})

# Helper function for computing weekly travel time.
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
        if travel_time is None:
            raise Exception(f"No connection from {base_station} to {dest_name}.")
        total_weekly_time += travel_time * visits
    return total_weekly_time

if __name__ == '__main__':
    app.run(debug=True)
