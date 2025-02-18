# where_to_live
App to help in search for where to live

# Travel Search Tool

This is a **Flask-based web application** that helps users find optimal places to live based on their weekly travel locations and other filters. The tool calculates total travel times using a graph-based London transport network and suggests ideal locations to minimize travel.

## Features
- **Find optimal living locations** based on travel frequency to different places.
- **Calculate total weekly travel times** for different potential home locations.
- **Filter results** by max travel time, max distance from a location, and other constraints.
- **Interactive UI** for entering travel locations and visualizing results.

## Installation

### 1. Clone the Repository
bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python app.py


Configuration

Modify config.py to adjust search parameters or default settings.
Upload CSV files (london.stations.csv, london.connections.csv, etc.) if using custom transport data.
Future Enhancements

Google Maps / Apple Maps Integration for visualization.
User authentication to save search history.
Support for multiple cities beyond London.
Contributing

Fork the repo.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Added feature").
Push to GitHub (git push origin feature-name).
Open a Pull Request.
License

MIT License © 2025 Fynn Rossi
