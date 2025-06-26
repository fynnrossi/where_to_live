// Functions to add or remove destination fields
function addDestinationField() {
  const container = document.getElementById("destinations");
  const div = document.createElement("div");
  div.className = "destination-field";
  div.innerHTML = `
    <input type="text" name="destination[]" placeholder="Enter destination station name" required>
    <input type="number" name="visits[]" placeholder="Visits per week" min="1" required>
    <button type="button" onclick="removeDestinationField(this)">Remove</button>
  `;
  container.appendChild(div);
}

function removeDestinationField(button) {
  const container = document.getElementById("destinations");
  if (container.children.length > 1) {
    container.removeChild(button.parentNode);
  } else {
    alert("At least one destination field must remain.");
  }
}

function search() {
  const formData = new FormData(document.getElementById("searchForm"));
  fetch("/search", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => displayResults(data))
    .catch((error) => {
      document.getElementById("results").innerText = "Error: " + error;
    });
}

function displayResults(data) {
  const resultsDiv = document.getElementById("results");
  if (data.error) {
    resultsDiv.innerText = data.error;
    return;
  }

  let html = "";

  // Display base station info, if available.
  if (data.comparable_station) {
    html += `<p>Base Station: ${data.comparable_station} — Total Weekly Travel Time: ${data.total_weekly_travel_time} minutes</p>`;
  }

  // Display the recommended stations table.
  html += "<table border='1' cellspacing='0' cellpadding='5'>";
  html += "<tr><th>Station Name</th><th>Total Weekly Travel Time</th></tr>";
  if (data.recommended_stations && data.recommended_stations.length > 0) {
    data.recommended_stations.forEach((item) => {
      html += `<tr><td>${item[0]}</td><td>${item[1]}</td></tr>`;
    });
  } else {
    html += "<tr><td colspan='2'>No recommended stations found.</td></tr>";
  }
  html += "</table>";

  // Display the routes in a simplified format.
  // We list each route as a plain paragraph without extra bold titles.
  if (data.routes || data.base_routes) {
    html += "<div style='margin-top:20px;'>";

    // For recommended station routes.
    if (data.routes) {
      for (const station in data.routes) {
        for (const dest in data.routes[station]) {
          html += `<p>${data.routes[station][dest]}</p>`;
        }
      }
    }

    // For base station routes.
    if (data.base_routes) {
      for (const dest in data.base_routes) {
        html += `<p>${data.base_routes[dest]}</p>`;
      }
    }
    html += "</div>";
  }

  resultsDiv.innerHTML = html;
}
