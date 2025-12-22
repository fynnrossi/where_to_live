(() => {
  const DEFAULT_SETTINGS = {
    enabledLines: { jubilee: false },
    thresholdMinutes: 10,
    hideMode: "fade",
  };

  let settings = { ...DEFAULT_SETTINGS };
  let jubileeStops = [];
  let lastError = null;
  let observer = null;
  let markerLookup = null;

  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function injectStyles() {
    fetch(chrome.runtime.getURL("ui/panel.css"))
      .then((resp) => resp.text())
      .then((css) => {
        const style = document.createElement("style");
        style.id = "rm-smart-panel-style";
        style.textContent = css;
        document.head.appendChild(style);
      })
      .catch(() => {});
  }

  async function loadSettings() {
    const stored = await chrome.storage.local.get("rm-smart-settings");
    settings = { ...DEFAULT_SETTINGS, ...(stored["rm-smart-settings"] || {}) };
  }

  async function saveSettings() {
    await chrome.storage.local.set({ "rm-smart-settings": settings });
  }

  function haversineMeters(a, b) {
    const R = 6371000;
    const toRad = (d) => (d * Math.PI) / 180;
    const dLat = toRad(b.lat - a.lat);
    const dLon = toRad(b.lon - a.lon);
    const lat1 = toRad(a.lat);
    const lat2 = toRad(b.lat);
    const sinLat = Math.sin(dLat / 2);
    const sinLon = Math.sin(dLon / 2);
    const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLon * sinLon;
    return 2 * R * Math.asin(Math.sqrt(h));
  }

  function thresholdMeters() {
    return (settings.thresholdMinutes || 0) * 80;
  }

  function extractListingId(card) {
    const link = card.querySelector("a[href*='properties']");
    if (!link) return null;
    const match = link.href.match(/properties\/(\d+)/);
    return match ? match[1] : null;
  }

  function parseJSONBlock(text) {
    if (!text) return null;
    const trimmed = text.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed);
    } catch (err) {
      const modelMatch = trimmed.match(/jsonModel\s*=\s*(\{[\s\S]*?\})\s*;?/);
      if (modelMatch) {
        try {
          return JSON.parse(modelMatch[1]);
        } catch (_) {
          return null;
        }
      }
      return null;
    }
  }

  function hydrateMarkerLookup() {
    if (markerLookup) return markerLookup;
    markerLookup = new Map();
    const scripts = Array.from(document.querySelectorAll("script"));
    for (const script of scripts) {
      const data = parseJSONBlock(script.textContent);
      if (!data) continue;
      const markers =
        data?.properties?.map ??
        data?.results?.map ??
        data?.markers ??
        data?.properties?.results ??
        data?.results;
      if (!Array.isArray(markers)) continue;
      markers.forEach((item) => {
        const id = item.id || item.propertyId || item.listingId || item.identifier;
        const lat = item.latitude ?? item.lat;
        const lon = item.longitude ?? item.lon ?? item.lng;
        if (id && typeof lat === "number" && typeof lon === "number") {
          markerLookup.set(String(id), { lat, lon });
        }
      });
      if (markerLookup.size) break;
    }
    return markerLookup;
  }

  function getListingCoordinates(card) {
    const latAttr = card.getAttribute("data-latitude") || card.dataset.latitude;
    const lonAttr = card.getAttribute("data-longitude") || card.dataset.longitude;
    if (latAttr && lonAttr) {
      const lat = parseFloat(latAttr);
      const lon = parseFloat(lonAttr);
      if (!Number.isNaN(lat) && !Number.isNaN(lon)) {
        return { lat, lon };
      }
    }
    const id = extractListingId(card);
    if (id) {
      const lookup = hydrateMarkerLookup();
      if (lookup.has(id)) return lookup.get(id);
    }
    return null;
  }

  function findListingCards() {
    const selectors = [
      "[data-testid='propertyCard']",
      ".propertyCard",
      "div[data-test='property-card']",
      "div.l-searchResult",
    ];
    for (const selector of selectors) {
      const cards = Array.from(document.querySelectorAll(selector));
      if (cards.length) return cards;
    }
    return [];
  }

  function clearFilterState(card) {
    card.classList.remove("rm-smart-faded", "rm-smart-hidden");
  }

  function applyFilterToListing(card, stops, mode, maxDistance) {
    clearFilterState(card);
    const coords = getListingCoordinates(card);
    if (!coords || !stops.length) return true;
    let minDistance = Infinity;
    for (const stop of stops) {
      const distance = haversineMeters(coords, stop);
      if (distance < minDistance) minDistance = distance;
      if (minDistance <= maxDistance) break;
    }
    const passes = minDistance <= maxDistance;
    if (!passes) {
      if (mode === "hide") {
        card.classList.add("rm-smart-hidden");
      } else {
        card.classList.add("rm-smart-faded");
      }
    }
    return passes;
  }

  function updateStats(shown, total) {
    const stats = document.querySelector("#rm-smart-stats");
    if (stats) stats.textContent = `Showing ${shown} / ${total} listings`;
  }

  function toggleWarning(message) {
    const warning = document.querySelector("#rm-smart-warning");
    if (!warning) return;
    if (message) {
      warning.textContent = message;
      warning.style.display = "block";
    } else {
      warning.style.display = "none";
    }
  }

  function applyFilters() {
    const cards = findListingCards();
    if (!cards.length) {
      updateStats(0, 0);
      return;
    }

    const enabled = settings.enabledLines.jubilee;
    const maxDistance = thresholdMeters();

    if (!enabled || lastError) {
      cards.forEach(clearFilterState);
      updateStats(cards.length, cards.length);
      return;
    }

    let shown = 0;
    cards.forEach((card) => {
      const passes = applyFilterToListing(
        card,
        jubileeStops,
        settings.hideMode,
        maxDistance
      );
      if (passes) shown += 1;
    });
    updateStats(shown, cards.length);
  }

  async function ensureStops() {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "GET_LINE_STOPS",
        lineId: "jubilee",
      });
      if (!response?.ok) {
        lastError = response?.error || "TfL request failed";
        toggleWarning(`Station data unavailable: ${lastError}`);
        return;
      }
      jubileeStops = response.stops;
      lastError = null;
      toggleWarning("");
    } catch (err) {
      lastError = err.message;
      toggleWarning(`Station data unavailable: ${lastError}`);
    }
  }

  function buildPanel() {
    if (document.getElementById("rm-smart-panel")) return;
    const panel = document.createElement("div");
    panel.id = "rm-smart-panel";
    panel.className = "rm-smart-panel";
    panel.innerHTML = `
      <h2>Smart Filters <span class="rm-smart-badge">MVP</span></h2>
      <div class="rm-smart-row">
        <label for="rm-smart-jubilee">Near Jubilee line</label>
        <input id="rm-smart-jubilee" type="checkbox" />
      </div>
      <div class="rm-smart-row">
        <label for="rm-smart-distance">Max walk (min)</label>
        <input id="rm-smart-distance" type="range" min="0" max="25" step="1" />
      </div>
      <div class="rm-smart-row">
        <span class="rm-smart-small">Distance: <span id="rm-smart-distance-label"></span></span>
        <select id="rm-smart-mode">
          <option value="fade">Fade</option>
          <option value="hide">Hide</option>
        </select>
      </div>
      <div class="rm-smart-divider"></div>
      <div class="rm-smart-row">
        <span id="rm-smart-stats" class="rm-smart-small">Loading listings...</span>
      </div>
      <div id="rm-smart-warning" class="rm-smart-warning"></div>
    `;
    document.body.appendChild(panel);
    wirePanel(panel);
  }

  function wirePanel(panel) {
    const toggle = panel.querySelector("#rm-smart-jubilee");
    const slider = panel.querySelector("#rm-smart-distance");
    const sliderLabel = panel.querySelector("#rm-smart-distance-label");
    const modeSelect = panel.querySelector("#rm-smart-mode");

    toggle.checked = !!settings.enabledLines.jubilee;
    slider.value = String(settings.thresholdMinutes);
    sliderLabel.textContent = `${settings.thresholdMinutes} min (${thresholdMeters()} m)`;
    modeSelect.value = settings.hideMode;

    toggle.addEventListener("change", async () => {
      settings.enabledLines.jubilee = toggle.checked;
      await saveSettings();
      applyFilters();
    });

    slider.addEventListener("input", async () => {
      const value = parseInt(slider.value, 10) || 0;
      settings.thresholdMinutes = value;
      sliderLabel.textContent = `${value} min (${value * 80} m)`;
      await saveSettings();
      applyFilters();
    });

    modeSelect.addEventListener("change", async () => {
      settings.hideMode = modeSelect.value;
      await saveSettings();
      applyFilters();
    });
  }

  function startObserver() {
    const handle = debounce(() => applyFilters(), 250);
    observer = new MutationObserver(() => handle());
    observer.observe(document.body, { childList: true, subtree: true });
  }

  async function init() {
    injectStyles();
    await loadSettings();
    buildPanel();
    await ensureStops();
    applyFilters();
    startObserver();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
