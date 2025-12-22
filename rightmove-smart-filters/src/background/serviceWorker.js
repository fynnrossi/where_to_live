const CACHE_KEY = "rm-smart-stations";
const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

async function getCachedStops(lineId) {
  const cache = await chrome.storage.local.get(CACHE_KEY);
  const lineCache = cache[CACHE_KEY]?.[lineId];
  if (!lineCache) return null;
  const isFresh = Date.now() - lineCache.fetchedAt < CACHE_TTL_MS;
  return isFresh ? lineCache.stops : null;
}

async function saveStops(lineId, stops) {
  const cache = await chrome.storage.local.get(CACHE_KEY);
  const payload = cache[CACHE_KEY] || {};
  payload[lineId] = {
    stops,
    fetchedAt: Date.now(),
  };
  await chrome.storage.local.set({ [CACHE_KEY]: payload });
}

function normaliseStop(raw) {
  const lat = raw.lat ?? raw.latitude;
  const lon = raw.lon ?? raw.longitude;
  if (typeof lat !== "number" || typeof lon !== "number") return null;
  return {
    id: raw.id,
    name: raw.commonName || raw.name,
    lat,
    lon,
  };
}

async function fetchLineStops(lineId) {
  const cached = await getCachedStops(lineId);
  if (cached) return cached;

  const url = `https://api.tfl.gov.uk/Line/${lineId}/StopPoints`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`TfL request failed with status ${response.status}`);
  }
  const data = await response.json();
  const stops = (Array.isArray(data) ? data : data.stopPoints || [])
    .map(normaliseStop)
    .filter(Boolean);
  if (!stops.length) {
    throw new Error("TfL response did not include stop coordinates.");
  }
  await saveStops(lineId, stops);
  return stops;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "GET_LINE_STOPS") {
    fetchLineStops(message.lineId)
      .then((stops) => sendResponse({ ok: true, stops }))
      .catch((error) => sendResponse({ ok: false, error: error.message }));
    return true; // keep channel open for async sendResponse
  }
});
