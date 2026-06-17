function fmtBool(value, goodLabel = "yes", badLabel = "no") {
  if (value === true) return `<span class="pill good">${goodLabel}</span>`;
  if (value === false) return `<span class="pill bad">${badLabel}</span>`;
  return `<span class="pill neutral">—</span>`;
}
function pill(value) { const cls = value || "neutral"; return `<span class="pill ${cls}">${value || "—"}</span>`; }
function shortTime(s) { if (!s) return "—"; try { return new Date(s).toLocaleString(); } catch { return s; } }
function num(value, digits = 2) { if (value === null || value === undefined || Number.isNaN(value)) return "—"; return Number(value).toFixed(digits); }
function coords(lat, lon) { if (lat === null || lat === undefined || lon === null || lon === undefined) return "—"; return `${Number(lat).toFixed(7)}, ${Number(lon).toFixed(7)}`; }
function val(id) { return document.getElementById(id)?.value?.trim() || ""; }
function checked(id) { return !!document.getElementById(id)?.checked; }
function escapeHTML(s) { return String(s).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c])); }
function setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }
async function postAction(url) { try { const r = await fetch(url, {method:"POST"}); if (!r.ok) throw new Error(await r.text()); await refreshAll(); } catch(e) { alert(`Action failed: ${e}`); } }
async function postJSON(url, payload) { const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)}); if (!r.ok) throw new Error(await r.text()); return await r.json(); }

// -----------------------------------------------------------------------------
// GPSD restart button
// -----------------------------------------------------------------------------
function setGpsdStatus(message, cls = "") {
  const el = document.getElementById("gpsd-restart-status");
  if (!el) return;
  el.textContent = message;
  el.className = `action-status small ${cls}`.trim();
}
async function restartGpsd() {
  const btn = document.getElementById("restart-gpsd-btn");
  const ok = confirm("Restart gpsd.socket/gpsd.service on all configured units?");
  if (!ok) { setGpsdStatus("GPSD restart: cancelled", "warn-text"); return; }
  if (btn) btn.disabled = true;
  setGpsdStatus("GPSD restart: running...", "warn-text");
  try {
    const r = await fetch("/api/gpsd/restart", { method: "POST" });
    let data;
    try { data = await r.json(); } catch { data = { ok: false, error: await r.text() }; }
    if (!r.ok) throw new Error(JSON.stringify(data));
    const results = data.results || {};
    const lines = Object.entries(results).map(([unit, res]) => {
      const state = res.ok ? "ok" : "FAILED";
      const detail = res.stdout || res.stderr || res.error || "";
      return `${unit}: ${state}${detail ? " — " + detail : ""}`;
    });
    const summary = `GPSD restart: ${data.ok ? "completed" : "completed with failures"}`;
    setGpsdStatus(`${summary}. ${lines.join(" | ")}`, data.ok ? "good-text" : "bad-text");
    const logs = document.getElementById("logs");
    if (logs) logs.textContent = `${new Date().toLocaleTimeString()} ${summary}\n${lines.join("\n")}\n\n${logs.textContent || ""}`;
    await refreshAll();
  } catch (e) {
    setGpsdStatus(`GPSD restart: failed — ${e}`, "bad-text");
    console.error("GPSD restart failed", e);
  } finally {
    if (btn) btn.disabled = false;
  }
}
window.restartGpsd = restartGpsd;

// -----------------------------------------------------------------------------
// Leaflet map, averaged positions, override pins, result layers
// -----------------------------------------------------------------------------
let aruMap = null;
let avgLayer = null;
let overrideLayer = null;
let resultLayer = null;
let avgMarkers = {};
let overrideMarkers = {};
let latestMapData = null;
let clickToPlace = false;
let overrideDirty = false;
const unitColors = {five: "#2563eb", zero: "#dc2626", one: "#16a34a", four: "#9333ea"};

function setMapStatus(msg, cls = "") {
  const el = document.getElementById("map-status");
  if (!el) return;
  el.textContent = msg;
  el.className = `small ${cls}`.trim();
}
function validLatLon(lat, lon) {
  const a = Number(lat), b = Number(lon);
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a) <= 90 && Math.abs(b) <= 180 && !(a === 0 && b === 0);
}

function getAverageLatLon(rec) {
  if (!rec) return [null, null];

  // /api/map/locations currently returns avg_lat/avg_lon.
  // Older frozen summary objects use lat_deg/lon_deg.
  // Manual override objects may use lat/lon.
  const lat =
    rec.avg_lat ?? rec.lat ?? rec.lat_deg ?? rec.latitude ?? rec.latest_lat;
  const lon =
    rec.avg_lon ?? rec.lon ?? rec.lon_deg ?? rec.longitude ?? rec.latest_lon;

  return [lat, lon];
}

async function fetchMapLocationsNoCache() {
  const r = await fetch("/api/map/locations", { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  latestMapData = data;
  return data;
}

function makeUnitDivIcon(unit, kind) {
  const color = unitColors[unit] || "#111827";
  const label = unit.slice(0, 1).toUpperCase();
  const border = kind === "override" ? "3px solid #fbbf24" : "2px solid white";
  return L.divIcon({
    className: "unit-div-icon",
    html: `<div style="background:${color};border:${border};">${label}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13]
  });
}
function initMap() {
  if (aruMap || !document.getElementById("aru-map")) return;
  if (typeof L === "undefined") {
    setMapStatus("Leaflet did not load; map unavailable. Check internet/cache.", "bad-text");
    return;
  }
  aruMap = L.map("aru-map", { scrollWheelZoom: true }).setView([37.8146, -122.2493], 19);

  const onlineEsriLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles © Esri",
    maxZoom: 22,
    maxNativeZoom: 20
  });

  const offlineEsriLayer = L.tileLayer("/tiles/esri/{z}/{x}/{y}.jpg", {
    attribution: "Offline cached imagery",
    maxZoom: 22,
    maxNativeZoom: 20,
    errorTileUrl: ""
  });

  const osmLayer = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "© OpenStreetMap contributors",
    maxZoom: 22
  });

  onlineEsriLayer.addTo(aruMap);
  L.control.layers({
    "Esri satellite online": onlineEsriLayer,
    "Cached satellite offline": offlineEsriLayer,
    "OpenStreetMap online": osmLayer
  }, {}, { collapsed: false }).addTo(aruMap);

  avgLayer = L.layerGroup().addTo(aruMap);
  overrideLayer = L.layerGroup().addTo(aruMap);
  resultLayer = L.layerGroup().addTo(aruMap);
  aruMap.on("click", (e) => {
    if (!clickToPlace) return;
    const unit = val("map-unit-select") || "five";
    setOverrideMarker(unit, e.latlng.lat, e.latlng.lng, true);
    clickToPlace = false;
    overrideDirty = true;
    setMapStatus(`Placed override pin for ${unit}. Drag it if needed, then save.`, "good-text");
  });
  setMapStatus("Map ready.", "good-text");
}
function setAverageMarker(unit, lat, lon, rec) {
  if (!aruMap || !avgLayer || !validLatLon(lat, lon)) return;
  if (avgMarkers[unit]) avgLayer.removeLayer(avgMarkers[unit]);
  const marker = L.marker([Number(lat), Number(lon)], { icon: makeUnitDivIcon(unit, "avg"), title: `${unit} average` });
  marker.bindPopup(`<strong>${escapeHTML(unit)}</strong> averaged location<br>${coords(lat, lon)}<br>fixes: ${rec?.n_fixes ?? 0}<br>quality: ${escapeHTML(rec?.quality || "unknown")}`);
  marker.addTo(avgLayer);
  avgMarkers[unit] = marker;
}
function setOverrideMarker(unit, lat, lon, enabled = true, sigma = 1.5) {
  if (!aruMap || !overrideLayer || !validLatLon(lat, lon)) return;
  if (overrideMarkers[unit]) overrideLayer.removeLayer(overrideMarkers[unit]);
  const marker = L.marker([Number(lat), Number(lon)], { draggable: true, icon: makeUnitDivIcon(unit, "override"), title: `${unit} override` });
  marker.aru = { unit, enabled, position_sigma_m: Number(sigma) || 1.5 };
  marker.bindPopup(`<strong>${escapeHTML(unit)}</strong> override pin<br>${coords(lat, lon)}<br>drag to adjust`);
  marker.on("dragend", () => { overrideDirty = true; renderOverrideTable(); });
  marker.addTo(overrideLayer);
  overrideMarkers[unit] = marker;
  overrideDirty = true;
  renderOverrideTable();
}
function renderOverrideTable() {
  const el = document.getElementById("override-table");
  if (!el) return;
  const units = Object.keys(overrideMarkers).sort();
  if (units.length === 0) { el.textContent = "No override pins loaded."; return; }
  el.innerHTML = `<table class="mini-table"><thead><tr><th>Unit</th><th>Override position</th><th>Controls</th></tr></thead><tbody>${units.map(unit => {
    const m = overrideMarkers[unit];
    const ll = m.getLatLng();
    return `<tr><td><strong>${escapeHTML(unit)}</strong></td><td>${coords(ll.lat, ll.lng)}</td><td><button class="link-button" onclick="removeOverrideMarker('${escapeHTML(unit)}')">remove</button></td></tr>`;
  }).join("")}</tbody></table>`;
}
function removeOverrideMarker(unit) {
  if (overrideMarkers[unit] && overrideLayer) overrideLayer.removeLayer(overrideMarkers[unit]);
  delete overrideMarkers[unit];
  overrideDirty = true;
  renderOverrideTable();
}
function fitMapToKnownMarkers() {
  if (!aruMap) return;
  const pts = [];
  Object.values(avgMarkers).forEach(m => pts.push(m.getLatLng()));
  Object.values(overrideMarkers).forEach(m => pts.push(m.getLatLng()));
  if (pts.length > 0) aruMap.fitBounds(L.latLngBounds(pts), { padding: [30, 30], maxZoom: 21 });
}
async function refreshMapLocations() {
  initMap();
  if (!aruMap) return;
  try {
    const data = await fetch("/api/map/locations").then(r => r.json());
    latestMapData = data;
    const units = data.units || {};
    const unitSelect = document.getElementById("map-unit-select");
    if (unitSelect) unitSelect.innerHTML = Object.keys(units).sort().map(u => `<option value="${escapeHTML(u)}">${escapeHTML(u)}</option>`).join("");
    for (const [unit, rec] of Object.entries(units)) {
      const [lat, lon] = getAverageLatLon(rec);
      setAverageMarker(unit, lat, lon, rec);
    }
    const overrides = data.overrides || {};
    const enabled = !!overrides.enabled;
    const enabledBox = document.getElementById("map-override-enabled");
    if (enabledBox) enabledBox.checked = enabled;
    if (overrides.units && !overrideDirty) {
      overrideLayer.clearLayers(); overrideMarkers = {};
      for (const [unit, rec] of Object.entries(overrides.units)) {
        if (rec.enabled !== false && validLatLon(rec.lat, rec.lon)) setOverrideMarker(unit, rec.lat, rec.lon, true, rec.position_sigma_m || 1.5);
      }
      overrideDirty = false;
    }
    renderOverrideTable();
    fitMapToKnownMarkers();
    setMapStatus("Map locations refreshed.", "good-text");
  } catch (e) {
    setMapStatus(`Map location refresh failed: ${e}`, "bad-text");
    console.error(e);
  }
}
async function createOverridesFromAverages() {
  initMap();

  if (!aruMap || !overrideLayer) {
    setMapStatus("Map is not ready yet; Leaflet may not have loaded.", "bad-text");
    return;
  }

  setMapStatus("Loading averaged locations…", "warn-text");

  let data;
  try {
    data = await fetchMapLocationsNoCache();
  } catch (e) {
    setMapStatus(`Could not load averaged locations: ${e}`, "bad-text");
    console.error("createOverridesFromAverages: fetch failed", e);
    return;
  }

  const units = data?.units || {};
  const validEntries = [];

  for (const [unit, rec] of Object.entries(units)) {
    const [lat, lon] = getAverageLatLon(rec);
    if (validLatLon(lat, lon)) {
      validEntries.push([unit, rec, lat, lon]);
    } else {
      console.warn("Skipping unit without valid averaged coordinates", unit, rec);
    }
  }

  if (validEntries.length === 0) {
    setMapStatus(
      `No valid averaged locations found. Units returned: ${Object.keys(units).join(", ") || "none"}`,
      "bad-text"
    );
    console.warn("Map locations payload had no valid averaged coordinates:", data);
    return;
  }

  overrideLayer.clearLayers();
  overrideMarkers = {};

  for (const [unit, rec, lat, lon] of validEntries) {
    const sigma =
      rec.position_sigma_m ??
      rec.scatter_horizontal_m ??
      rec.median_eph_m ??
      1.5;
    setOverrideMarker(unit, lat, lon, true, sigma);
  }

  const enabledBox = document.getElementById("map-override-enabled");
  if (enabledBox) enabledBox.checked = true;

  fitMapToKnownMarkers();
  overrideDirty = true;

  setMapStatus(
    `Created ${validEntries.length} draggable override pins from averaged locations. Drag and save them.`,
    "good-text"
  );
}
function collectOverridePayload() {
  const units = {};
  for (const [unit, marker] of Object.entries(overrideMarkers)) {
    const ll = marker.getLatLng();
    units[unit] = { lat: ll.lat, lon: ll.lng, enabled: true, position_sigma_m: marker.aru?.position_sigma_m || 1.5 };
  }
  return { enabled: checked("map-override-enabled"), units };
}
async function saveMapOverrides() {
  try {
    const payload = collectOverridePayload();
    if (Object.keys(payload.units).length === 0) return alert("No override pins to save. Create pins from averages or place pins on the map first.");
    const res = await postJSON("/api/map/location-overrides", payload);
    overrideDirty = false;
    setMapStatus(`Saved override file: ${res.path}`, "good-text");
    await refreshAll();
  } catch (e) { alert(`Could not save map overrides: ${e}`); }
}
async function clearMapOverrides() {
  if (!confirm("Clear saved override pins?")) return;
  try {
    const r = await fetch("/api/map/location-overrides", { method: "DELETE" });
    if (!r.ok) throw new Error(await r.text());
    if (overrideLayer) overrideLayer.clearLayers();
    overrideMarkers = {};
    overrideDirty = false;
    renderOverrideTable();
    setMapStatus("Override pins cleared.", "warn-text");
    await refreshAll();
  } catch (e) { alert(`Could not clear overrides: ${e}`); }
}
function enableClickToPlacePin() {
  initMap();
  clickToPlace = true;
  const unit = val("map-unit-select") || "selected unit";
  setMapStatus(`Click the map to place override pin for ${unit}.`, "warn-text");
}
function clearResultLayers() {
  if (resultLayer) resultLayer.clearLayers();
  setMapStatus("Result layers cleared.", "warn-text");
}
async function loadLatestResultLayers(kind) {
  initMap();
  if (!aruMap) return;
  try {
    const data = await fetch(`/api/map/result-layers?kind=${encodeURIComponent(kind)}`).then(r => r.json());
    if (!data.artifacts || data.artifacts.length === 0) { setMapStatus(`No ${kind} outputs found.`, "bad-text"); return; }
    clearResultLayers();
    let drew = false;
    for (const art of data.geojson || []) {
      const gj = await fetch(art.url).then(r => r.json());
      const layer = L.geoJSON(gj, {
        pointToLayer: (feature, latlng) => L.circleMarker(latlng, { radius: feature.properties?.radius || 7, weight: 2, fillOpacity: 0.85 }),
        onEachFeature: (feature, layer) => {
          const p = feature.properties || {};
          const title = p.name || p.unit || p.kind || art.name;
          layer.bindPopup(`<strong>${escapeHTML(title)}</strong><br>${escapeHTML(JSON.stringify(p, null, 2))}`);
        }
      });
      layer.addTo(resultLayer); drew = true;
    }
    if (drew) {
      const b = resultLayer.getBounds?.();
      if (b && b.isValid()) aruMap.fitBounds(b, { padding: [30,30], maxZoom: 21 });
      setMapStatus(`Loaded ${kind} GeoJSON layer(s) from ${data.dir}.`, "good-text");
    } else if ((data.html || []).length > 0) {
      previewArtifact(data.html[0]);
      setMapStatus(`No GeoJSON for latest ${kind}; opened Folium HTML in Artifact Preview.`, "warn-text");
    }
  } catch (e) {
    setMapStatus(`Could not load ${kind} result layers: ${e}`, "bad-text");
    console.error(e);
  }
}

async function saveMapOverridesSilentlyIfNeeded(useOverrides) {
  if (!useOverrides || !overrideDirty || Object.keys(overrideMarkers).length === 0) return;
  const payload = collectOverridePayload();
  await postJSON("/api/map/location-overrides", payload);
  overrideDirty = false;
  setMapStatus("Saved pending override pin edits before starting job.", "good-text");
}

// -----------------------------------------------------------------------------
// Pipeline job submitters
// -----------------------------------------------------------------------------
async function submitAcquire() {
  try {
    const payload = {timestamp:val("acq-timestamp"), clip_half_s:Number(val("acq-half")||30), units:val("acq-units")||"zero,one,four,five", force_refetch:checked("acq-force"), no_rotate_if_active:checked("acq-no-rotate")};
    if (!payload.timestamp) return alert("Timestamp is required.");
    await postJSON("/api/jobs/acquire", payload);
    await refreshAll();
  } catch(e) { alert(`Acquire failed to start: ${e}`); }
}
async function submitCalibrate() {
  try {
    const payload = {timestamp:val("cal-timestamp"), clip_half_s:Number(val("cal-half")||10), units:"zero,one,four,five", ref:"five", source_unit:val("cal-source-unit")||"five", source_lat:val("cal-source-lat"), source_lon:val("cal-source-lon"), event_ref_offset:val("cal-offset"), calibration_txt:val("cal-txt")||"active", use_location_override:checked("cal-use-overrides")};
    if (!payload.timestamp) return alert("Timestamp is required.");
    if ((payload.source_lat && !payload.source_lon)||(!payload.source_lat&&payload.source_lon)) return alert("Provide both source latitude and longitude, or neither.");
    await saveMapOverridesSilentlyIfNeeded(payload.use_location_override);
    await postJSON("/api/jobs/calibrate", payload);
    await refreshAll();
  } catch(e) { alert(`Calibration failed to start: ${e}`); }
}
async function submitLocalize() {
  try {
    const payload = {timestamp:val("loc-timestamp"), clip_half_s:Number(val("loc-half")||30), units:"zero,one,four,five", ref:"five", mode:val("loc-mode")||"impulse", calibration:val("loc-calibration")||"active", event_ref_offset:val("loc-offset"), bird_bandpass:val("loc-bird-bandpass")||"1000,9000", use_location_override:checked("loc-use-overrides")};
    if (!payload.timestamp) return alert("Timestamp is required.");
    await saveMapOverridesSilentlyIfNeeded(payload.use_location_override);
    await postJSON("/api/jobs/localize", payload);
    await refreshAll();
  } catch(e) { alert(`Localization failed to start: ${e}`); }
}

// -----------------------------------------------------------------------------
// Artifacts and status rendering
// -----------------------------------------------------------------------------
function artifactLinks(artifacts, max=8) {
  if (!artifacts || artifacts.length===0) return "—";
  return artifacts.slice(0,max).map(a => `<button class="link-button" onclick='previewArtifact(${JSON.stringify(a)})'>${escapeHTML(a.name)}</button>`).join(" ");
}
async function previewArtifact(a) {
  const box=document.getElementById("artifact-preview"); if (!a || !box) return;
  if (a.kind==="html") box.innerHTML=`<div class="preview-title">${escapeHTML(a.path)}</div><iframe src="${a.url}"></iframe>`;
  else if (a.kind==="image") box.innerHTML=`<div class="preview-title">${escapeHTML(a.path)}</div><img src="${a.url}" alt="${escapeHTML(a.name)}">`;
  else if (a.kind==="text" || a.kind==="json" || a.kind==="geojson") { const r=await fetch(a.url); const text=await r.text(); box.innerHTML=`<div class="preview-title">${escapeHTML(a.path)}</div><pre>${escapeHTML(text)}</pre>`; }
  else window.open(a.url,"_blank");
  box.className="preview";
}
async function loadJobLog(jobId) { const r=await fetch(`/api/jobs/${jobId}/log`); const text=await r.text(); const url=URL.createObjectURL(new Blob([text], {type:"text/plain"})); previewArtifact({kind:"text", name:`${jobId}.log`, path:`job ${jobId}`, url}); }
function renderJobs(data) {
  const body=document.getElementById("jobs-body"); if(!body) return; body.innerHTML="";
  for (const j of (data.jobs||[])) {
    const row=document.createElement("tr");
    const outputs=Object.entries(j.outputs||{}).map(([k,v])=>`<div><span class="small">${escapeHTML(k)}</span>: ${escapeHTML(v)}</div>`).join("")||"—";
    const st=j.status==="succeeded"?"good":j.status==="failed"?"bad":j.status==="running"?"warn":"neutral";
    row.innerHTML=`<td>${shortTime(j.created_at)}<br><span class="small">${escapeHTML(j.job_id)}</span></td><td>${escapeHTML(j.kind)}</td><td>${pill(st)}<br><button class="link-button" onclick="loadJobLog('${j.job_id}')">log</button></td><td>${outputs}</td><td>${artifactLinks(j.artifacts,12)}</td><td>${escapeHTML(j.error||"")}</td>`;
    body.appendChild(row);
  }
}
function renderArtifactGroup(el, rows, emptyLabel) {
  if(!el) return;
  if(!rows||rows.length===0) { el.textContent=emptyLabel; return; }
  el.innerHTML=rows.map(row=>{
    const arts=artifactLinks(row.artifacts||[row],10);
    const title=row.id||row.name||row.path;
    const active=row.active?` <span class="pill good">active</span>`:"";
    const activate=row.path&&row.name&&row.name.endsWith(".txt")?` <button class="link-button" onclick='activateCalibration(${JSON.stringify(row.path)})'>activate</button>`:"";
    return `<div class="artifact-row"><strong>${escapeHTML(title)}</strong>${active}${activate}<br><span class="small">${escapeHTML(row.path||"")}</span><br>${arts}</div>`;
  }).join("");
}
async function activateCalibration(path) { try { await postJSON("/api/calibrations/activate", {path}); await refreshAll(); } catch(e) { alert(`Could not activate calibration: ${e}`); } }

async function refreshStatus() {
  const r=await fetch("/api/status");
  const data=await r.json();
  setText("last-update", shortTime(data.last_update));
  const locState=data.location?.state||"—";
  const locClass=locState==="collecting"?"good":locState==="frozen"?"warn":"neutral";
  const locEl=document.getElementById("location-state");
  if (locEl) locEl.outerHTML=`<span id="location-state" class="pill ${locClass}">${locState}</span>`;
  const statusBody=document.getElementById("status-body"); if (statusBody) statusBody.innerHTML="";
  const locationBody=document.getElementById("location-body"); if (locationBody) locationBody.innerHTML="";
  const units=data.units||{}; const summaries=data.location?.summaries||{};
  for (const unit of Object.keys(units).sort()) {
    const s=units[unit]; const summary=summaries[unit]||{};
    const gpsText=s.gps_has_fix?`${s.gps_mode}D`:"no";
    const gpsClass=s.gps_mode===3?"good":s.gps_mode===2?"warn":"bad";
    const diskText=s.disk_free_gb!==null&&s.disk_free_gb!==undefined?`${num(s.disk_free_gb,1)} GB`:"—";
    const wifiText=s.wifi_connected?`${s.wifi_signal_dbm ?? "?"} dBm`:"no";
    if (statusBody) {
      const row=document.createElement("tr");
      row.innerHTML=`<td><strong>${unit}</strong></td><td>${pill(s.health)}</td><td>${fmtBool(s.reachable,"yes","no")}</td><td><span class="pill ${gpsClass}">${gpsText}</span></td><td>${fmtBool(s.pps_selected,s.pps_source||"yes","no")}</td><td>${fmtBool(s.sbts_running,"run","off")}</td><td>${fmtBool(s.jackd_running,"run","off")}</td><td>${diskText}</td><td>${wifiText}</td><td>${summary.n_fixes ?? 0}</td><td>${shortTime(s.last_seen)}</td>`;
      statusBody.appendChild(row);
    }
    if (locationBody) {
      const latest=s.gps||{};
      const locRow=document.createElement("tr");
      locRow.innerHTML=`<td><strong>${unit}</strong></td><td>${pill(summary.quality)}</td><td>${coords(latest.lat,latest.lon)}<br><span class="small">eph ${num(latest.eph_m,1)} m</span></td><td>${coords(summary.lat_deg,summary.lon_deg)}<br><span class="small">alt ${num(summary.alt_m,1)} m</span></td><td>${summary.n_fixes ?? 0}</td><td>${num(summary.median_eph_m,1)} m</td><td>${num(summary.scatter_horizontal_m,1)} m</td>`;
      locationBody.appendChild(locRow);
    }
  }
  setText("logs", (data.logs||[]).slice(-80).join("\n")||"No logs yet.");
}
async function refreshAll() {
  await refreshStatus();
  try {
    const [cfg,jobs,events,calibrations,localizations]=await Promise.all([
      fetch("/api/dashboard-config").then(r=>r.json()),
      fetch("/api/jobs").then(r=>r.json()),
      fetch("/api/events").then(r=>r.json()),
      fetch("/api/calibrations").then(r=>r.json()),
      fetch("/api/localizations").then(r=>r.json())
    ]);
    setText("active-calibration", cfg.active_calibration||"none");
    const locOverride = cfg.location_override || {};
    setText("active-location-override", locOverride.enabled ? (locOverride.path || "enabled") : "disabled");
    const overrideBox = document.getElementById("map-override-enabled");
    if (overrideBox && document.activeElement !== overrideBox) overrideBox.checked = !!locOverride.enabled;
    renderJobs(jobs);
    renderArtifactGroup(document.getElementById("events-list"), events.events, "No fetched events yet.");
    renderArtifactGroup(document.getElementById("calibrations-list"), calibrations.calibrations, "No calibrations yet.");
    renderArtifactGroup(document.getElementById("localizations-list"), localizations.localizations, "No localizations yet.");
  } catch(e) { console.warn("refreshAll partial failure", e); }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  refreshMapLocations();
});
refreshAll();
setInterval(refreshAll, 3000);
setInterval(refreshMapLocations, 15000);
