import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import axios from 'axios';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

const createIcon = (color) => new L.DivIcon({
  className: '',
  html: `<div style="background:${color};width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,0.5)"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
  popupAnchor: [0, -8],
});

// Hours after window end: 0 = caught in window (green) → 48+ = very late (red)
function lateHours(pred) {
  if (!pred.prediction_time || !pred.actual_time) return 0;
  const fix = (s) => (s.endsWith('Z') || s.includes('+')) ? s : s + 'Z';
  const predT = new Date(fix(pred.prediction_time));
  const actT  = new Date(fix(pred.actual_time));
  const windowMs = (pred.predicted_dt || 15) * 60000;
  const windowEnd = new Date(predT.getTime() + windowMs);
  return Math.max(0, (actT - windowEnd) / 3600000);
}

// Green (in-window) → yellow (24h late) → red (48h late)
function matchedColor(pred) {
  const h = lateHours(pred);
  const t = Math.min(h / 48, 1);           // 0 → 1 over 0-48h
  const hue = 120 * (1 - t);               // 120=green → 0=red
  return `hsl(${hue.toFixed(0)}, 75%, 45%)`;
}

const matchedIcon = createIcon('#22c55e');

export default function AllPredictionsMap({ onClose, initialFilter = 'all' }) {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState(initialFilter); // all | matched | missed | pending

  useEffect(() => {
    // Fetch all predictions (limit 5000 covers any realistic volume)
    axios.get('/api/predictions?limit=5000&page=1')
      .then(r => {
        setPredictions(r.data.predictions || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const getStatus = (pred) => {
    if (!pred.verified) return 'pending';
    return pred.correct ? 'matched' : 'missed';
  };

  const visible = predictions.filter(pred => {
    if (filter === 'all') return true;
    return getStatus(pred) === filter;
  });

  const counts = {
    all: predictions.length,
    matched: predictions.filter(p => getStatus(p) === 'matched').length,
    missed: predictions.filter(p => getStatus(p) === 'missed').length,
    pending: predictions.filter(p => getStatus(p) === 'pending').length,
  };

  const circleColor = (pred) => {
    const s = getStatus(pred);
    if (s === 'matched') return matchedColor(pred);  // gradient: green (in-window) → red (48h late)
    if (s === 'missed')  return '#ef4444';
    return '#f97316';
  };

  return (
    <div className="fixed inset-0 bg-black/85 z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-zinc-900 border-b border-zinc-700 shrink-0">
        <div className="flex items-center gap-4">
          <h2 className="text-orange-500 font-bold text-lg">All Predictions Map</h2>
          <div className="flex gap-1.5 text-xs">
            {[
              { key: 'all',     label: 'All',     color: 'bg-zinc-700 text-zinc-200' },
              { key: 'pending', label: 'Pending',  color: 'bg-yellow-900/60 text-yellow-400' },
              { key: 'matched', label: 'Matched',  color: 'bg-green-900/60 text-green-400' },
              { key: 'missed',  label: 'Missed',   color: 'bg-red-900/60 text-red-400' },
            ].map(f => (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className={`px-2 py-1 rounded font-medium transition-all ${
                  filter === f.key
                    ? `${f.color} ring-1 ring-current`
                    : 'bg-zinc-800 text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {f.label} <span className="opacity-70">{counts[f.key]}</span>
              </button>
            ))}
          </div>
          {/* Legend */}
          <div className="hidden sm:flex gap-3 text-xs text-zinc-400 ml-2">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-500 inline-block border border-white/40"></span>Pending</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500 inline-block border border-white/40"></span>Matched</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 inline-block border border-white/40"></span>Missed</span>
            <span className="flex items-center gap-1 ml-1">
              <span className="w-4 h-0 border-t border-dashed border-zinc-400 inline-block"></span>
              250km radius
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-zinc-400 hover:text-white transition-colors p-1"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Map */}
      <div className="flex-1 relative">
        {/* Gradient legend — shown when viewing matched predictions */}
        {!loading && filter === 'matched' && (
          <div className="absolute top-4 left-4 z-[1000] bg-zinc-900/90 backdrop-blur-sm border border-zinc-700 rounded-lg p-3 shadow-xl">
            <div className="text-xs font-semibold text-zinc-300 mb-2">Match Timing</div>
            <div className="flex items-stretch gap-2">
              <div
                className="w-4 rounded"
                style={{
                  height: '120px',
                  background: 'linear-gradient(to bottom, hsl(120,75%,45%), hsl(60,75%,45%), hsl(0,75%,45%))',
                }}
              />
              <div className="flex flex-col justify-between text-[10px] text-zinc-400 font-mono py-0.5">
                <span>In window</span>
                <span>12h late</span>
                <span>24h late</span>
                <span>36h late</span>
                <span>48h+ late</span>
              </div>
            </div>
          </div>
        )}
        {loading ? (
          <div className="flex items-center justify-center h-full text-zinc-400">
            <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin mr-3" />
            Loading predictions...
          </div>
        ) : (
          <MapContainer
            center={[20, 0]}
            zoom={2}
            style={{ height: '100%', width: '100%' }}
            className="z-0"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {visible.map(pred => {
              const lat = pred.predicted_lat;
              const lon = pred.predicted_lon;
              if (lat == null || lon == null) return null;
              const color = circleColor(pred);
              const status = getStatus(pred);

              return (
                <Circle
                  key={pred.id}
                  center={[lat, lon]}
                  radius={250000}
                  pathOptions={{
                    color,
                    fillColor: color,
                    fillOpacity: status === 'matched' ? 0.15 : 0.08,
                    weight: status === 'matched' ? 2 : 1.5,
                    dashArray: status === 'pending' ? '4 4' : undefined,
                  }}
                >
                  <Popup>
                    <div className="text-sm min-w-[160px]">
                      <div className="font-bold mb-1" style={{ color }}>
                        #{pred.id} — {status.charAt(0).toUpperCase() + status.slice(1)}
                      </div>
                      <div><strong>Predicted:</strong> {Number(pred.predicted_lat).toFixed(2)}°, {Number(pred.predicted_lon).toFixed(2)}°</div>
                      <div><strong>Mag:</strong> M{Number(pred.predicted_mag).toFixed(1)}</div>
                      {pred.predicted_place && <div className="text-gray-500 text-xs mt-0.5">{pred.predicted_place}</div>}
                      {status === 'matched' && pred.actual_lat != null && (() => {
                        const h = lateHours(pred);
                        return (
                          <div className="mt-1 pt-1 border-t border-gray-200">
                            <div className="font-medium" style={{ color }}>Actual EQ</div>
                            <div>{Number(pred.actual_lat).toFixed(2)}°, {Number(pred.actual_lon).toFixed(2)}°</div>
                            <div><strong>Late by:</strong> {h < 0.1 ? 'in window' : h < 1 ? `${Math.round(h * 60)}m` : `${h.toFixed(1)}h`}</div>
                            {pred.actual_place && <div className="text-gray-500 text-xs">{pred.actual_place}</div>}
                          </div>
                        );
                      })()}
                    </div>
                  </Popup>
                </Circle>
              );
            })}

            {/* Actual earthquake pins for matched predictions */}
            {visible
              .filter(p => getStatus(p) === 'matched' && p.actual_lat != null && p.actual_lon != null)
              .map(pred => {
                const color = matchedColor(pred);
                const h = lateHours(pred);
                return (
                  <Marker
                    key={`actual-${pred.id}`}
                    position={[pred.actual_lat, pred.actual_lon]}
                    icon={createIcon(color)}
                  >
                    <Popup>
                      <div className="text-sm">
                        <div className="font-bold mb-1" style={{ color }}>Actual EQ (#{pred.id})</div>
                        <div>{Number(pred.actual_lat).toFixed(2)}°, {Number(pred.actual_lon).toFixed(2)}°</div>
                        <div><strong>Mag:</strong> M{Number(pred.actual_mag).toFixed(1)}</div>
                        <div><strong>Late by:</strong> {h < 0.1 ? 'in window' : `${h.toFixed(1)}h`}</div>
                        {pred.actual_place && <div className="text-gray-500 text-xs">{pred.actual_place}</div>}
                      </div>
                    </Popup>
                  </Marker>
                );
              })}
          </MapContainer>
        )}
      </div>
    </div>
  );
}
