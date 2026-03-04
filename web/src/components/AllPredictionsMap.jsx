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

const matchedIcon = createIcon('#22c55e');

export default function AllPredictionsMap({ onClose }) {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // all | matched | missed | pending

  useEffect(() => {
    axios.get('/api/predictions?limit=100&page=1')
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
    if (s === 'matched') return '#22c55e';
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
                      {status === 'matched' && pred.actual_lat != null && (
                        <div className="mt-1 pt-1 border-t border-gray-200">
                          <div className="text-green-600 font-medium">Actual EQ</div>
                          <div>{Number(pred.actual_lat).toFixed(2)}°, {Number(pred.actual_lon).toFixed(2)}°</div>
                          {pred.actual_place && <div className="text-gray-500 text-xs">{pred.actual_place}</div>}
                        </div>
                      )}
                    </div>
                  </Popup>
                </Circle>
              );
            })}

            {/* Actual earthquake pins for matched predictions */}
            {visible
              .filter(p => getStatus(p) === 'matched' && p.actual_lat != null && p.actual_lon != null)
              .map(pred => (
                <Marker
                  key={`actual-${pred.id}`}
                  position={[pred.actual_lat, pred.actual_lon]}
                  icon={matchedIcon}
                >
                  <Popup>
                    <div className="text-sm">
                      <div className="font-bold text-green-600 mb-1">Actual EQ (#{pred.id})</div>
                      <div>{Number(pred.actual_lat).toFixed(2)}°, {Number(pred.actual_lon).toFixed(2)}°</div>
                      <div><strong>Mag:</strong> M{Number(pred.actual_mag).toFixed(1)}</div>
                      {pred.actual_place && <div className="text-gray-500 text-xs">{pred.actual_place}</div>}
                    </div>
                  </Popup>
                </Marker>
              ))}
          </MapContainer>
        )}
      </div>
    </div>
  );
}
