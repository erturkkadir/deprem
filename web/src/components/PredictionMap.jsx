import { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default marker icons for Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

// Custom marker icons
const createIcon = (color) => new L.DivIcon({
  className: 'custom-marker',
  html: `<div style="
    background-color: ${color};
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 3px solid white;
    box-shadow: 0 2px 5px rgba(0,0,0,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
  "></div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  popupAnchor: [0, -12],
});

const predictedIcon = createIcon('#f97316'); // Orange
const matchedIcon = createIcon('#22c55e');   // Green
const missedIcon = createIcon('#ef4444');    // Red

// Component to fit bounds
function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds && bounds.length > 0) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 6 });
    }
  }, [map, bounds]);
  return null;
}

export default function PredictionMap({ prediction, onClose }) {
  if (!prediction) return null;

  const predLat = prediction.predicted_lat;
  const predLon = prediction.predicted_lon;
  const actualLat = prediction.actual_lat;
  const actualLon = prediction.actual_lon;
  const hasActual = actualLat !== null && actualLat !== undefined;
  const isMatched = prediction.verified && prediction.correct;

  // Calculate bounds to fit both markers
  const bounds = [];
  if (predLat !== null && predLon !== null) {
    bounds.push([predLat, predLon]);
  }
  if (hasActual) {
    bounds.push([actualLat, actualLon]);
  }

  // Default center if no valid coordinates
  const defaultCenter = bounds.length > 0
    ? bounds[0]
    : [0, 0];

  const formatTime = (timeStr) => {
    if (!timeStr) return '—';
    try {
      const date = new Date(timeStr);
      return date.toLocaleString();
    } catch {
      return '—';
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-zinc-900 rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden border border-zinc-700">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-700">
          <div className="flex items-center gap-3">
            <span className="text-xl font-bold text-orange-500">#{prediction.id}</span>
            <span className="text-zinc-400 text-sm">
              {formatTime(prediction.prediction_time)}
            </span>
            {prediction.verified && (
              <span className={`px-2 py-0.5 rounded text-xs ${
                isMatched ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
              }`}>
                {isMatched ? 'Matched' : 'Missed'}
              </span>
            )}
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
        <div className="h-[400px] relative">
          <MapContainer
            center={defaultCenter}
            zoom={3}
            style={{ height: '100%', width: '100%' }}
            className="z-0"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {bounds.length > 0 && <FitBounds bounds={bounds} />}

            {/* Predicted location marker */}
            {predLat !== null && predLon !== null && (
              <>
                <Circle
                  center={[predLat, predLon]}
                  radius={250000}
                  pathOptions={{
                    color: '#f97316',
                    fillColor: '#f97316',
                    fillOpacity: 0.1,
                    weight: 2,
                    dashArray: '5, 5'
                  }}
                />
                <Marker position={[predLat, predLon]} icon={predictedIcon}>
                  <Popup>
                    <div className="text-sm">
                      <div className="font-bold text-orange-600 mb-1">Predicted Location</div>
                      <div><strong>Lat:</strong> {predLat?.toFixed(2)}°</div>
                      <div><strong>Lon:</strong> {predLon?.toFixed(2)}°</div>
                      <div><strong>Mag:</strong> {prediction.predicted_mag?.toFixed(1)}</div>
                      <div><strong>Time:</strong> {prediction.predicted_dt} min</div>
                      {prediction.predicted_place && (
                        <div className="mt-1 text-gray-600">{prediction.predicted_place}</div>
                      )}
                    </div>
                  </Popup>
                </Marker>
              </>
            )}

            {/* Actual location marker */}
            {hasActual && (
              <Marker
                position={[actualLat, actualLon]}
                icon={isMatched ? matchedIcon : missedIcon}
              >
                <Popup>
                  <div className="text-sm">
                    <div className={`font-bold mb-1 ${isMatched ? 'text-green-600' : 'text-red-600'}`}>
                      Actual Earthquake
                    </div>
                    <div><strong>Lat:</strong> {actualLat?.toFixed(2)}°</div>
                    <div><strong>Lon:</strong> {actualLon?.toFixed(2)}°</div>
                    <div><strong>Mag:</strong> {prediction.actual_mag?.toFixed(1)}</div>
                    <div><strong>Time:</strong> {prediction.actual_dt} min</div>
                    {prediction.actual_place && (
                      <div className="mt-1 text-gray-600">{prediction.actual_place}</div>
                    )}
                  </div>
                </Popup>
              </Marker>
            )}
          </MapContainer>
        </div>

        {/* Legend & Stats */}
        <div className="p-4 border-t border-zinc-700">
          <div className="flex flex-wrap gap-4 justify-between">
            {/* Legend */}
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-orange-500 border-2 border-white"></div>
                <span className="text-zinc-400">Predicted</span>
              </div>
              {hasActual && (
                <div className="flex items-center gap-1.5">
                  <div className={`w-3 h-3 rounded-full border-2 border-white ${isMatched ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <span className="text-zinc-400">Actual {isMatched ? '(Match)' : '(Miss)'}</span>
                </div>
              )}
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-0.5 border-t-2 border-dashed border-orange-500"></div>
                <span className="text-zinc-400">250km radius</span>
              </div>
            </div>

            {/* Distance info */}
            {hasActual && prediction.diff_lat !== null && prediction.diff_lon !== null && (
              <div className="text-xs text-zinc-400">
                <span className="text-zinc-500">Distance error: </span>
                <span className={isMatched ? 'text-green-400' : 'text-red-400'}>
                  {prediction.diff_lat}° lat, {prediction.diff_lon}° lon
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
