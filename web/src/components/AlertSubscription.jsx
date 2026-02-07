import { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, CircleMarker, Circle, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const API_URL = import.meta.env.VITE_API_URL || '';

function LocationPicker({ onSelect, selected, radiusKm }) {
  useMapEvents({
    click(e) {
      onSelect({ lat: e.latlng.lat, lon: e.latlng.lng });
    },
  });

  if (!selected) return null;

  return (
    <>
      <Circle
        center={[selected.lat, selected.lon]}
        radius={radiusKm * 1000}
        pathOptions={{ color: '#f97316', fillColor: '#f97316', fillOpacity: 0.08, weight: 1 }}
      />
      <CircleMarker
        center={[selected.lat, selected.lon]}
        radius={6}
        pathOptions={{ color: '#f97316', fillColor: '#f97316', fillOpacity: 1, weight: 2 }}
      />
    </>
  );
}

export default function AlertSubscription() {
  const [isOpen, setIsOpen] = useState(false);
  const [email, setEmail] = useState('');
  const [location, setLocation] = useState(null);
  const [radiusKm, setRadiusKm] = useState(500);
  const [status, setStatus] = useState(null); // { type: 'success'|'error'|'verify', msg: '' }
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [checkedEmail, setCheckedEmail] = useState('');

  const fetchAlerts = useCallback(async (emailToCheck) => {
    if (!emailToCheck) return;
    try {
      const res = await fetch(`${API_URL}/api/alerts/status?email=${encodeURIComponent(emailToCheck)}`);
      const data = await res.json();
      if (data.success) {
        setAlerts(data.alerts || []);
        setCheckedEmail(emailToCheck);
      }
    } catch {
      // silent
    }
  }, []);

  const handleSubscribe = async (e) => {
    e.preventDefault();
    if (!email || !location) return;

    setIsSubmitting(true);
    setStatus(null);

    try {
      const res = await fetch(`${API_URL}/api/alerts/subscribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, lat: location.lat, lon: location.lon, radius_km: radiusKm }),
      });
      const data = await res.json();

      if (data.success) {
        setStatus({
          type: 'verify',
          msg: `Verification email sent to ${email}. Please check your inbox and click the link to activate alerts.`
        });
        fetchAlerts(email);
      } else {
        setStatus({ type: 'error', msg: data.error || 'Failed to subscribe' });
      }
    } catch (err) {
      setStatus({ type: 'error', msg: 'Network error' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUnsubscribe = async (token) => {
    try {
      const res = await fetch(`${API_URL}/api/alerts/unsubscribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      const data = await res.json();
      if (data.success) {
        setAlerts(prev => prev.filter(a => a.token !== token));
      }
    } catch {
      // silent
    }
  };

  // Check for existing alerts when email changes (debounced)
  useEffect(() => {
    if (!email || !email.includes('@')) return;
    const timer = setTimeout(() => fetchAlerts(email), 800);
    return () => clearTimeout(timer);
  }, [email, fetchAlerts]);

  return (
    <section className="py-4">
      <div className="max-w-7xl mx-auto px-4">
        <div className="card">
          {/* Header - always visible */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full flex items-center justify-between text-left"
          >
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <h3 className="text-white font-bold text-sm sm:text-base">Get Earthquake Alerts</h3>
              <span className="text-zinc-500 text-xs hidden sm:inline">Receive email when predictions are made near you</span>
            </div>
            <svg
              className={`w-5 h-5 text-zinc-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Collapsible content */}
          {isOpen && (
            <div className="mt-4 pt-4 border-t border-zinc-700">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Map */}
                <div>
                  <p className="text-zinc-400 text-xs mb-2">Click on the map to set your location:</p>
                  <div className="rounded-lg overflow-hidden border border-zinc-700" style={{ height: '250px' }}>
                    <MapContainer
                      center={[20, 0]}
                      zoom={2}
                      style={{ height: '100%', width: '100%' }}
                      scrollWheelZoom={true}
                    >
                      <TileLayer
                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                        attribution='&copy; <a href="https://carto.com">CARTO</a>'
                      />
                      <LocationPicker onSelect={setLocation} selected={location} radiusKm={radiusKm} />
                    </MapContainer>
                  </div>
                  {location && (
                    <p className="text-zinc-500 text-xs mt-1">
                      Selected: {location.lat.toFixed(2)}, {location.lon.toFixed(2)}
                    </p>
                  )}
                </div>

                {/* Form */}
                <div className="flex flex-col justify-between">
                  <form onSubmit={handleSubscribe} className="space-y-3">
                    <div>
                      <label className="text-zinc-400 text-xs block mb-1">Email</label>
                      <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="your@email.com"
                        required
                        className="w-full bg-zinc-700 text-white text-sm rounded px-3 py-2 border border-zinc-600 focus:outline-none focus:border-orange-500"
                      />
                    </div>

                    <div>
                      <label className="text-zinc-400 text-xs block mb-1">Alert Radius</label>
                      <select
                        value={radiusKm}
                        onChange={(e) => setRadiusKm(Number(e.target.value))}
                        className="w-full bg-zinc-700 text-white text-sm rounded px-3 py-2 border border-zinc-600 focus:outline-none focus:border-orange-500"
                      >
                        <option value={250}>250 km</option>
                        <option value={500}>500 km</option>
                        <option value={750}>750 km</option>
                        <option value={1000}>1000 km</option>
                      </select>
                    </div>

                    <button
                      type="submit"
                      disabled={!email || !location || isSubmitting}
                      className="btn-primary w-full text-sm py-2"
                    >
                      {isSubmitting ? 'Sending verification...' : 'Subscribe'}
                    </button>
                  </form>

                  {/* Status message */}
                  {status && (
                    <div className={`mt-3 px-3 py-2 rounded text-xs ${
                      status.type === 'verify'
                        ? 'bg-blue-900/30 text-blue-400 border border-blue-800'
                        : status.type === 'success'
                          ? 'bg-green-900/30 text-green-400 border border-green-800'
                          : 'bg-red-900/30 text-red-400 border border-red-800'
                    }`}>
                      {status.type === 'verify' && (
                        <div className="flex items-start gap-2">
                          <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          </svg>
                          <span>{status.msg}</span>
                        </div>
                      )}
                      {status.type !== 'verify' && status.msg}
                    </div>
                  )}

                  {/* Existing subscriptions */}
                  {alerts.length > 0 && (
                    <div className="mt-3">
                      <p className="text-zinc-400 text-xs mb-2">Your subscriptions:</p>
                      <div className="space-y-1 max-h-32 overflow-y-auto custom-scrollbar">
                        {alerts.map((alert) => (
                          <div key={alert.id} className="flex items-center justify-between bg-zinc-800 rounded px-2 py-1.5 text-xs">
                            <div className="flex items-center gap-2">
                              {alert.verified && alert.active ? (
                                <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" title="Active" />
                              ) : (
                                <span className="w-2 h-2 rounded-full bg-yellow-500 flex-shrink-0" title="Pending verification" />
                              )}
                              <span className="text-zinc-300">
                                {alert.lat.toFixed(1)}, {alert.lon.toFixed(1)}
                                <span className="text-zinc-500 ml-1">({alert.radius_km}km)</span>
                              </span>
                              {!alert.verified && (
                                <span className="text-yellow-500 text-[10px]">check email</span>
                              )}
                            </div>
                            <button
                              onClick={() => handleUnsubscribe(alert.token)}
                              className="text-red-400 hover:text-red-300 ml-2"
                              title="Unsubscribe"
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                              </svg>
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
