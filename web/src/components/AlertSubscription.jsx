import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
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

export default function AlertSubscription({ alwaysOpen = false }) {
  const { t } = useTranslation();
  const [isOpen, setIsOpen] = useState(alwaysOpen);
  const [email, setEmail] = useState('');
  const [location, setLocation] = useState(null);
  const [radiusKm, setRadiusKm] = useState(500);
  const [status, setStatus] = useState(null);
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
          msg: t('alerts.verificationSent', { email })
        });
        fetchAlerts(email);
      } else {
        setStatus({ type: 'error', msg: data.error || t('alerts.failedToSubscribe') });
      }
    } catch (err) {
      setStatus({ type: 'error', msg: t('alerts.networkError') });
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

  useEffect(() => {
    if (!email || !email.includes('@')) return;
    const timer = setTimeout(() => fetchAlerts(email), 800);
    return () => clearTimeout(timer);
  }, [email, fetchAlerts]);

  const innerContent = (
            <div>

              {/* How it works */}
              <div className="mb-5 bg-zinc-800/50 rounded-lg p-4">
                <h4 className="text-zinc-300 text-xs font-semibold uppercase tracking-wider mb-3">{t('alerts.howItWorks')}</h4>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="flex items-start gap-2.5">
                    <span className="w-5 h-5 rounded-full bg-orange-500/20 text-orange-400 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">1</span>
                    <div>
                      <p className="text-white text-xs font-medium">{t('alerts.step1Title')}</p>
                      <p className="text-zinc-500 text-[11px] mt-0.5">{t('alerts.step1Desc')}</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2.5">
                    <span className="w-5 h-5 rounded-full bg-orange-500/20 text-orange-400 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">2</span>
                    <div>
                      <p className="text-white text-xs font-medium">{t('alerts.step2Title')}</p>
                      <p className="text-zinc-500 text-[11px] mt-0.5">{t('alerts.step2Desc')}</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-2.5">
                    <span className="w-5 h-5 rounded-full bg-orange-500/20 text-orange-400 text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">3</span>
                    <div>
                      <p className="text-white text-xs font-medium">{t('alerts.step3Title')}</p>
                      <p className="text-zinc-500 text-[11px] mt-0.5">{t('alerts.step3Desc')}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="mb-5 rounded-lg p-3 bg-amber-500/5 border border-amber-500/30 flex items-start gap-2.5">
                <svg className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <p className="text-amber-400/90 text-[11px] leading-relaxed">
                  <span className="font-semibold">{t('alerts.experimentalWarning')}</span> — {t('alerts.experimentalDesc')}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Map */}
                <div>
                  <p className="text-zinc-400 text-xs mb-2 flex items-center gap-1.5">
                    <svg className="w-3.5 h-3.5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    </svg>
                    {t('alerts.clickMapPrompt')}
                  </p>
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
                  {location ? (
                    <p className="text-zinc-400 text-xs mt-1.5 flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-orange-500 inline-block"></span>
                      {t('alerts.monitoringAt', { lat: location.lat.toFixed(2), lon: location.lon.toFixed(2), radius: radiusKm })}
                    </p>
                  ) : (
                    <p className="text-zinc-600 text-xs mt-1.5 italic">{t('alerts.noLocationSelected')}</p>
                  )}
                </div>

                {/* Form */}
                <div className="flex flex-col justify-between">
                  <form onSubmit={handleSubscribe} className="space-y-3">
                    <div>
                      <label className="text-zinc-400 text-xs block mb-1">{t('alerts.emailLabel')}</label>
                      <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder={t('alerts.emailPlaceholder')}
                        required
                        className="w-full bg-zinc-700 text-white text-sm rounded px-3 py-2 border border-zinc-600 focus:outline-none focus:border-orange-500"
                      />
                    </div>

                    <div>
                      <label className="text-zinc-400 text-xs block mb-1">
                        {t('alerts.radiusLabel')}
                        <span className="text-zinc-600 ml-1">{t('alerts.radiusHelp')}</span>
                      </label>
                      <select
                        value={radiusKm}
                        onChange={(e) => setRadiusKm(Number(e.target.value))}
                        className="w-full bg-zinc-700 text-white text-sm rounded px-3 py-2 border border-zinc-600 focus:outline-none focus:border-orange-500"
                      >
                        <option value={250}>{t('alerts.radius250')}</option>
                        <option value={500}>{t('alerts.radius500')}</option>
                        <option value={750}>{t('alerts.radius750')}</option>
                        <option value={1000}>{t('alerts.radius1000')}</option>
                      </select>
                    </div>

                    <button
                      type="submit"
                      disabled={!email || !location || isSubmitting}
                      className="btn-primary w-full text-sm py-2.5 flex items-center justify-center gap-2"
                    >
                      {isSubmitting ? (
                        <>
                          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          {t('alerts.subscribing')}
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                          </svg>
                          {t('alerts.subscribe')}
                        </>
                      )}
                    </button>

                    <p className="text-zinc-600 text-[11px] text-center">
                      {t('alerts.unsubscribeNote')}
                    </p>
                  </form>

                  {/* Status message */}
                  {status && (
                    <div className={`mt-3 px-3 py-2.5 rounded text-xs ${
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
                      <p className="text-zinc-400 text-xs mb-2">{t('alerts.activeSubscriptions')}</p>
                      <div className="space-y-1 max-h-32 overflow-y-auto custom-scrollbar">
                        {alerts.map((alert) => (
                          <div key={alert.id} className="flex items-center justify-between bg-zinc-800 rounded px-2 py-1.5 text-xs">
                            <div className="flex items-center gap-2">
                              {alert.verified && alert.active ? (
                                <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" title={t('alerts.statusActive')} />
                              ) : (
                                <span className="w-2 h-2 rounded-full bg-yellow-500 flex-shrink-0" title={t('alerts.statusPendingVerify')} />
                              )}
                              <span className="text-zinc-300">
                                {alert.lat.toFixed(1)}°, {alert.lon.toFixed(1)}°
                                <span className="text-zinc-500 ml-1">({alert.radius_km} km)</span>
                              </span>
                              {!alert.verified && (
                                <span className="text-yellow-500 text-[10px]">{t('alerts.checkEmailToVerify')}</span>
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
  );

  if (alwaysOpen) {
    return innerContent;
  }

  return (
    <section className="py-4">
      <div className="max-w-7xl mx-auto px-4">
        <div className="card">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full flex items-center justify-between text-left"
          >
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-orange-500/15 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </div>
              <div>
                <h3 className="text-white font-bold text-sm sm:text-base">{t('alerts.title')}</h3>
                <p className="text-zinc-500 text-xs mt-0.5">
                  {t('alerts.subtitle')}
                </p>
              </div>
            </div>
            <svg
              className={`w-5 h-5 text-zinc-400 transition-transform flex-shrink-0 ml-2 ${isOpen ? 'rotate-180' : ''}`}
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {isOpen && (
            <div className="mt-4 pt-4 border-t border-zinc-700">
              {innerContent}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
