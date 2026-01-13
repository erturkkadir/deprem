import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix default marker icons for Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

// Get color based on magnitude
const getMagColor = (mag) => {
  if (mag >= 7) return '#9333ea'; // Purple - Major
  if (mag >= 6) return '#dc2626'; // Red - Strong
  if (mag >= 5) return '#f97316'; // Orange - Moderate
  if (mag >= 4) return '#eab308'; // Yellow - Light
  if (mag >= 3) return '#22c55e'; // Green - Minor
  return '#06b6d4'; // Cyan - Micro
};

// Get size based on magnitude
const getMagRadius = (mag) => {
  if (mag >= 7) return 25;
  if (mag >= 6) return 20;
  if (mag >= 5) return 15;
  if (mag >= 4) return 12;
  if (mag >= 3) return 8;
  return 5;
};

// Get label for magnitude
const getMagLabel = (mag) => {
  if (mag >= 7) return 'Major';
  if (mag >= 6) return 'Strong';
  if (mag >= 5) return 'Moderate';
  if (mag >= 4) return 'Light';
  if (mag >= 3) return 'Minor';
  return 'Micro';
};

// Format time ago
const formatTimeAgo = (isoString) => {
  if (!isoString) return '';
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return '';
  const diff = Math.floor((new Date() - date) / 1000);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
};

// Format time for display
const formatTime = (date) => {
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
};

const formatDateTime = (date) => {
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

// Animated earthquake marker with ultra-dramatic effects
function AnimatedEarthquakeMarker({ earthquake, isNew, isRecent, opacity = 1, ageMinutes = 0 }) {
  const [ripplePhase, setRipplePhase] = useState(0);
  const [showRipples, setShowRipples] = useState(isRecent);
  const [pulsePhase, setPulsePhase] = useState(0);
  const [birthScale, setBirthScale] = useState(isNew ? 0 : 1);

  // Birth animation - markers expand from center when first appearing
  useEffect(() => {
    if (isNew) {
      setBirthScale(0);
      let scale = 0;
      const birthAnim = setInterval(() => {
        scale += 0.08;
        if (scale >= 1.2) {
          scale = 1;
          clearInterval(birthAnim);
        } else if (scale > 1) {
          scale = 1 + (1.2 - scale) * 0.5; // Bounce back
        }
        setBirthScale(scale);
      }, 16);
      return () => clearInterval(birthAnim);
    }
  }, [isNew]);

  // Multi-ring ripple effect for recent earthquakes
  useEffect(() => {
    if (isRecent || isNew) {
      setShowRipples(true);
      const interval = setInterval(() => {
        setRipplePhase(prev => (prev + 0.05) % 1);
      }, 30);

      // Keep rippling for recent earthquakes (10 minutes)
      const timeout = isNew ? setTimeout(() => {
        // Keep showing if still recent
      }, 3000) : null;

      return () => {
        clearInterval(interval);
        if (timeout) clearTimeout(timeout);
      };
    } else {
      setShowRipples(false);
    }
  }, [isRecent, isNew]);

  // Gentle pulse for all visible earthquakes
  useEffect(() => {
    if (opacity > 0.3) {
      const interval = setInterval(() => {
        setPulsePhase(prev => (prev + 0.02) % 1);
      }, 30);
      return () => clearInterval(interval);
    }
  }, [opacity]);

  const mag = earthquake.mag || 0;
  const color = getMagColor(mag);
  const baseRadius = getMagRadius(mag);

  // Dynamic radius based on age and pulse
  const ageScale = 0.5 + (opacity * 0.5); // Older = smaller
  const pulseScale = isRecent ? 1 + Math.sin(pulsePhase * Math.PI * 2) * 0.15 : 1;
  const radius = baseRadius * ageScale * pulseScale * birthScale;

  // Glow intensity based on freshness
  const glowIntensity = isRecent ? 1 : opacity * 0.5;

  // Generate multiple ripple rings at different phases
  const rippleRings = showRipples ? [0, 0.33, 0.66].map(offset => {
    const phase = (ripplePhase + offset) % 1;
    const scale = 1 + phase * 2.5;
    const ringOpacity = Math.max(0, 1 - phase);
    return { scale, opacity: ringOpacity * 0.6 };
  }) : [];

  return (
    <>
      {/* Outer glow aura for recent earthquakes */}
      {isRecent && (
        <CircleMarker
          center={[earthquake.lat, earthquake.lon]}
          radius={baseRadius * 2.5 * birthScale}
          pathOptions={{
            color: 'transparent',
            fillColor: color,
            fillOpacity: 0.15 + Math.sin(pulsePhase * Math.PI * 2) * 0.1,
            weight: 0,
          }}
        />
      )}

      {/* Multiple expanding ripple rings */}
      {rippleRings.map((ring, i) => (
        <CircleMarker
          key={`ripple-${i}`}
          center={[earthquake.lat, earthquake.lon]}
          radius={baseRadius * ring.scale * birthScale}
          pathOptions={{
            color: color,
            fillColor: 'transparent',
            fillOpacity: 0,
            weight: mag >= 5 ? 3 : 2,
            opacity: ring.opacity * glowIntensity,
            dashArray: mag >= 6 ? undefined : '4,4',
          }}
        />
      ))}

      {/* Inner bright core for very recent */}
      {isRecent && opacity > 0.8 && (
        <CircleMarker
          center={[earthquake.lat, earthquake.lon]}
          radius={Math.max(3, radius * 0.4)}
          pathOptions={{
            color: '#fff',
            fillColor: '#fff',
            fillOpacity: 0.9,
            weight: 1,
            opacity: 0.9,
          }}
        />
      )}

      {/* Main marker */}
      <CircleMarker
        center={[earthquake.lat, earthquake.lon]}
        radius={Math.max(2, radius)}
        pathOptions={{
          color: isRecent ? '#fff' : color,
          fillColor: color,
          fillOpacity: Math.max(0.1, opacity * 0.85),
          weight: isRecent ? 3 : Math.max(1, 2 * opacity),
          opacity: Math.max(0.15, opacity),
        }}
        eventHandlers={{
          mouseover: (e) => {
            e.target.setStyle({ fillOpacity: 1, weight: 4, opacity: 1 });
          },
          mouseout: (e) => {
            e.target.setStyle({
              fillOpacity: Math.max(0.1, opacity * 0.85),
              weight: isRecent ? 3 : Math.max(1, 2 * opacity),
              opacity: Math.max(0.15, opacity)
            });
          },
        }}
      >
        <Popup>
          <div className="text-sm min-w-[200px]">
            <div className="flex items-center gap-2 mb-2">
              <span
                className="px-2 py-1 rounded text-white font-bold text-sm"
                style={{ backgroundColor: color }}
              >
                M{mag.toFixed(1)}
              </span>
              <span className="text-gray-500 text-xs">{getMagLabel(mag)}</span>
              {isRecent && (
                <span className="px-1.5 py-0.5 bg-orange-500 text-white text-[10px] rounded animate-pulse">
                  NEW
                </span>
              )}
            </div>
            <div className="font-medium text-gray-800 mb-2">{earthquake.place || 'Unknown Location'}</div>
            <div className="text-xs text-gray-600 space-y-1">
              <div><strong>Time:</strong> {formatDateTime(new Date(earthquake.time))}</div>
              <div><strong>Depth:</strong> {earthquake.depth?.toFixed(1) || '?'} km</div>
              <div><strong>Lat:</strong> {earthquake.lat?.toFixed(2)}°</div>
              <div><strong>Lon:</strong> {earthquake.lon?.toFixed(2)}°</div>
            </div>
          </div>
        </Popup>
      </CircleMarker>
    </>
  );
}

// Timeline component
function Timeline({
  earthquakes,
  currentTime,
  startTime,
  endTime,
  onTimeChange,
  isPlaying,
  onPlayPause,
  playbackSpeed,
  onSpeedChange
}) {
  const timelineRef = useRef(null);

  const totalDuration = endTime - startTime;
  const progress = ((currentTime - startTime) / totalDuration) * 100;

  // Group earthquakes by hour for the histogram
  const hourlyData = useMemo(() => {
    const hours = {};
    const hourMs = 60 * 60 * 1000;

    for (let t = startTime; t <= endTime; t += hourMs) {
      hours[t] = { count: 0, maxMag: 0 };
    }

    earthquakes.forEach(eq => {
      const eqTime = new Date(eq.time).getTime();
      const hourKey = Math.floor(eqTime / hourMs) * hourMs;
      if (hours[hourKey]) {
        hours[hourKey].count++;
        hours[hourKey].maxMag = Math.max(hours[hourKey].maxMag, eq.mag || 0);
      }
    });

    return Object.entries(hours).map(([time, data]) => ({
      time: parseInt(time),
      ...data
    }));
  }, [earthquakes, startTime, endTime]);

  const maxCount = Math.max(...hourlyData.map(h => h.count), 1);

  const handleTimelineClick = (e) => {
    if (!timelineRef.current) return;
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const newTime = startTime + (totalDuration * percent);
    onTimeChange(Math.max(startTime, Math.min(endTime, newTime)));
  };

  return (
    <div className="bg-zinc-900/95 backdrop-blur border-t border-zinc-700 p-3">
      {/* Playback Controls - Compact */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          {/* Play/Pause Button - Enhanced */}
          <button
            onClick={onPlayPause}
            className={`relative p-3 rounded-full transition-all transform hover:scale-110 ${
              isPlaying
                ? 'bg-gradient-to-br from-orange-500 to-red-500 text-white shadow-lg shadow-orange-500/50'
                : 'bg-gradient-to-br from-zinc-700 to-zinc-800 hover:from-orange-600 hover:to-orange-700 text-white border border-zinc-600 hover:border-orange-500'
            }`}
          >
            {isPlaying && (
              <div className="absolute inset-0 rounded-full bg-orange-500 animate-ping opacity-30" />
            )}
            {isPlaying ? (
              <svg className="w-5 h-5 relative z-10" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
              </svg>
            ) : (
              <svg className="w-5 h-5 relative z-10" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z"/>
              </svg>
            )}
          </button>

          {/* Speed Control - Enhanced */}
          <div className="flex bg-zinc-800 rounded-lg overflow-hidden border border-zinc-700">
            {[1, 2, 5, 10].map(speed => (
              <button
                key={speed}
                onClick={() => onSpeedChange(speed)}
                className={`px-3 py-1.5 text-xs font-bold transition-all ${
                  playbackSpeed === speed
                    ? 'bg-gradient-to-b from-orange-500 to-orange-600 text-white shadow-inner'
                    : 'text-zinc-400 hover:text-white hover:bg-zinc-700'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>

          {/* Playback Status */}
          {isPlaying && (
            <div className="flex items-center gap-1.5 px-2 py-1 bg-orange-500/10 rounded-full border border-orange-500/30">
              <div className="flex gap-0.5">
                <div className="w-1 h-3 bg-orange-500 rounded animate-pulse" style={{ animationDelay: '0ms' }} />
                <div className="w-1 h-4 bg-orange-400 rounded animate-pulse" style={{ animationDelay: '150ms' }} />
                <div className="w-1 h-2 bg-orange-500 rounded animate-pulse" style={{ animationDelay: '300ms' }} />
              </div>
              <span className="text-orange-400 text-[10px] font-medium">Playing</span>
            </div>
          )}
        </div>

        {/* Current Time Display */}
        <div className="text-center">
          <div className="text-lg font-bold text-white font-mono">
            {formatDateTime(new Date(currentTime))}
          </div>
          <div className="text-zinc-500 text-[10px]">
            {formatTimeAgo(new Date(currentTime).toISOString())}
          </div>
        </div>

        {/* Time Range */}
        <div className="text-right text-[10px]">
          <div className="text-zinc-400">
            <span className="text-green-400">{formatTime(new Date(startTime))}</span>
            <span className="text-zinc-600 mx-1">-</span>
            <span className="text-red-400">{formatTime(new Date(endTime))}</span>
          </div>
          <div className="text-zinc-600">24h</div>
        </div>
      </div>

      {/* Timeline Bar with Histogram - Compact */}
      <div
        ref={timelineRef}
        className="relative h-12 bg-zinc-800 rounded-lg cursor-pointer overflow-hidden group"
        onClick={handleTimelineClick}
      >
        {/* Histogram bars */}
        <div className="absolute inset-0 flex items-end px-1">
          {hourlyData.map((hour, i) => (
            <div
              key={hour.time}
              className="flex-1 mx-px transition-all duration-200"
              style={{
                height: `${(hour.count / maxCount) * 100}%`,
                backgroundColor: hour.maxMag >= 5 ? getMagColor(hour.maxMag) : '#3f3f46',
                opacity: hour.time <= currentTime ? 1 : 0.3,
              }}
              title={`${hour.count} earthquakes`}
            />
          ))}
        </div>

        {/* Progress overlay */}
        <div
          className="absolute inset-y-0 left-0 bg-orange-500/20 pointer-events-none transition-all"
          style={{ width: `${progress}%` }}
        />

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-orange-500 shadow-lg shadow-orange-500/50 transition-all"
          style={{ left: `${progress}%`, transform: 'translateX(-50%)' }}
        >
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-orange-500 rounded-full border-2 border-white" />
          <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-3 h-3 bg-orange-500 rounded-full border-2 border-white" />
        </div>

        {/* Hour markers */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between px-2 pb-1">
          {[0, 6, 12, 18, 24].map(h => (
            <span key={h} className="text-[10px] text-zinc-600 font-mono">
              {h === 0 ? '-24h' : h === 24 ? 'now' : `-${24-h}h`}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// Earthquake event feed - simple list
function EventFeed({ earthquakes, currentTime }) {
  const visibleEarthquakes = useMemo(() => {
    return earthquakes
      .filter(eq => new Date(eq.time).getTime() <= currentTime)
      .slice(0, 100);
  }, [earthquakes, currentTime]);

  return (
    <div>
      <div className="px-3 py-2 border-b border-zinc-700 bg-zinc-800 flex items-center justify-between sticky top-0">
        <span className="text-zinc-400 text-[10px] uppercase">Event Feed</span>
        <span className="text-orange-500 text-xs font-bold">{visibleEarthquakes.length}</span>
      </div>
      {visibleEarthquakes.length === 0 ? (
        <div className="p-4 text-zinc-500 text-sm text-center">
          <div className="animate-pulse">Waiting...</div>
        </div>
      ) : (
        <div className="divide-y divide-zinc-700/50">
          {visibleEarthquakes.map((eq, index) => {
            const isVeryRecent = index < 3;
            return (
              <div
                key={eq.id || index}
                className={`p-2 transition-all duration-500 ${
                  isVeryRecent
                    ? 'bg-gradient-to-r from-orange-500/10 to-transparent border-l-2 border-orange-500'
                    : 'hover:bg-zinc-700/30'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`w-10 text-center px-1 py-0.5 rounded text-white font-bold text-[10px] ${
                      isVeryRecent ? 'animate-pulse' : ''
                    }`}
                    style={{ backgroundColor: getMagColor(eq.mag) }}
                  >
                    M{eq.mag?.toFixed(1)}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-white text-[11px] truncate">{eq.place || 'Unknown'}</p>
                    <p className="text-zinc-500 text-[9px] font-mono">
                      {formatDateTime(new Date(eq.time))}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Stats display component - compact version
function StatsDisplay({ earthquakes, currentTime }) {
  const stats = useMemo(() => {
    const visible = earthquakes.filter(eq => new Date(eq.time).getTime() <= currentTime);
    const counts = { total: visible.length, m5plus: 0, m4plus: 0, m3plus: 0 };
    let strongest = null;

    visible.forEach(eq => {
      const mag = eq.mag || 0;
      if (mag >= 5) counts.m5plus++;
      if (mag >= 4) counts.m4plus++;
      if (mag >= 3) counts.m3plus++;
      if (!strongest || mag > strongest.mag) {
        strongest = eq;
      }
    });

    return { counts, strongest };
  }, [earthquakes, currentTime]);

  return (
    <div className="p-3 border-b border-zinc-700">
      {/* Total + Magnitude breakdown in one row */}
      <div className="flex items-center gap-2 mb-2">
        <div className="text-center flex-shrink-0">
          <div className="text-2xl font-bold text-orange-500 font-mono">
            {stats.counts.total}
          </div>
          <div className="text-zinc-500 text-[9px] uppercase">Events</div>
        </div>
        <div className="flex-1 grid grid-cols-3 gap-1">
          <div className="bg-zinc-800/50 rounded p-1 text-center">
            <div className="text-sm font-bold text-red-500">{stats.counts.m5plus}</div>
            <div className="text-[8px] text-zinc-500">M5+</div>
          </div>
          <div className="bg-zinc-800/50 rounded p-1 text-center">
            <div className="text-sm font-bold text-orange-500">{stats.counts.m4plus}</div>
            <div className="text-[8px] text-zinc-500">M4+</div>
          </div>
          <div className="bg-zinc-800/50 rounded p-1 text-center">
            <div className="text-sm font-bold text-yellow-500">{stats.counts.m3plus}</div>
            <div className="text-[8px] text-zinc-500">M3+</div>
          </div>
        </div>
      </div>

      {/* Strongest earthquake - compact */}
      {stats.strongest && (
        <div className="bg-gradient-to-r from-red-900/20 to-transparent rounded p-2 border-l-2 border-red-500">
          <div className="flex items-center gap-2">
            <span
              className="px-1.5 py-0.5 rounded text-white font-bold text-xs"
              style={{ backgroundColor: getMagColor(stats.strongest.mag) }}
            >
              M{stats.strongest.mag?.toFixed(1)}
            </span>
            <p className="text-white text-xs truncate flex-1">{stats.strongest.place}</p>
          </div>
        </div>
      )}

      {/* Compact color legend */}
      <div className="flex flex-wrap gap-1 mt-2">
        {[
          { label: '7+', color: '#9333ea' },
          { label: '6+', color: '#dc2626' },
          { label: '5+', color: '#f97316' },
          { label: '4+', color: '#eab308' },
          { label: '3+', color: '#22c55e' },
          { label: '<3', color: '#06b6d4' },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-1 text-[9px] text-zinc-400">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function RealtimeMap() {
  const [earthquakes, setEarthquakes] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [minMag, setMinMag] = useState(2);

  // Time-lapse state
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const [currentTime, setCurrentTime] = useState(Date.now());
  const [timeRange, setTimeRange] = useState({ start: Date.now() - 24*60*60*1000, end: Date.now() });
  const [newEqIds, setNewEqIds] = useState(new Set());

  const playIntervalRef = useRef(null);

  // Fetch earthquakes
  const fetchEarthquakes = useCallback(async () => {
    try {
      const response = await fetch(`/api/earthquakes-24h?min_mag=${minMag}`);
      if (!response.ok) throw new Error('Failed to fetch earthquakes');
      const data = await response.json();

      if (data.success) {
        // Sort by time ascending for playback
        const sorted = [...data.earthquakes].sort((a, b) =>
          new Date(a.time) - new Date(b.time)
        );

        // Track new earthquakes for animation
        const currentIds = new Set(earthquakes.map(eq => eq.id));
        const newIds = new Set();
        sorted.forEach(eq => {
          if (!currentIds.has(eq.id)) {
            newIds.add(eq.id);
          }
        });
        setNewEqIds(newIds);

        if (newIds.size > 0) {
          setTimeout(() => setNewEqIds(new Set()), 3000);
        }

        setEarthquakes(sorted);

        // Update time range
        const now = Date.now();
        const start = now - 24 * 60 * 60 * 1000;
        setTimeRange({ start, end: now });

        // Always set currentTime to now when data loads (unless actively playing)
        // Use ref to check playing state to avoid stale closure
        if (!playIntervalRef.current) {
          setCurrentTime(now);
        }

        setError(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, [minMag]);

  // Initial fetch and polling
  useEffect(() => {
    fetchEarthquakes();
    const interval = setInterval(fetchEarthquakes, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [minMag]);

  // Playback logic
  useEffect(() => {
    if (isPlaying) {
      // Calculate time increment per frame (aim for 60fps feel)
      // At 1x speed, 24 hours = 60 seconds of playback
      // So each second represents 24 minutes of real time
      const msPerFrame = 50; // 20fps
      const realTimePerFrame = (24 * 60 * 60 * 1000) / (60 * 1000 / msPerFrame) * playbackSpeed;

      playIntervalRef.current = setInterval(() => {
        setCurrentTime(prev => {
          const next = prev + realTimePerFrame;
          if (next >= timeRange.end) {
            setIsPlaying(false);
            return timeRange.end;
          }
          return next;
        });
      }, msPerFrame);
    } else {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    }

    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, timeRange.end]);

  // Handle play/pause
  const handlePlayPause = () => {
    if (!isPlaying) {
      // If at end, restart from beginning
      if (currentTime >= timeRange.end - 1000) {
        setCurrentTime(timeRange.start);
      }
    }
    setIsPlaying(!isPlaying);
  };

  // Filter earthquakes by current time
  const visibleEarthquakes = useMemo(() => {
    return earthquakes.filter(eq => {
      const eqTime = new Date(eq.time).getTime();
      return eqTime <= currentTime &&
             eq.lat !== null && eq.lat !== undefined &&
             eq.lon !== null && eq.lon !== undefined;
    });
  }, [earthquakes, currentTime]);

  // Calculate opacity for each earthquake based on age
  // Fast fade: 0-30min = full bright, 30-60min = fade out completely
  const fadeStartMs = 30 * 60 * 1000;  // Start fading after 30 minutes
  const fadeEndMs = 60 * 60 * 1000;    // Fully faded by 60 minutes
  const recentThreshold = 10 * 60 * 1000; // 10 minutes - "brand new" with special effects

  const earthquakeOpacities = useMemo(() => {
    const opacities = new Map();
    earthquakes.forEach(eq => {
      const eqTime = new Date(eq.time).getTime();
      if (eqTime <= currentTime) {
        const age = currentTime - eqTime;

        if (age < fadeStartMs) {
          // 0-30 min: Full opacity
          opacities.set(eq.id, 1.0);
        } else if (age < fadeEndMs) {
          // 30-60 min: Fade from 1.0 to 0.1
          const fadeProgress = (age - fadeStartMs) / (fadeEndMs - fadeStartMs);
          opacities.set(eq.id, Math.max(0.1, 1.0 - (fadeProgress * 0.9)));
        } else {
          // After 60 min: Nearly invisible
          opacities.set(eq.id, 0.1);
        }
      }
    });
    return opacities;
  }, [earthquakes, currentTime]);

  const recentEqIds = useMemo(() => {
    const recent = new Set();
    earthquakes.forEach(eq => {
      const eqTime = new Date(eq.time).getTime();
      if (eqTime <= currentTime && currentTime - eqTime < recentThreshold) {
        recent.add(eq.id);
      }
    });
    return recent;
  }, [earthquakes, currentTime]);

  return (
    <div className="h-screen bg-zinc-900 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 bg-gradient-to-r from-zinc-900 to-zinc-800 border-b-2 border-orange-500 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/" className="text-zinc-400 hover:text-white transition-colors">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </Link>
              <div>
                <h1 className="text-xl font-bold text-orange-500 flex items-center gap-2">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Earthquake Time Machine
                </h1>
                <p className="text-zinc-500 text-xs">Watch 24 hours of seismic activity unfold</p>
              </div>
            </div>

            {/* Live/Playback indicator */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${
              isPlaying ? 'bg-orange-500/20' : 'bg-green-500/20'
            }`}>
              <div className={`w-2.5 h-2.5 rounded-full ${
                isPlaying ? 'bg-orange-500 animate-pulse' : 'bg-green-500 animate-pulse'
              }`} />
              <span className={`text-xs font-medium ${
                isPlaying ? 'text-orange-400' : 'text-green-400'
              }`}>
                {isPlaying ? 'PLAYBACK' : 'LIVE'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Sidebar */}
        <div className="lg:w-72 bg-zinc-800/50 border-b lg:border-b-0 lg:border-r border-zinc-700 flex flex-col overflow-hidden">
          {/* Fixed Header: Filter + Stats */}
          <div className="flex-shrink-0">
            {/* Magnitude Filter */}
            <div className="p-2 border-b border-zinc-700 bg-zinc-800">
              <div className="flex items-center justify-between">
                <span className="text-zinc-400 text-xs font-medium">Filter</span>
                <select
                  value={minMag}
                  onChange={(e) => setMinMag(Number(e.target.value))}
                  className="bg-zinc-700 text-white text-sm px-2 py-1 rounded border border-zinc-600 focus:outline-none focus:border-orange-500"
                >
                  <option value={0}>All</option>
                  <option value={2}>M2+</option>
                  <option value={3}>M3+</option>
                  <option value={4}>M4+</option>
                  <option value={5}>M5+</option>
                  <option value={6}>M6+</option>
                </select>
              </div>
            </div>

            {/* Compact Stats */}
            <StatsDisplay earthquakes={earthquakes} currentTime={currentTime} />
          </div>

          {/* Scrollable Event Feed */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            <EventFeed earthquakes={earthquakes} currentTime={currentTime} />
          </div>
        </div>

        {/* Map Container */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Map */}
          <div className="flex-1 relative min-h-0">
            {isLoading && (
              <div className="absolute inset-0 bg-zinc-900/80 flex items-center justify-center z-10">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto mb-4"></div>
                  <p className="text-zinc-400">Loading earthquake data...</p>
                </div>
              </div>
            )}

            <MapContainer
              center={[20, 0]}
              zoom={2}
              style={{ height: '100%', width: '100%' }}
              className="z-0"
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              />

              {/* Render earthquake markers */}
              {visibleEarthquakes.map((eq, index) => (
                <AnimatedEarthquakeMarker
                  key={eq.id || index}
                  earthquake={eq}
                  isNew={newEqIds.has(eq.id)}
                  isRecent={recentEqIds.has(eq.id)}
                  opacity={earthquakeOpacities.get(eq.id) || 0.5}
                />
              ))}
            </MapContainer>

            {/* Map overlay stats */}
            <div className="absolute top-4 right-4 bg-zinc-900/90 backdrop-blur rounded-lg p-3 border border-zinc-700 z-[1000]">
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-500 font-mono">
                  {visibleEarthquakes.length}
                </div>
                <div className="text-zinc-500 text-[10px] uppercase">Visible Events</div>
              </div>
            </div>

            {/* Visual Legend - Explains the animation */}
            <div className="absolute bottom-28 right-4 bg-zinc-900/95 backdrop-blur rounded-lg p-3 border border-zinc-700 z-[1000] max-w-[180px]">
              <div className="text-zinc-400 text-[10px] uppercase mb-2 font-bold">How to Read</div>

              {/* Time fade explanation */}
              <div className="space-y-2">
                {/* Fresh earthquake */}
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <div className="w-4 h-4 rounded-full bg-orange-500 animate-pulse" />
                    <div className="absolute inset-0 w-4 h-4 rounded-full bg-orange-500/30 animate-ping" />
                  </div>
                  <div>
                    <div className="text-white text-[11px] font-medium">Pulsing + Rings</div>
                    <div className="text-zinc-500 text-[9px]">Just happened (&lt;10 min)</div>
                  </div>
                </div>

                {/* Recent earthquake */}
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-orange-500 opacity-100" />
                  <div>
                    <div className="text-white text-[11px] font-medium">Bright & Full</div>
                    <div className="text-zinc-500 text-[9px]">Recent (10-30 min)</div>
                  </div>
                </div>

                {/* Fading earthquake */}
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-500 opacity-50" />
                  <div>
                    <div className="text-white text-[11px] font-medium">Fading Away</div>
                    <div className="text-zinc-500 text-[9px]">Aging (30-60 min)</div>
                  </div>
                </div>

                {/* Old earthquake */}
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-orange-500 opacity-20" />
                  <div>
                    <div className="text-white text-[11px] font-medium">Ghost</div>
                    <div className="text-zinc-500 text-[9px]">Old (&gt;60 min)</div>
                  </div>
                </div>
              </div>

              {/* Divider */}
              <div className="border-t border-zinc-700 my-2" />

              {/* Tip */}
              <div className="text-zinc-500 text-[9px] italic">
                Watch markers fade as time passes during playback
              </div>
            </div>

            {/* Instructions overlay - always show when not playing */}
            {!isPlaying && (
              <div className="absolute bottom-24 left-1/2 -translate-x-1/2 bg-gradient-to-r from-zinc-900/95 via-zinc-800/95 to-zinc-900/95 backdrop-blur rounded-xl px-6 py-3 border border-orange-500/50 z-[1000] shadow-lg shadow-orange-500/20">
                <div className="flex items-center gap-3 text-orange-400">
                  <div className="relative">
                    <svg className="w-8 h-8 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z"/>
                    </svg>
                    <div className="absolute inset-0 w-8 h-8 animate-ping">
                      <svg className="w-8 h-8 opacity-30" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z"/>
                      </svg>
                    </div>
                  </div>
                  <div>
                    <div className="text-sm font-bold">
                      {currentTime >= timeRange.end - 1000 ? 'Replay Time-Lapse' : 'Continue Playback'}
                    </div>
                    <div className="text-[10px] text-orange-300/70">
                      Watch 24 hours unfold - earthquakes appear, pulse, then fade
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Timeline - fixed at bottom */}
          <div className="flex-shrink-0">
            <Timeline
              earthquakes={earthquakes}
              currentTime={currentTime}
              startTime={timeRange.start}
              endTime={timeRange.end}
              onTimeChange={setCurrentTime}
              isPlaying={isPlaying}
              onPlayPause={handlePlayPause}
              playbackSpeed={playbackSpeed}
              onSpeedChange={setPlaybackSpeed}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
