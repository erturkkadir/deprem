import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// ============================================
// EARTHQUAKE SOUND ENGINE (Web Audio API)
// Spooky, dramatic sounds with reverb/echo
// ============================================
class EarthquakeSoundEngine {
  constructor() {
    this.audioContext = null;
    this.masterGain = null;
    this.convolver = null; // Reverb
    this.reverbGain = null;
    this.dryGain = null;
    this.isEnabled = false;
    this.volume = 0.5;
  }

  init() {
    if (this.audioContext) return;
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

      // Master output
      this.masterGain = this.audioContext.createGain();
      this.masterGain.gain.value = this.volume;
      this.masterGain.connect(this.audioContext.destination);

      // Create reverb (convolver) for spooky echo
      this.convolver = this.audioContext.createConvolver();
      this.convolver.buffer = this.createReverbImpulse(3.5, 4.0); // Long, dark reverb

      // Dry/wet mix for reverb
      this.dryGain = this.audioContext.createGain();
      this.reverbGain = this.audioContext.createGain();
      this.dryGain.gain.value = 0.4; // Less dry
      this.reverbGain.gain.value = 0.8; // More reverb for spooky effect

      this.dryGain.connect(this.masterGain);
      this.convolver.connect(this.reverbGain);
      this.reverbGain.connect(this.masterGain);

      this.isEnabled = true;
    } catch (e) {
      console.warn('Web Audio API not supported:', e);
      this.isEnabled = false;
    }
  }

  // Create impulse response for reverb (simulates large cave/hall)
  createReverbImpulse(duration, decay) {
    const sampleRate = this.audioContext.sampleRate;
    const length = sampleRate * duration;
    const impulse = this.audioContext.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
      const channelData = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        // Exponential decay with random noise
        const envelope = Math.pow(1 - i / length, decay);
        // Add some randomness for natural reverb
        channelData[i] = (Math.random() * 2 - 1) * envelope;
      }
    }
    return impulse;
  }

  setVolume(vol) {
    this.volume = Math.max(0, Math.min(1, vol));
    if (this.masterGain) {
      this.masterGain.gain.value = this.volume;
    }
  }

  setEnabled(enabled) {
    this.isEnabled = enabled;
    if (enabled && !this.audioContext) {
      this.init();
    }
  }

  // Play earthquake sound - BLAST AND CRACK style
  playEarthquakeSound(magnitude) {
    if (!this.isEnabled || !this.audioContext) return;

    // Resume audio context if suspended (browser autoplay policy)
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    const now = this.audioContext.currentTime;

    // Scale parameters by magnitude (2-8 range)
    const magNormalized = Math.max(0, Math.min(1, (magnitude - 2) / 6));

    // Duration: longer for bigger quakes (0.8s for M2, 2.5s for M8)
    const duration = 0.8 + (magNormalized * 1.7);

    // Volume: scale dramatically by magnitude
    // M2: 0.15, M4: 0.4, M6: 0.7, M8: 1.0
    const volume = 0.15 + (magNormalized * 0.85);

    // === EXPLOSIVE BLAST (main impact) ===
    // Sharp attack with rapid frequency sweep down - like an explosion
    const blastOsc = this.audioContext.createOscillator();
    const blastGain = this.audioContext.createGain();
    blastOsc.type = 'sawtooth';
    // Start very high, sweep down rapidly
    blastOsc.frequency.setValueAtTime(800 + magNormalized * 400, now);
    blastOsc.frequency.exponentialRampToValueAtTime(60, now + 0.08);
    blastOsc.frequency.exponentialRampToValueAtTime(30, now + 0.3);

    blastGain.gain.setValueAtTime(0, now);
    blastGain.gain.linearRampToValueAtTime(volume * 0.9, now + 0.002); // Instant attack
    blastGain.gain.exponentialRampToValueAtTime(volume * 0.4, now + 0.03);
    blastGain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);

    const blastFilter = this.audioContext.createBiquadFilter();
    blastFilter.type = 'lowpass';
    blastFilter.frequency.setValueAtTime(3000, now);
    blastFilter.frequency.exponentialRampToValueAtTime(200, now + 0.2);

    blastOsc.connect(blastFilter);
    blastFilter.connect(blastGain);
    blastGain.connect(this.dryGain);
    blastGain.connect(this.convolver);

    // === CRACK TRANSIENT (sharp high-frequency snap) ===
    const crackLength = 0.08;
    const crackBuffer = this.audioContext.createBuffer(2, this.audioContext.sampleRate * crackLength, this.audioContext.sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const crackData = crackBuffer.getChannelData(channel);
      for (let i = 0; i < crackData.length; i++) {
        const t = i / crackData.length;
        // Very sharp attack, instant decay - like a whip crack
        const env = Math.pow(1 - t, 8);
        crackData[i] = (Math.random() * 2 - 1) * env;
      }
    }

    const crackSource = this.audioContext.createBufferSource();
    crackSource.buffer = crackBuffer;

    const crackFilter = this.audioContext.createBiquadFilter();
    crackFilter.type = 'highpass';
    crackFilter.frequency.setValueAtTime(2000, now);
    crackFilter.Q.setValueAtTime(1, now);

    const crackGain = this.audioContext.createGain();
    crackGain.gain.setValueAtTime(volume * 0.7, now);
    crackGain.gain.exponentialRampToValueAtTime(0.001, now + crackLength);

    crackSource.connect(crackFilter);
    crackFilter.connect(crackGain);
    crackGain.connect(this.dryGain);

    // === SECONDARY CRACK (slightly delayed for layering) ===
    const crack2Length = 0.06;
    const crack2Buffer = this.audioContext.createBuffer(2, this.audioContext.sampleRate * crack2Length, this.audioContext.sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const crack2Data = crack2Buffer.getChannelData(channel);
      for (let i = 0; i < crack2Data.length; i++) {
        const t = i / crack2Data.length;
        const env = Math.pow(1 - t, 10);
        crack2Data[i] = (Math.random() * 2 - 1) * env;
      }
    }

    const crack2Source = this.audioContext.createBufferSource();
    crack2Source.buffer = crack2Buffer;

    const crack2Filter = this.audioContext.createBiquadFilter();
    crack2Filter.type = 'bandpass';
    crack2Filter.frequency.setValueAtTime(4000, now);
    crack2Filter.Q.setValueAtTime(2, now);

    const crack2Gain = this.audioContext.createGain();
    crack2Gain.gain.setValueAtTime(volume * 0.5, now);

    crack2Source.connect(crack2Filter);
    crack2Filter.connect(crack2Gain);
    crack2Gain.connect(this.dryGain);

    // === EXPLOSION BODY (mid-frequency punch) ===
    const bodyOsc = this.audioContext.createOscillator();
    const bodyGain = this.audioContext.createGain();
    bodyOsc.type = 'square';
    bodyOsc.frequency.setValueAtTime(120, now);
    bodyOsc.frequency.exponentialRampToValueAtTime(40, now + 0.15);

    bodyGain.gain.setValueAtTime(0, now);
    bodyGain.gain.linearRampToValueAtTime(volume * 0.6, now + 0.005);
    bodyGain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);

    const bodyFilter = this.audioContext.createBiquadFilter();
    bodyFilter.type = 'lowpass';
    bodyFilter.frequency.setValueAtTime(500, now);

    bodyOsc.connect(bodyFilter);
    bodyFilter.connect(bodyGain);
    bodyGain.connect(this.dryGain);
    bodyGain.connect(this.convolver);

    // === DEBRIS/RUBBLE NOISE (the aftermath) ===
    const debrisLength = 0.3 + magNormalized * 0.4;
    const debrisBuffer = this.audioContext.createBuffer(2, this.audioContext.sampleRate * debrisLength, this.audioContext.sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const debrisData = debrisBuffer.getChannelData(channel);
      for (let i = 0; i < debrisData.length; i++) {
        const t = i / debrisData.length;
        // Irregular decay like falling debris
        const env = Math.pow(1 - t, 1.5) * (0.7 + Math.random() * 0.3);
        debrisData[i] = (Math.random() * 2 - 1) * env;
      }
    }

    const debrisSource = this.audioContext.createBufferSource();
    debrisSource.buffer = debrisBuffer;

    const debrisFilter = this.audioContext.createBiquadFilter();
    debrisFilter.type = 'bandpass';
    debrisFilter.frequency.setValueAtTime(800, now);
    debrisFilter.frequency.exponentialRampToValueAtTime(300, now + debrisLength);
    debrisFilter.Q.setValueAtTime(0.5, now);

    const debrisGain = this.audioContext.createGain();
    debrisGain.gain.setValueAtTime(volume * 0.4, now + 0.05);
    debrisGain.gain.exponentialRampToValueAtTime(0.001, now + debrisLength);

    debrisSource.connect(debrisFilter);
    debrisFilter.connect(debrisGain);
    debrisGain.connect(this.dryGain);
    debrisGain.connect(this.convolver);

    // === LOW-END THUMP (subsonic punch) ===
    const thumpOsc = this.audioContext.createOscillator();
    const thumpGain = this.audioContext.createGain();
    thumpOsc.type = 'sine';
    thumpOsc.frequency.setValueAtTime(60, now);
    thumpOsc.frequency.exponentialRampToValueAtTime(25, now + 0.2);

    thumpGain.gain.setValueAtTime(0, now);
    thumpGain.gain.linearRampToValueAtTime(volume * 0.8, now + 0.01);
    thumpGain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);

    thumpOsc.connect(thumpGain);
    thumpGain.connect(this.dryGain);

    // === SECONDARY BLASTS for big quakes (M5+) ===
    if (magnitude >= 5) {
      const delays = [0.15, 0.35];
      delays.forEach((delay, i) => {
        const echoBlast = this.audioContext.createOscillator();
        const echoGain = this.audioContext.createGain();
        echoBlast.type = 'sawtooth';
        echoBlast.frequency.setValueAtTime(400 - i * 100, now + delay);
        echoBlast.frequency.exponentialRampToValueAtTime(40, now + delay + 0.1);

        echoGain.gain.setValueAtTime(0, now);
        echoGain.gain.linearRampToValueAtTime(volume * (0.4 - i * 0.15), now + delay);
        echoGain.gain.exponentialRampToValueAtTime(0.001, now + delay + 0.25);

        echoBlast.connect(echoGain);
        echoGain.connect(this.convolver);

        echoBlast.start(now + delay);
        echoBlast.stop(now + delay + 0.4);
      });
    }

    // === CRUMBLING TAIL (extended decay for bigger quakes) ===
    if (magnitude >= 4) {
      const tailLength = 0.5 + magNormalized * 0.8;
      const tailBuffer = this.audioContext.createBuffer(2, this.audioContext.sampleRate * tailLength, this.audioContext.sampleRate);
      for (let channel = 0; channel < 2; channel++) {
        const tailData = tailBuffer.getChannelData(channel);
        for (let i = 0; i < tailData.length; i++) {
          const t = i / tailData.length;
          const env = Math.pow(1 - t, 2);
          tailData[i] = (Math.random() * 2 - 1) * env * 0.5;
        }
      }

      const tailSource = this.audioContext.createBufferSource();
      tailSource.buffer = tailBuffer;

      const tailFilter = this.audioContext.createBiquadFilter();
      tailFilter.type = 'lowpass';
      tailFilter.frequency.setValueAtTime(400, now);

      const tailGain = this.audioContext.createGain();
      tailGain.gain.setValueAtTime(volume * 0.3, now + 0.2);
      tailGain.gain.exponentialRampToValueAtTime(0.001, now + 0.2 + tailLength);

      tailSource.connect(tailFilter);
      tailFilter.connect(tailGain);
      tailGain.connect(this.convolver);

      tailSource.start(now + 0.15);
      tailSource.stop(now + 0.2 + tailLength);
    }

    // Start main sounds
    blastOsc.start(now);
    blastOsc.stop(now + 0.5);
    crackSource.start(now);
    crackSource.stop(now + crackLength + 0.05);
    crack2Source.start(now + 0.02);
    crack2Source.stop(now + 0.02 + crack2Length + 0.05);
    bodyOsc.start(now);
    bodyOsc.stop(now + 0.4);
    debrisSource.start(now + 0.03);
    debrisSource.stop(now + 0.03 + debrisLength + 0.1);
    thumpOsc.start(now);
    thumpOsc.stop(now + 0.6);

    // Cleanup
    setTimeout(() => {
      [blastOsc, bodyOsc, thumpOsc].forEach(o => { try { o.disconnect(); } catch(e) {} });
      [blastGain, crackGain, crack2Gain, bodyGain, debrisGain, thumpGain].forEach(g => { try { g.disconnect(); } catch(e) {} });
      try { crackSource.disconnect(); crack2Source.disconnect(); debrisSource.disconnect(); } catch(e) {}
    }, (duration + 3) * 1000);
  }

  // Create distortion curve for waveshaper (kept for potential future use)
  makeDistortionCurve(amount) {
    const samples = 44100;
    const curve = new Float32Array(samples);
    const deg = Math.PI / 180;
    for (let i = 0; i < samples; i++) {
      const x = (i * 2) / samples - 1;
      curve[i] = ((3 + amount) * x * 20 * deg) / (Math.PI + amount * Math.abs(x));
    }
    return curve;
  }

  // Ambient tension drone - SPOOKIER version
  startAmbientDrone() {
    if (!this.isEnabled || !this.audioContext || this.ambientOsc) return;

    const now = this.audioContext.currentTime;

    // Deep, ominous drone
    this.ambientOsc = this.audioContext.createOscillator();
    this.ambientGain = this.audioContext.createGain();
    this.ambientOsc.type = 'sine';
    this.ambientOsc.frequency.setValueAtTime(30, now); // Very low

    // Second oscillator for thickness
    this.ambientOsc2 = this.audioContext.createOscillator();
    this.ambientGain2 = this.audioContext.createGain();
    this.ambientOsc2.type = 'triangle';
    this.ambientOsc2.frequency.setValueAtTime(60, now); // Octave up

    // LFO for unsettling movement
    this.ambientLfo = this.audioContext.createOscillator();
    this.ambientLfoGain = this.audioContext.createGain();
    this.ambientLfo.frequency.setValueAtTime(0.05, now); // Very slow
    this.ambientLfoGain.gain.setValueAtTime(8, now);
    this.ambientLfo.connect(this.ambientLfoGain);
    this.ambientLfoGain.connect(this.ambientOsc.frequency);

    // Second LFO for the higher oscillator (different rate for complexity)
    this.ambientLfo2 = this.audioContext.createOscillator();
    this.ambientLfoGain2 = this.audioContext.createGain();
    this.ambientLfo2.frequency.setValueAtTime(0.08, now);
    this.ambientLfoGain2.gain.setValueAtTime(4, now);
    this.ambientLfo2.connect(this.ambientLfoGain2);
    this.ambientLfoGain2.connect(this.ambientOsc2.frequency);

    this.ambientGain.gain.setValueAtTime(0, now);
    this.ambientGain.gain.linearRampToValueAtTime(0.15, now + 3);
    this.ambientGain2.gain.setValueAtTime(0, now);
    this.ambientGain2.gain.linearRampToValueAtTime(0.08, now + 3);

    this.ambientOsc.connect(this.ambientGain);
    this.ambientOsc2.connect(this.ambientGain2);
    this.ambientGain.connect(this.masterGain);
    this.ambientGain2.connect(this.convolver); // Through reverb for depth

    this.ambientOsc.start(now);
    this.ambientOsc2.start(now);
    this.ambientLfo.start(now);
    this.ambientLfo2.start(now);
  }

  stopAmbientDrone() {
    if (!this.ambientOsc) return;

    const now = this.audioContext.currentTime;
    this.ambientGain.gain.linearRampToValueAtTime(0, now + 2);
    this.ambientGain2.gain.linearRampToValueAtTime(0, now + 2);

    setTimeout(() => {
      if (this.ambientOsc) {
        try {
          this.ambientOsc.stop();
          this.ambientOsc.disconnect();
          this.ambientOsc2.stop();
          this.ambientOsc2.disconnect();
          this.ambientLfo.stop();
          this.ambientLfo.disconnect();
          this.ambientLfo2.stop();
          this.ambientLfo2.disconnect();
          this.ambientLfoGain.disconnect();
          this.ambientLfoGain2.disconnect();
          this.ambientGain.disconnect();
          this.ambientGain2.disconnect();
        } catch(e) {}
        this.ambientOsc = null;
      }
    }, 2500);
  }

  // Tick sound - subtle ghostly click
  playTick() {
    if (!this.isEnabled || !this.audioContext) return;

    const now = this.audioContext.currentTime;
    const osc = this.audioContext.createOscillator();
    const gain = this.audioContext.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(600, now);
    osc.frequency.exponentialRampToValueAtTime(200, now + 0.08);

    gain.gain.setValueAtTime(0.08, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);

    osc.connect(gain);
    gain.connect(this.convolver); // Through reverb

    osc.start(now);
    osc.stop(now + 0.1);
  }
}

// Global sound engine instance
const soundEngine = new EarthquakeSoundEngine();

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
    <div className="bg-zinc-900 border-t border-zinc-700 px-2 sm:px-3 pt-2 sm:pt-3 pb-[max(0.5rem,env(safe-area-inset-bottom))]">
      {/* Playback Controls */}
      <div className="flex items-center justify-between mb-2 gap-2">
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Play/Pause Button */}
          <button
            onClick={onPlayPause}
            className={`relative p-2 sm:p-2.5 rounded-full transition-all transform hover:scale-105 ${
              isPlaying
                ? 'bg-gradient-to-br from-orange-500 to-red-500 text-white shadow-md shadow-orange-500/50'
                : 'bg-gradient-to-br from-zinc-700 to-zinc-800 hover:from-orange-600 hover:to-orange-700 text-white border border-zinc-600 hover:border-orange-500'
            }`}
          >
            {isPlaying ? (
              <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
              </svg>
            ) : (
              <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z"/>
              </svg>
            )}
          </button>

          {/* Speed Control */}
          <div className="flex bg-zinc-800 rounded-lg overflow-hidden border border-zinc-700">
            {[1, 2, 5, 10].map(speed => (
              <button
                key={speed}
                onClick={() => onSpeedChange(speed)}
                className={`px-2 sm:px-3 py-1.5 text-xs sm:text-sm font-bold transition-all ${
                  playbackSpeed === speed
                    ? 'bg-gradient-to-b from-orange-500 to-orange-600 text-white shadow-inner'
                    : 'text-zinc-400 hover:text-white hover:bg-zinc-700'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>

          {/* Playback Status - Hidden on mobile */}
          {isPlaying && (
            <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 bg-orange-500/10 rounded-full border border-orange-500/30">
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
        <div className="text-center flex-1 min-w-0">
          <div className="text-sm sm:text-lg font-bold text-white font-mono truncate">
            {formatDateTime(new Date(currentTime))}
          </div>
        </div>

        {/* Time Range - Hidden on mobile */}
        <div className="hidden sm:block text-right text-xs">
          <div className="text-zinc-400">
            <span className="text-green-400">{formatTime(new Date(startTime))}</span>
            <span className="text-zinc-600 mx-1">-</span>
            <span className="text-red-400">{formatTime(new Date(endTime))}</span>
          </div>
          <div className="text-zinc-600">24h window</div>
        </div>
      </div>

      {/* Timeline Bar with Histogram */}
      <div
        ref={timelineRef}
        className="relative h-10 sm:h-12 bg-zinc-800 rounded-lg cursor-pointer overflow-hidden group"
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

        {/* Hour markers - Fewer on mobile */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between px-1 sm:px-2 pb-0.5 sm:pb-1">
          {[0, 12, 24].map(h => (
            <span key={h} className="text-[8px] sm:text-[10px] text-zinc-600 font-mono sm:hidden">
              {h === 0 ? '-24h' : h === 24 ? 'now' : `-${24-h}h`}
            </span>
          ))}
          {[0, 6, 12, 18, 24].map(h => (
            <span key={h} className="hidden sm:block text-[10px] text-zinc-600 font-mono">
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

  // Sound state
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [soundVolume, setSoundVolume] = useState(0.5);
  const lastPlayedEqRef = useRef(new Set()); // Track which earthquakes we've played sounds for

  // Mobile sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);

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

  // Initialize sound engine on mount (will activate on first user interaction due to browser policy)
  useEffect(() => {
    // Pre-initialize but audio won't actually play until user interacts with page
    soundEngine.init();
  }, []);

  // Sound toggle effect
  useEffect(() => {
    soundEngine.setEnabled(soundEnabled);
    soundEngine.setVolume(soundVolume);

    if (soundEnabled && isPlaying) {
      soundEngine.startAmbientDrone();
    } else {
      soundEngine.stopAmbientDrone();
    }

    return () => {
      soundEngine.stopAmbientDrone();
    };
  }, [soundEnabled, isPlaying, soundVolume]);

  // Play sounds when new earthquakes appear during playback
  useEffect(() => {
    if (!soundEnabled || !isPlaying) return;

    // Find earthquakes that just became visible (within tolerance)
    const tolerance = 60000; // 1 minute tolerance for sound triggering
    earthquakes.forEach(eq => {
      const eqTime = new Date(eq.time).getTime();
      // Check if earthquake just crossed the current time threshold
      if (eqTime <= currentTime && eqTime > currentTime - tolerance) {
        if (!lastPlayedEqRef.current.has(eq.id)) {
          lastPlayedEqRef.current.add(eq.id);
          soundEngine.playEarthquakeSound(eq.mag || 3);
        }
      }
    });
  }, [currentTime, earthquakes, soundEnabled, isPlaying]);

  // Reset played sounds when playback restarts
  useEffect(() => {
    if (currentTime <= timeRange.start + 1000) {
      lastPlayedEqRef.current.clear();
    }
  }, [currentTime, timeRange.start]);

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
      <header className="flex-shrink-0 bg-gradient-to-r from-zinc-900 to-zinc-800 border-b border-orange-500/50 z-50">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 py-2 sm:py-3">
          <div className="flex items-center justify-between gap-2">
            {/* Left: Back + Title */}
            <div className="flex items-center gap-3 sm:gap-4 min-w-0">
              <Link to="/" className="text-zinc-400 hover:text-white transition-colors flex-shrink-0">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </Link>
              <div className="min-w-0">
                <h1 className="text-base sm:text-xl font-bold text-orange-500 flex items-center gap-2 truncate">
                  <svg className="w-5 h-5 sm:w-6 sm:h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="hidden sm:inline">Earthquake</span> Time Machine
                </h1>
              </div>
            </div>

            {/* Right: Controls */}
            <div className="flex items-center gap-1 sm:gap-3 flex-shrink-0">
              {/* Sound Controls - Compact on mobile */}
              <div className="flex items-center gap-1 sm:gap-2 bg-zinc-800 rounded-full px-2 sm:px-3 py-1 sm:py-1.5 border border-zinc-700">
                <button
                  onClick={() => {
                    if (!soundEnabled) {
                      soundEngine.init();
                    }
                    setSoundEnabled(!soundEnabled);
                  }}
                  className={`p-0.5 sm:p-1 rounded-full transition-all ${
                    soundEnabled
                      ? 'text-orange-400 hover:text-orange-300'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                  title={soundEnabled ? 'Mute sounds' : 'Enable sounds'}
                >
                  {soundEnabled ? (
                    <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                    </svg>
                  )}
                </button>
                {/* Hide volume slider on mobile */}
                {soundEnabled && (
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={soundVolume * 100}
                    onChange={(e) => {
                      const vol = Number(e.target.value) / 100;
                      setSoundVolume(vol);
                      soundEngine.setVolume(vol);
                    }}
                    className="hidden sm:block w-16 h-1 bg-zinc-600 rounded-lg appearance-none cursor-pointer accent-orange-500"
                    title={`Volume: ${Math.round(soundVolume * 100)}%`}
                  />
                )}
              </div>

              {/* Live/Playback indicator - Compact on mobile */}
              <div className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-full ${
                isPlaying ? 'bg-orange-500/20' : 'bg-green-500/20'
              }`}>
                <div className={`w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full ${
                  isPlaying ? 'bg-orange-500 animate-pulse' : 'bg-green-500 animate-pulse'
                }`} />
                <span className={`text-[10px] sm:text-xs font-medium ${
                  isPlaying ? 'text-orange-400' : 'text-green-400'
                }`}>
                  {isPlaying ? 'PLAY' : 'LIVE'}
                </span>
              </div>

              {/* Mobile sidebar toggle */}
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-1.5 bg-zinc-800 rounded-lg border border-zinc-700 text-zinc-400 hover:text-white"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {sidebarOpen ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  )}
                </svg>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden relative">
        {/* Mobile Sidebar Overlay */}
        {sidebarOpen && (
          <div
            className="lg:hidden fixed inset-0 bg-black/50 z-40"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar - Slide out panel on mobile, fixed on desktop */}
        <div className={`
          fixed lg:relative inset-y-0 left-0 z-50 lg:z-auto
          w-72 lg:w-72 bg-zinc-800 lg:bg-zinc-800/50
          border-r border-zinc-700
          flex flex-col overflow-hidden
          transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}>
          {/* Mobile Close Button */}
          <div className="lg:hidden flex items-center justify-between p-2 border-b border-zinc-700 bg-zinc-900">
            <span className="text-orange-500 font-bold text-sm">Filters & Events</span>
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-1 text-zinc-400 hover:text-white"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

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
              zoom={1}
              minZoom={1}
              maxBoundsViscosity={1.0}
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

            {/* Map overlay stats - Compact on mobile */}
            <div className="absolute top-2 sm:top-4 right-2 sm:right-4 bg-zinc-900/90 backdrop-blur rounded-lg p-2 sm:p-3 border border-zinc-700 z-[1000]">
              <div className="text-center">
                <div className="text-xl sm:text-3xl font-bold text-orange-500 font-mono">
                  {visibleEarthquakes.length}
                </div>
                <div className="text-zinc-500 text-[8px] sm:text-[10px] uppercase">Events</div>
              </div>
            </div>

            {/* Visual Legend - Hidden on mobile, visible on larger screens */}
            <div className="hidden lg:block absolute bottom-12 right-4 bg-zinc-900/90 backdrop-blur rounded p-2 border border-zinc-700 z-[1000]">
              <div className="text-zinc-400 text-[9px] uppercase mb-1.5 font-bold">Legend</div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-500 animate-pulse" />
                  <span className="text-zinc-300 text-[10px]">New (&lt;10m)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-orange-500 opacity-60" />
                  <span className="text-zinc-300 text-[10px]">Fading (30-60m)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-orange-500 opacity-20" />
                  <span className="text-zinc-300 text-[10px]">Old (&gt;60m)</span>
                </div>
              </div>
            </div>

            {/* Instructions overlay - Simpler on mobile */}
            {!isPlaying && (
              <div className="absolute bottom-4 sm:bottom-6 left-1/2 -translate-x-1/2 bg-zinc-900/90 backdrop-blur rounded-lg px-3 py-1.5 sm:px-4 sm:py-2 border border-orange-500/50 z-[1000]">
                <div className="flex items-center gap-2 text-orange-400">
                  <svg className="w-5 h-5 sm:w-6 sm:h-6 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z"/>
                  </svg>
                  <span className="text-xs sm:text-sm font-bold">
                    {currentTime >= timeRange.end - 1000 ? 'Replay' : 'Press Play'}
                  </span>
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
