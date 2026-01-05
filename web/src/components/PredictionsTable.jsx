import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { fetchPredictions } from '../store/earthquakeSlice';

const FILTERS = [
  { key: '', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'matched', label: 'Matched' },
  { key: 'missed', label: 'Missed' },
];

function getStatus(pred) {
  if (!pred.verified) {
    // Calculate if prediction window has expired
    if (pred.prediction_time) {
      // Server times are UTC - append Z if not present to ensure correct parsing
      let timeStr = pred.prediction_time;
      if (!timeStr.endsWith('Z') && !timeStr.includes('+')) {
        timeStr += 'Z';
      }
      const predTime = new Date(timeStr);
      const now = new Date();
      // predicted_dt is in minutes, default to 60 if not set
      const windowMinutes = pred.predicted_dt || 60;
      const windowEndTime = new Date(predTime.getTime() + windowMinutes * 60 * 1000);
      // If current time is past the prediction window, it's missed
      if (now > windowEndTime) return 'missed';
    }
    return 'pending';
  }
  return pred.correct ? 'matched' : 'missed';
}

function FilterButton({ active, label, count, onClick, colorClass }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-1 rounded text-xs font-medium transition-all ${
        active
          ? `${colorClass} ring-1 ring-current`
          : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
      }`}
    >
      {label}
      {count !== undefined && (
        <span className={`ml-1 px-1 py-0.5 rounded text-[10px] ${
          active ? 'bg-black/20' : 'bg-zinc-700'
        }`}>
          {count}
        </span>
      )}
    </button>
  );
}

function StatusBadge({ pred }) {
  const status = getStatus(pred);

  if (status === 'pending') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-yellow-900/30 text-yellow-400 border border-yellow-600">
        <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />
        Pending
      </span>
    );
  }

  if (status === 'matched') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-green-900/30 text-green-400 border border-green-600">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        Matched
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-900/30 text-red-400 border border-red-600">
      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
      Missed
    </span>
  );
}

function TimeDiffCompare({ predictionTime, predictedWindow, actualTime, verified }) {
  // predictedWindow = the countdown window (e.g., 60 minutes)
  // actualTime = ISO timestamp when the matched earthquake occurred

  // Calculate how many minutes after prediction the EQ occurred
  let actualMinutesAfter = null;
  if (predictionTime && actualTime) {
    try {
      const predDate = new Date(predictionTime);
      const actualDate = new Date(actualTime);
      if (!isNaN(predDate.getTime()) && !isNaN(actualDate.getTime())) {
        actualMinutesAfter = Math.round((actualDate - predDate) / 60000);
      }
    } catch {
      actualMinutesAfter = null;
    }
  }

  const hasActual = actualMinutesAfter !== null;
  const windowMinutes = predictedWindow || 60;

  // Format time nicely
  const formatTime = (minutes) => {
    if (minutes === null || minutes === undefined) return '—';
    const absMin = Math.abs(minutes);
    if (absMin < 60) return `${absMin}m`;
    if (absMin < 1440) return `${(absMin / 60).toFixed(1)}h`;
    return `${(absMin / 1440).toFixed(1)}d`;
  };

  // Check validity and status
  const isInvalidMatch = hasActual && actualMinutesAfter < 0; // EQ before prediction = bad data
  const isWithinWindow = hasActual && actualMinutesAfter >= 0 && actualMinutesAfter <= windowMinutes;

  return (
    <div className="bg-zinc-800/50 rounded p-2">
      <div className="text-[10px] font-medium mb-1 text-purple-400">TIME WINDOW</div>

      <div className="flex items-center justify-between gap-2">
        {/* Predicted Window */}
        <div className="text-center flex-1">
          <div className="text-[9px] text-zinc-500 uppercase">Window</div>
          <div className="font-mono font-bold text-sm text-purple-400">
            {formatTime(windowMinutes)}
          </div>
        </div>

        {/* Arrow */}
        <div className="text-zinc-600 text-xs">→</div>

        {/* When EQ occurred */}
        <div className="text-center flex-1">
          <div className="text-[9px] text-zinc-500 uppercase">Occurred</div>
          <div className={`font-mono font-bold text-sm ${
            isInvalidMatch ? 'text-red-400' : 'text-white'
          }`}>
            {hasActual ? (isInvalidMatch ? `${actualMinutesAfter}m` : `+${formatTime(actualMinutesAfter)}`) : '—'}
          </div>
        </div>
      </div>

      {/* Status bar */}
      {verified && hasActual && (
        <div className={`mt-1.5 text-center text-[10px] font-medium px-1.5 py-0.5 rounded ${
          isInvalidMatch
            ? 'bg-red-900/30 text-red-400'
            : isWithinWindow
              ? 'bg-green-900/30 text-green-400'
              : 'bg-yellow-900/30 text-yellow-400'
        }`}>
          {isInvalidMatch
            ? '⚠ Invalid: Before prediction'
            : isWithinWindow
              ? `✓ Match at +${formatTime(actualMinutesAfter)}`
              : `Outside window (+${formatTime(actualMinutesAfter)})`}
        </div>
      )}
    </div>
  );
}

function ParamCompare({ label, predicted, actual, diff, unit, tolerance, colorClass }) {
  const hasActual = actual !== null && actual !== undefined;
  const hasDiff = diff !== null && diff !== undefined;
  const isWithinTolerance = hasDiff && Math.abs(diff) <= tolerance;

  return (
    <div className="bg-zinc-800/50 rounded p-2">
      <div className={`text-[10px] font-medium mb-1 ${colorClass}`}>{label}</div>

      <div className="flex items-center justify-between gap-2">
        {/* Predicted */}
        <div className="text-center flex-1">
          <div className="text-[9px] text-zinc-500 uppercase">Pred</div>
          <div className={`font-mono font-bold text-sm ${colorClass}`}>
            {predicted !== null && predicted !== undefined
              ? `${Number(predicted).toFixed(1)}${unit}`
              : '—'}
          </div>
        </div>

        {/* Arrow */}
        <div className="text-zinc-600 text-xs">→</div>

        {/* Actual */}
        <div className="text-center flex-1">
          <div className="text-[9px] text-zinc-500 uppercase">Actual</div>
          <div className="font-mono font-bold text-sm text-white">
            {hasActual ? `${Number(actual).toFixed(1)}${unit}` : '—'}
          </div>
        </div>
      </div>

      {/* Difference bar */}
      {hasDiff && (
        <div className={`mt-1.5 text-center text-[10px] font-medium px-1.5 py-0.5 rounded ${
          isWithinTolerance
            ? 'bg-green-900/30 text-green-400'
            : 'bg-red-900/30 text-red-400'
        }`}>
          {isWithinTolerance ? '✓' : '✗'} {diff > 0 ? '+' : ''}{Number(diff).toFixed(1)}{unit}
        </div>
      )}
    </div>
  );
}

export default function PredictionsTable() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { predictions, pagination, isLoading } = useSelector((state) => state.earthquake);
  const [filter, setFilter] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [, setTick] = useState(0);

  // Fetch predictions on mount and when filter/page changes
  useEffect(() => {
    dispatch(fetchPredictions({ page: currentPage, limit: 20, filter }));
  }, [dispatch, filter, currentPage]);

  // Force re-render every 30 seconds to update expired statuses
  useEffect(() => {
    const interval = setInterval(() => {
      setTick(t => t + 1);
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
  };

  const handleFilterChange = (newFilter) => {
    setFilter(newFilter);
    setCurrentPage(1); // Reset to page 1 when filter changes
  };

  const openMapInNewTab = (pred) => {
    // Calculate window end time
    const windowMinutes = pred.predicted_dt || 60;
    let windowEndTime = '';
    if (pred.prediction_time) {
      const predDate = new Date(pred.prediction_time);
      const windowEnd = new Date(predDate.getTime() + windowMinutes * 60 * 1000);
      windowEndTime = windowEnd.toISOString();
    }

    const params = new URLSearchParams({
      id: pred.id || '',
      plat: pred.predicted_lat ?? '',
      plon: pred.predicted_lon ?? '',
      pmag: pred.predicted_mag ?? '',
      pdt: pred.predicted_dt ?? '',
      pplace: pred.predicted_place || '',
      alat: pred.actual_lat ?? '',
      alon: pred.actual_lon ?? '',
      amag: pred.actual_mag ?? '',
      adt: pred.actual_dt ?? '',
      aplace: pred.actual_place || '',
      atime: pred.actual_time || '', // Exact actual earthquake time
      dlat: pred.diff_lat ?? '',
      dlon: pred.diff_lon ?? '',
      verified: pred.verified ? 'true' : 'false',
      correct: pred.correct ? 'true' : 'false',
      time: pred.prediction_time || '', // Prediction start time
      wend: windowEndTime, // Window end time
    });
    window.open(`/map.html?${params.toString()}`, '_blank');
  };

  const filterColors = {
    '': 'bg-orange-900/30 text-orange-400',
    pending: 'bg-yellow-900/30 text-yellow-400',
    matched: 'bg-green-900/30 text-green-400',
    missed: 'bg-red-900/30 text-red-400',
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '—';
    try {
      const date = new Date(timeStr);
      if (isNaN(date.getTime())) return '—';
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return '—';
    }
  };

  const getTimeAgo = (timeStr) => {
    if (!timeStr) return '';
    try {
      const date = new Date(timeStr);
      if (isNaN(date.getTime())) return '';
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return 'just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      return `${diffDays}d ago`;
    } catch {
      return '';
    }
  };

  return (
    <section className="py-4">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-bold text-orange-500">Predictions</h2>
            <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">
              {pagination.total} total
            </span>
          </div>
          <div className="flex gap-1.5">
            {FILTERS.map(f => (
              <FilterButton
                key={f.key}
                active={filter === f.key}
                label={f.label}
                onClick={() => handleFilterChange(f.key)}
                colorClass={filterColors[f.key]}
              />
            ))}
          </div>
        </div>

        {isLoading ? (
          <div className="card text-center py-6 text-zinc-500">
            <div className="w-6 h-6 border-2 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <p className="text-sm">Loading...</p>
          </div>
        ) : predictions.length === 0 ? (
          <div className="card text-center py-6 text-zinc-500">
            <svg className="w-10 h-10 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-sm">{filter === '' ? 'No predictions yet' : `No ${filter} predictions`}</p>
          </div>
        ) : (
          <div className="space-y-2">
            {predictions.map((pred, index) => (
              <div
                key={pred.id}
                className={`card !p-3 cursor-pointer hover:bg-zinc-800/80 transition-colors ${index === 0 ? 'ring-1 ring-orange-500/50' : ''}`}
                onClick={() => navigate(`/prediction/${pred.id}`)}
              >
                {/* Header row */}
                <div className="flex items-center justify-between mb-2 pb-2 border-b border-zinc-700">
                  <div className="flex items-center gap-3">
                    <span className="text-lg font-bold text-orange-500">#{pred.id}</span>
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-white text-sm font-medium">{formatTime(pred.prediction_time)}</span>
                        <span className="text-[10px] text-zinc-500 bg-zinc-800 px-1.5 py-0.5 rounded">{getTimeAgo(pred.prediction_time)}</span>
                      </div>
                      <div className="flex items-center gap-1.5 text-xs">
                        <svg className="w-3 h-3 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        </svg>
                        <span className="text-orange-300 font-medium truncate max-w-[200px]">
                          {pred.predicted_place || `(${pred.predicted_lat}°, ${pred.predicted_lon}°)`}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => { e.stopPropagation(); openMapInNewTab(pred); }}
                      className="flex items-center gap-1 px-2 py-1 rounded-lg bg-blue-900/50 hover:bg-blue-800/70 text-blue-400 hover:text-blue-300 border border-blue-700 transition-colors text-xs font-medium"
                      title="View on map"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                      Map
                    </button>
                    <StatusBadge pred={pred} />
                  </div>
                </div>

                {/* 4 Parameter Grid */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                  <ParamCompare
                    label="LATITUDE"
                    predicted={pred.predicted_lat}
                    actual={pred.actual_lat}
                    diff={pred.diff_lat}
                    unit="°"
                    tolerance={10}
                    colorClass="text-orange-400"
                  />
                  <ParamCompare
                    label="LONGITUDE"
                    predicted={pred.predicted_lon}
                    actual={pred.actual_lon}
                    diff={pred.diff_lon}
                    unit="°"
                    tolerance={20}
                    colorClass="text-blue-400"
                  />
                  <TimeDiffCompare
                    predictionTime={pred.prediction_time}
                    predictedWindow={pred.predicted_dt}
                    actualTime={pred.actual_time}
                    verified={pred.verified}
                  />
                  <ParamCompare
                    label="MAGNITUDE"
                    predicted={pred.predicted_mag}
                    actual={pred.actual_mag}
                    diff={pred.diff_mag}
                    unit=""
                    tolerance={1.0}
                    colorClass="text-red-400"
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {pagination.total_pages > 1 && (
          <div className="flex items-center justify-center gap-2 mt-4">
            <button
              onClick={() => handlePageChange(1)}
              disabled={currentPage <= 1}
              className="px-2 py-1 rounded text-xs bg-zinc-800 text-zinc-400 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              First
            </button>
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage <= 1}
              className="px-3 py-1 rounded text-xs bg-zinc-800 text-zinc-400 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Prev
            </button>
            <span className="px-3 py-1 text-xs text-zinc-300">
              Page {currentPage} of {pagination.total_pages}
            </span>
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage >= pagination.total_pages}
              className="px-3 py-1 rounded text-xs bg-zinc-800 text-zinc-400 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Next
            </button>
            <button
              onClick={() => handlePageChange(pagination.total_pages)}
              disabled={currentPage >= pagination.total_pages}
              className="px-2 py-1 rounded text-xs bg-zinc-800 text-zinc-400 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Last
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
