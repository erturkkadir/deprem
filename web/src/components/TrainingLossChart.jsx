import { useEffect, useState, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea } from 'recharts';
import axios from 'axios';

function TrainingLossChart() {
  const [lossData, setLossData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Zoom state
  const [zoomLeft, setZoomLeft] = useState(null);
  const [zoomRight, setZoomRight] = useState(null);
  const [refAreaLeft, setRefAreaLeft] = useState('');
  const [refAreaRight, setRefAreaRight] = useState('');
  const [isSelecting, setIsSelecting] = useState(false);

  useEffect(() => {
    const fetchLossData = async () => {
      try {
        const response = await axios.get('/api/training-loss?limit=2000');
        setLossData(response.data.loss_history || []);
        setError(null);
      } catch (err) {
        setError('Failed to load training data');
        console.error('Error fetching loss data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchLossData();
    const interval = setInterval(fetchLossData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Format step for display
  const formatStep = (step) => {
    if (step >= 1000000) return `${(step / 1000000).toFixed(1)}M`;
    if (step >= 1000) return `${(step / 1000).toFixed(1)}k`;
    return step;
  };

  // Get data within zoom range
  const getZoomedData = useCallback(() => {
    if (zoomLeft === null || zoomRight === null) return lossData;
    return lossData.filter(d => d.step >= zoomLeft && d.step <= zoomRight);
  }, [lossData, zoomLeft, zoomRight]);

  const zoomedData = getZoomedData();

  // Mouse handlers for zoom selection
  const handleMouseDown = (e) => {
    if (e && e.activeLabel) {
      setRefAreaLeft(e.activeLabel);
      setIsSelecting(true);
    }
  };

  const handleMouseMove = (e) => {
    if (isSelecting && e && e.activeLabel) {
      setRefAreaRight(e.activeLabel);
    }
  };

  const handleMouseUp = () => {
    if (refAreaLeft && refAreaRight) {
      const left = Math.min(refAreaLeft, refAreaRight);
      const right = Math.max(refAreaLeft, refAreaRight);

      if (right - left > 100) {
        setZoomLeft(left);
        setZoomRight(right);
      }
    }
    setRefAreaLeft('');
    setRefAreaRight('');
    setIsSelecting(false);
  };

  const resetZoom = () => {
    setZoomLeft(null);
    setZoomRight(null);
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-zinc-800 border border-zinc-700 rounded p-2 text-xs">
          <p className="text-zinc-400 mb-1">Step: {label?.toLocaleString()}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed(4)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">Training Progress</h3>
        <div className="h-64 flex items-center justify-center text-zinc-500">
          Loading...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">Training Progress</h3>
        <div className="h-64 flex items-center justify-center text-red-400 text-sm">
          {error}
        </div>
      </div>
    );
  }

  if (lossData.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
        <h3 className="text-sm font-medium text-zinc-400 mb-3">Training Progress</h3>
        <div className="h-64 flex items-center justify-center text-zinc-500 text-sm">
          No training data yet
        </div>
      </div>
    );
  }

  // Get stats from zoomed or full data
  const displayData = zoomedData.length > 0 ? zoomedData : lossData;
  const latest = lossData[lossData.length - 1] || {};
  const minTrainLoss = Math.min(...displayData.map(d => d.train_loss).filter(v => v != null));
  const minValLoss = Math.min(...displayData.map(d => d.val_loss).filter(v => v != null));
  const maxTrainLoss = Math.max(...displayData.map(d => d.train_loss).filter(v => v != null));
  const maxValLoss = Math.max(...displayData.map(d => d.val_loss).filter(v => v != null));

  const isZoomed = zoomLeft !== null && zoomRight !== null;

  return (
    <div className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-zinc-400">Training Progress</h3>
          {isZoomed && (
            <button
              onClick={resetZoom}
              className="px-2 py-0.5 text-[10px] bg-zinc-700 hover:bg-zinc-600 text-zinc-300 rounded transition-colors"
            >
              Reset Zoom
            </button>
          )}
        </div>
        <div className="flex gap-4 text-xs">
          <span className="text-blue-400">
            Train: {latest.train_loss?.toFixed(3)} (min: {minTrainLoss?.toFixed(3)})
          </span>
          <span className="text-green-400">
            Val: {latest.val_loss?.toFixed(3)} (min: {minValLoss?.toFixed(3)})
          </span>
          <span className="text-zinc-500">
            Step: {formatStep(latest.step)}
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="h-72 select-none">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={zoomedData}
            margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="step"
              tickFormatter={formatStep}
              stroke="#666"
              tick={{ fill: '#888', fontSize: 10 }}
              domain={['dataMin', 'dataMax']}
              allowDataOverflow
            />
            <YAxis
              stroke="#666"
              tick={{ fill: '#888', fontSize: 10 }}
              domain={[
                Math.floor(Math.min(minTrainLoss, minValLoss) * 0.95),
                Math.ceil(Math.max(maxTrainLoss, maxValLoss) * 1.05)
              ]}
              tickFormatter={(val) => val.toFixed(1)}
              allowDataOverflow
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '10px' }}
              iconSize={8}
            />
            <Line
              type="monotone"
              dataKey="train_loss"
              name="Train Loss"
              stroke="#60a5fa"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="val_loss"
              name="Val Loss"
              stroke="#4ade80"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
            {refAreaLeft && refAreaRight && (
              <ReferenceArea
                x1={refAreaLeft}
                x2={refAreaRight}
                strokeOpacity={0.3}
                fill="#60a5fa"
                fillOpacity={0.2}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer with instructions and stats */}
      <div className="mt-3 flex items-center justify-between text-[10px] text-zinc-500">
        <span>
          {isZoomed
            ? `Zoomed: ${formatStep(zoomLeft)} - ${formatStep(zoomRight)} (${zoomedData.length} points)`
            : 'Drag to zoom in on a region'
          }
        </span>
        <span>
          Random baseline: ~20.6 | Current: {latest.train_loss?.toFixed(2)} ({((20.6 - (latest.train_loss || 20.6)) / 20.6 * 100).toFixed(1)}% better)
        </span>
      </div>
    </div>
  );
}

export default TrainingLossChart;
