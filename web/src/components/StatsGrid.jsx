import { useSelector } from 'react-redux';

function StatCard({ value, label, color = 'orange', animate = false }) {
  const colorClasses = {
    orange: 'text-orange-500',
    green: 'text-green-500',
    blue: 'text-blue-500',
    purple: 'text-purple-500',
  };

  return (
    <div className="stat-card group">
      <div className={`text-4xl font-bold ${colorClasses[color]} ${animate ? 'animate-pulse' : ''}`}>
        {value}
      </div>
      <div className="text-zinc-400 text-sm mt-2 group-hover:text-zinc-300 transition-colors">
        {label}
      </div>
    </div>
  );
}

export default function StatsGrid() {
  const { stats, lastFetchTime } = useSelector((state) => state.earthquake);

  const formatTime = (timestamp) => {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const timeSinceUpdate = () => {
    if (!lastFetchTime) return '';
    const seconds = Math.floor((Date.now() - lastFetchTime) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <section className="py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-orange-500">
            Prediction Statistics
          </h2>
          <div className="flex items-center gap-2 text-zinc-500 text-sm">
            <svg className="w-4 h-4 animate-spin-slow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>Updated {timeSinceUpdate()}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            value={`${stats.successRate?.toFixed(1) || 0}%`}
            label="Success Rate"
            color="green"
            animate={stats.successRate > 50}
          />
          <StatCard
            value={stats.totalPredictions || 0}
            label="Total Predictions"
            color="blue"
          />
          <StatCard
            value={stats.correctPredictions || 0}
            label="Correct Predictions"
            color="purple"
          />
          <StatCard
            value={formatTime(stats.lastUpdated).split(',')[0] || 'Never'}
            label="Last Updated"
            color="orange"
          />
        </div>
      </div>
    </section>
  );
}
