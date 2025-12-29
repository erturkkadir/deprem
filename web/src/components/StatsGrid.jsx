import { useSelector } from 'react-redux';

function StatCard({ value, label, color = 'orange', animate = false }) {
  const colorClasses = {
    orange: 'text-orange-500',
    green: 'text-green-500',
    blue: 'text-blue-500',
    purple: 'text-purple-500',
  };

  return (
    <div className="stat-card group !p-3">
      <div className={`text-2xl font-bold ${colorClasses[color]} ${animate ? 'animate-pulse' : ''}`}>
        {value}
      </div>
      <div className="text-zinc-400 text-xs mt-1 group-hover:text-zinc-300 transition-colors">
        {label}
      </div>
    </div>
  );
}

export default function StatsGrid() {
  const { stats, liveData, lastFetchTime } = useSelector((state) => state.earthquake);

  const timeSinceUpdate = () => {
    if (!lastFetchTime) return '';
    const seconds = Math.floor((Date.now() - lastFetchTime) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  // Use liveData.stats if available (updates every 10s), fallback to stats (updates every 2min)
  const liveStats = liveData?.stats;
  const total = liveStats?.total_predictions ?? stats.totalPredictions ?? 0;
  const verified = liveStats?.verified_predictions ?? stats.verifiedPredictions ?? 0;
  const matched = liveStats?.correct_predictions ?? stats.correctPredictions ?? 0;
  const missed = verified - matched;
  const pending = total - verified; // Should be 1 (current active prediction)
  const successRate = liveStats?.success_rate ?? stats.successRate ?? 0;

  return (
    <section className="py-4">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-bold text-orange-500">Statistics</h2>
          <span className="text-zinc-500 text-xs">Updated {timeSinceUpdate()}</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard
            value={total}
            label="Total"
            color="blue"
          />
          <StatCard
            value={matched}
            label="Matched"
            color="green"
          />
          <StatCard
            value={missed}
            label="Missed"
            color="purple"
          />
          <StatCard
            value={pending}
            label="Pending"
            color="orange"
            animate={pending > 0}
          />
          <StatCard
            value={`${successRate.toFixed(1)}%`}
            label="Success Rate"
            color="green"
            animate={successRate > 50}
          />
        </div>
      </div>
    </section>
  );
}
