import { useSelector } from 'react-redux';
import { useTranslation } from 'react-i18next';

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
  const { t } = useTranslation();
  const { stats, liveData, lastFetchTime } = useSelector((state) => state.earthquake);

  const timeSinceUpdate = () => {
    if (!lastFetchTime) return '';
    const seconds = Math.floor((Date.now() - lastFetchTime) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  // Use liveData.stats if available (updates every 10s), fallback to stats (updates every 2min)
  // "Ya bildik ya bilemedik": only earthquakes are graded; late catch = success.
  const liveStats = liveData?.stats;
  const total = parseInt(liveStats?.total_predictions ?? stats.totalPredictions) || 0;
  const caught = parseInt(liveStats?.events_caught) || 0;
  const late = parseInt(liveStats?.late_catches) || 0;
  const missedEvents = parseInt(liveStats?.events_missed) || 0;
  const eventSuccess = liveStats?.event_success != null ? parseFloat(liveStats.event_success) : null;

  return (
    <section className="py-4">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-bold text-orange-500">{t('stats.title')}</h2>
          <span className="text-zinc-500 text-xs">{t('stats.updated', { time: timeSinceUpdate() })}</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <StatCard
            value={eventSuccess != null ? `${eventSuccess.toFixed(0)}%` : '—'}
            label={t('live.eventSuccess')}
            color="green"
            animate={eventSuccess != null && eventSuccess >= 90}
          />
          <StatCard
            value={total}
            label={t('stats.total')}
            color="blue"
          />
          <StatCard
            value={caught}
            label={t('live.caught')}
            color="green"
          />
          <StatCard
            value={late}
            label={t('live.lateCatches')}
            color="blue"
          />
          <StatCard
            value={missedEvents}
            label={t('live.missedEvents')}
            color="purple"
          />
        </div>
      </div>
    </section>
  );
}
