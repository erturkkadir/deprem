import { useSelector } from 'react-redux';

function StatusBadge({ verified, correct }) {
  if (!verified) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-yellow-900/30 text-yellow-400 border border-yellow-600">
        <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />
        Pending
      </span>
    );
  }

  if (correct) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-green-900/30 text-green-400 border border-green-600">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        Correct
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-900/30 text-red-400 border border-red-600">
      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
      Incorrect
    </span>
  );
}

export default function PredictionsTable() {
  const { predictions } = useSelector((state) => state.earthquake);

  const sortedPredictions = [...predictions].reverse();

  return (
    <section className="py-8">
      <div className="max-w-7xl mx-auto px-4">
        <h2 className="text-2xl font-bold text-orange-500 mb-6">
          Recent Predictions
        </h2>

        <div className="card overflow-hidden">
          {sortedPredictions.length === 0 ? (
            <div className="text-center py-12 text-zinc-500">
              <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <p>No predictions yet</p>
              <p className="text-sm mt-1">Make your first prediction above!</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-orange-500 text-zinc-900">
                    <th className="px-4 py-3 text-left font-semibold">ID</th>
                    <th className="px-4 py-3 text-left font-semibold">Timestamp</th>
                    <th className="px-4 py-3 text-right font-semibold">Predicted</th>
                    <th className="px-4 py-3 text-right font-semibold">Actual</th>
                    <th className="px-4 py-3 text-right font-semibold">Diff</th>
                    <th className="px-4 py-3 text-center font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPredictions.map((pred, index) => (
                    <tr
                      key={pred.id}
                      className={`border-b border-zinc-700 hover:bg-zinc-800/50 transition-colors ${
                        index === 0 ? 'bg-zinc-800/30' : ''
                      }`}
                    >
                      <td className="px-4 py-3 text-zinc-400">
                        #{pred.id}
                      </td>
                      <td className="px-4 py-3 text-zinc-300">
                        {new Date(pred.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-orange-400">
                        {pred.predicted_lat?.toFixed(2)}°
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-zinc-300">
                        {pred.actual_lat !== null ? `${pred.actual_lat?.toFixed(2)}°` : '—'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono">
                        {pred.difference !== undefined ? (
                          <span className={pred.difference <= 5 ? 'text-green-400' : pred.difference <= 10 ? 'text-yellow-400' : 'text-red-400'}>
                            {pred.difference?.toFixed(1)}°
                          </span>
                        ) : '—'}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <StatusBadge verified={pred.verified} correct={pred.correct} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
