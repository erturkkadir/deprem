import { useDispatch, useSelector } from 'react-redux';
import { makePrediction, refreshUSGSData, clearCurrentPrediction } from '../store/earthquakeSlice';

export default function PredictionPanel() {
  const dispatch = useDispatch();
  const { currentPrediction, isPredicting, isRefreshing, error } = useSelector(
    (state) => state.earthquake
  );

  const handlePredict = () => {
    dispatch(makePrediction());
  };

  const handleRefresh = () => {
    dispatch(refreshUSGSData());
  };

  const handleClear = () => {
    dispatch(clearCurrentPrediction());
  };

  return (
    <section className="py-8 bg-gradient-to-b from-zinc-900 to-zinc-800">
      <div className="max-w-7xl mx-auto px-4">
        <div className="card max-w-2xl mx-auto">
          <h2 className="text-2xl font-bold text-orange-500 mb-6 text-center">
            Make a Prediction
          </h2>

          <p className="text-zinc-400 text-center mb-8">
            Predict the approximate latitude of the next significant earthquake using our
            complex-valued transformer model trained on historical seismic data.
          </p>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
            <button
              onClick={handlePredict}
              disabled={isPredicting}
              className="btn-primary flex items-center justify-center gap-2"
            >
              {isPredicting ? (
                <>
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Predicting...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Predict Next Earthquake</span>
                </>
              )}
            </button>

            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="btn-secondary flex items-center justify-center gap-2"
            >
              {isRefreshing ? (
                <>
                  <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Refreshing...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span>Refresh USGS Data</span>
                </>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-900/30 border border-red-500 rounded-lg p-4 mb-6">
              <p className="text-red-400 text-center">{error}</p>
            </div>
          )}

          {/* Prediction Result */}
          {currentPrediction && (
            <div className="bg-zinc-800 border-2 border-orange-500 rounded-xl p-6 animate-fade-in">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-orange-400">Prediction Result</h3>
                <button
                  onClick={handleClear}
                  className="text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="text-center">
                <div className="text-5xl font-bold text-orange-500 mb-2">
                  {currentPrediction.latitude?.toFixed(2)}°
                </div>
                <p className="text-zinc-400">Predicted Latitude</p>

                <div className="mt-4 pt-4 border-t border-zinc-700">
                  <p className="text-sm text-zinc-500">
                    Raw value: {currentPrediction.raw_value} |
                    Time: {new Date(currentPrediction.timestamp).toLocaleString()}
                  </p>
                </div>

                {/* Visual indicator of location */}
                <div className="mt-6 relative h-8 bg-zinc-700 rounded-full overflow-hidden">
                  <div
                    className="absolute top-0 bottom-0 w-2 bg-orange-500 rounded-full transform -translate-x-1/2 transition-all duration-500"
                    style={{
                      left: `${((currentPrediction.latitude + 90) / 180) * 100}%`,
                    }}
                  />
                  <div className="absolute inset-0 flex justify-between items-center px-2 text-xs text-zinc-400">
                    <span>-90°</span>
                    <span>0°</span>
                    <span>+90°</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
