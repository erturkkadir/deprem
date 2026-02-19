import { useState, useEffect, useCallback } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import axios from 'axios';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-xs">
      <p className="text-zinc-300 font-medium">{label}</p>
      <p className="text-orange-400">{payload[0].value.toLocaleString()} earthquakes</p>
    </div>
  );
};

export default function EarthquakeHistory() {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 640);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 640);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const today = new Date().toISOString().split('T')[0];
  const oneYearAgo = new Date(Date.now() - 365 * 86400000).toISOString().split('T')[0];

  const [startDate, setStartDate] = useState(oneYearAgo);
  const [endDate, setEndDate] = useState(today);
  const [minMag, setMinMag] = useState('2.0');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await axios.get(`/api/earthquake-history?start=${startDate}&end=${endDate}&min_mag=${minMag}`);
      setData(res.data);
    } catch (err) {
      console.error('Failed to fetch earthquake history:', err);
    } finally {
      setLoading(false);
    }
  }, [startDate, endDate, minMag]);

  useEffect(() => {
    if (isOpen && !data) fetchData();
  }, [isOpen]);

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
                <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h3 className="text-white font-bold text-sm sm:text-base">Historical Earthquake Patterns</h3>
                <p className="text-zinc-500 text-xs mt-0.5">
                  Distribution by month and day of month within a custom date range
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
              {/* Controls */}
              <div className="grid grid-cols-2 sm:flex sm:flex-wrap items-end gap-2 sm:gap-3 mb-5">
                <div className="col-span-1">
                  <label className="block text-zinc-500 text-[10px] uppercase tracking-wider mb-1">From</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={e => setStartDate(e.target.value)}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-orange-500/50"
                  />
                </div>
                <div className="col-span-1">
                  <label className="block text-zinc-500 text-[10px] uppercase tracking-wider mb-1">To</label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={e => setEndDate(e.target.value)}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-orange-500/50"
                  />
                </div>
                <div className="col-span-1">
                  <label className="block text-zinc-500 text-[10px] uppercase tracking-wider mb-1">Min Mag</label>
                  <select
                    value={minMag}
                    onChange={e => setMinMag(e.target.value)}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-orange-500/50"
                  >
                    <option value="2.0">M2.0+</option>
                    <option value="3.0">M3.0+</option>
                    <option value="4.0">M4.0+</option>
                    <option value="5.0">M5.0+</option>
                    <option value="6.0">M6.0+</option>
                  </select>
                </div>
                <div className="col-span-1 flex items-end gap-2">
                  <button
                    onClick={fetchData}
                    disabled={loading}
                    className="bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 border border-orange-500/30 rounded px-4 py-1.5 text-xs font-medium transition-colors disabled:opacity-50 flex-1 sm:flex-none"
                  >
                    {loading ? 'Loading...' : 'Apply'}
                  </button>
                  {data && (
                    <span className="text-zinc-500 text-[10px] sm:text-xs whitespace-nowrap">
                      {data.total.toLocaleString()}
                    </span>
                  )}
                </div>
              </div>

              {loading && !data && (
                <div className="text-center text-zinc-500 text-sm py-8">Loading...</div>
              )}

              {data && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* By Month */}
                  <div>
                    <h4 className="text-zinc-400 text-xs font-medium mb-2">Earthquakes by Month</h4>
                    <div className="h-52">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.by_month} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="label" tick={{ fill: '#a1a1aa', fontSize: isMobile ? 8 : 10 }} interval={0} />
                          <YAxis tick={{ fill: '#a1a1aa', fontSize: 10 }} />
                          <Tooltip content={<CustomTooltip />} />
                          <Bar dataKey="count" fill="#f97316" radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* By Day of Month */}
                  <div>
                    <h4 className="text-zinc-400 text-xs font-medium mb-2">Earthquakes by Day of Month</h4>
                    <div className="h-52">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.by_day} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                          <XAxis dataKey="day" tick={{ fill: '#a1a1aa', fontSize: 9 }} interval={isMobile ? 4 : 1} />
                          <YAxis tick={{ fill: '#a1a1aa', fontSize: 10 }} />
                          <Tooltip content={<CustomTooltip />} />
                          <Bar dataKey="count" fill="#f97316" radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
