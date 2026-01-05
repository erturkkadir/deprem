import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// API base URL - set to Python server address
// For same server: use ''
// For different server: use full URL like 'http://192.168.1.177:3000'
const API_BASE = import.meta.env.VITE_API_URL || '';

// Async thunks for API calls
export const fetchModelStatus = createAsyncThunk(
  'earthquake/fetchModelStatus',
  async () => {
    const response = await fetch(`${API_BASE}/api/model/status`);
    return response.json();
  }
);

export const fetchStats = createAsyncThunk(
  'earthquake/fetchStats',
  async () => {
    const response = await fetch(`${API_BASE}/api/stats`);
    return response.json();
  }
);

export const fetchLiveData = createAsyncThunk(
  'earthquake/fetchLiveData',
  async () => {
    const response = await fetch(`${API_BASE}/api/live`);
    return response.json();
  }
);

export const fetchPredictions = createAsyncThunk(
  'earthquake/fetchPredictions',
  async ({ page = 1, limit = 20, filter = '' } = {}) => {
    const params = new URLSearchParams({ page, limit });
    if (filter) params.append('filter', filter);
    const response = await fetch(`${API_BASE}/api/predictions?${params}`);
    return response.json();
  }
);

export const makePrediction = createAsyncThunk(
  'earthquake/makePrediction',
  async () => {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    return response.json();
  }
);

export const refreshUSGSData = createAsyncThunk(
  'earthquake/refreshUSGSData',
  async () => {
    const response = await fetch(`${API_BASE}/api/refresh`, {
      method: 'POST',
    });
    return response.json();
  }
);

export const triggerCycle = createAsyncThunk(
  'earthquake/triggerCycle',
  async () => {
    const response = await fetch(`${API_BASE}/api/cycle`, {
      method: 'POST',
    });
    return response.json();
  }
);

export const recordMatch = createAsyncThunk(
  'earthquake/recordMatch',
  async ({ predictionId, earthquake, distance }) => {
    const response = await fetch(`${API_BASE}/api/match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prediction_id: predictionId,
        earthquake_id: earthquake.id,
        earthquake_lat: earthquake.lat,
        earthquake_lon: earthquake.lon,
        earthquake_mag: earthquake.mag,
        earthquake_time: earthquake.time,
        distance: distance,
      }),
    });
    return response.json();
  }
);

export const verifyPrediction = createAsyncThunk(
  'earthquake/verifyPrediction',
  async ({ predictionId, actualLat, tolerance }) => {
    const response = await fetch(`${API_BASE}/api/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prediction_id: predictionId,
        actual_lat: actualLat,
        tolerance: tolerance || 5,
      }),
    });
    return response.json();
  }
);

const initialState = {
  // Model status
  modelStatus: {
    loaded: false,
    device: 'unknown',
    modelType: 'unknown',
    config: {},
    currentCheckpoint: null,
    checkpointTime: null,
    training: {
      latestStep: null,
      latestLoss: null,
      checkpointStep: null,
      checkpointLoss: null,
    },
  },

  // Statistics
  stats: {
    successRate: 0,
    totalPredictions: 0,
    verifiedPredictions: 0,
    correctPredictions: 0,
    lastUpdated: null,
  },

  // Predictions
  predictions: [],
  pagination: {
    page: 1,
    limit: 20,
    total: 0,
    total_pages: 1,
    has_next: false,
    has_prev: false,
  },
  currentPrediction: null,

  // Live data (for LiveDashboard)
  liveData: {
    latest_prediction: null,
    recent_earthquakes: [],
    stats: null,
    timestamp: null,
  },

  // UI State
  isLoading: false,
  isPredicting: false,
  isRefreshing: false,
  isLoadingLive: false,
  isCycling: false,
  error: null,

  // Live updates
  lastFetchTime: null,
  isConnected: false,

  // Match tracking
  isRecordingMatch: false,
  matchRecorded: false,
};

const earthquakeSlice = createSlice({
  name: 'earthquake',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    setConnected: (state, action) => {
      state.isConnected = action.payload;
    },
    clearCurrentPrediction: (state) => {
      state.currentPrediction = null;
    },
    resetMatchRecorded: (state) => {
      state.matchRecorded = false;
    },
  },
  extraReducers: (builder) => {
    builder
      // Model Status
      .addCase(fetchModelStatus.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchModelStatus.fulfilled, (state, action) => {
        state.isLoading = false;
        const training = action.payload.training || {};
        state.modelStatus = {
          loaded: action.payload.loaded,
          device: action.payload.device,
          modelType: action.payload.model_type,
          config: action.payload.config,
          currentCheckpoint: action.payload.current_checkpoint,
          checkpointTime: action.payload.checkpoint_time,
          training: {
            latestStep: training.latest_step,
            latestLoss: training.latest_loss,
            checkpointStep: training.checkpoint_step,
            checkpointLoss: training.checkpoint_loss,
          },
        };
        state.isConnected = true;
      })
      .addCase(fetchModelStatus.rejected, (state, action) => {
        state.isLoading = false;
        state.isConnected = false;
        state.error = action.error.message;
      })

      // Stats
      .addCase(fetchStats.fulfilled, (state, action) => {
        if (action.payload.success) {
          state.stats = {
            successRate: parseFloat(action.payload.stats.success_rate) || 0,
            totalPredictions: parseInt(action.payload.stats.total_predictions) || 0,
            verifiedPredictions: parseInt(action.payload.stats.verified_predictions) || 0,
            correctPredictions: parseInt(action.payload.stats.correct_predictions) || 0,
            lastUpdated: action.payload.stats.last_updated,
          };
          state.predictions = action.payload.recent_predictions || [];
        }
        state.lastFetchTime = Date.now();
      })

      // Prediction
      .addCase(makePrediction.pending, (state) => {
        state.isPredicting = true;
        state.error = null;
      })
      .addCase(makePrediction.fulfilled, (state, action) => {
        state.isPredicting = false;
        if (action.payload.success) {
          state.currentPrediction = action.payload.prediction;
        } else {
          state.error = action.payload.error;
        }
      })
      .addCase(makePrediction.rejected, (state, action) => {
        state.isPredicting = false;
        state.error = action.error.message;
      })

      // Refresh USGS
      .addCase(refreshUSGSData.pending, (state) => {
        state.isRefreshing = true;
      })
      .addCase(refreshUSGSData.fulfilled, (state) => {
        state.isRefreshing = false;
      })
      .addCase(refreshUSGSData.rejected, (state, action) => {
        state.isRefreshing = false;
        state.error = action.error.message;
      })

      // Live Data
      .addCase(fetchLiveData.pending, (state) => {
        state.isLoadingLive = true;
      })
      .addCase(fetchLiveData.fulfilled, (state, action) => {
        state.isLoadingLive = false;
        if (action.payload.success) {
          state.liveData = {
            latest_prediction: action.payload.latest_prediction,
            recent_earthquakes: action.payload.recent_earthquakes || [],
            stats: action.payload.stats,
            match_info: action.payload.match_info,
            closest_match: action.payload.closest_match,
            prediction_status: action.payload.prediction_status,
            server_time: action.payload.server_time,
          };
          state.isConnected = true;
        }
      })
      .addCase(fetchLiveData.rejected, (state, action) => {
        state.isLoadingLive = false;
        state.isConnected = false;
        state.error = action.error.message;
      })

      // Fetch Predictions
      .addCase(fetchPredictions.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchPredictions.fulfilled, (state, action) => {
        state.isLoading = false;
        if (action.payload.success) {
          state.predictions = action.payload.predictions || [];
          if (action.payload.pagination) {
            state.pagination = action.payload.pagination;
          }
        }
      })
      .addCase(fetchPredictions.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message;
      })

      // Trigger Cycle
      .addCase(triggerCycle.pending, (state) => {
        state.isCycling = true;
      })
      .addCase(triggerCycle.fulfilled, (state) => {
        state.isCycling = false;
      })
      .addCase(triggerCycle.rejected, (state, action) => {
        state.isCycling = false;
        state.error = action.error.message;
      })

      // Verify
      .addCase(verifyPrediction.fulfilled, (state, action) => {
        if (action.payload.success) {
          state.stats = {
            ...state.stats,
            ...action.payload.stats,
          };
        }
      })

      // Record Match
      .addCase(recordMatch.pending, (state) => {
        state.isRecordingMatch = true;
      })
      .addCase(recordMatch.fulfilled, (state, action) => {
        state.isRecordingMatch = false;
        if (action.payload.success) {
          state.matchRecorded = true;
          if (action.payload.stats) {
            state.stats = {
              successRate: parseFloat(action.payload.stats.success_rate) || 0,
              totalPredictions: parseInt(action.payload.stats.total_predictions) || 0,
              verifiedPredictions: parseInt(action.payload.stats.verified_predictions) || 0,
              correctPredictions: parseInt(action.payload.stats.correct_predictions) || 0,
              lastUpdated: Date.now(),
            };
          }
        }
      })
      .addCase(recordMatch.rejected, (state, action) => {
        state.isRecordingMatch = false;
        state.error = action.error.message;
      });
  },
});

export const { clearError, setConnected, clearCurrentPrediction, resetMatchRecorded } = earthquakeSlice.actions;
export default earthquakeSlice.reducer;
