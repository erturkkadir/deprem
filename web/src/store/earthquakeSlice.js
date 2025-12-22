import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Use relative path - Vite proxy will forward to Flask server
const API_BASE = '';

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
    const response = await fetch(`${API_BASE}/api/recent-earthquakes`);
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
  },

  // Statistics
  stats: {
    successRate: 0,
    totalPredictions: 0,
    correctPredictions: 0,
    lastUpdated: null,
  },

  // Predictions
  predictions: [],
  currentPrediction: null,

  // UI State
  isLoading: false,
  isPredicting: false,
  isRefreshing: false,
  error: null,

  // Live updates
  lastFetchTime: null,
  isConnected: false,
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
  },
  extraReducers: (builder) => {
    builder
      // Model Status
      .addCase(fetchModelStatus.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchModelStatus.fulfilled, (state, action) => {
        state.isLoading = false;
        state.modelStatus = {
          loaded: action.payload.loaded,
          device: action.payload.device,
          modelType: action.payload.model_type,
          config: action.payload.config,
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
            successRate: action.payload.stats.success_rate || 0,
            totalPredictions: action.payload.stats.total_predictions || 0,
            correctPredictions: action.payload.stats.correct_predictions || 0,
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

      // Verify
      .addCase(verifyPrediction.fulfilled, (state, action) => {
        if (action.payload.success) {
          state.stats = {
            ...state.stats,
            ...action.payload.stats,
          };
        }
      });
  },
});

export const { clearError, setConnected, clearCurrentPrediction } = earthquakeSlice.actions;
export default earthquakeSlice.reducer;
