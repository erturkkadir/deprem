import { configureStore } from '@reduxjs/toolkit';
import earthquakeReducer from './earthquakeSlice';

export const store = configureStore({
  reducer: {
    earthquake: earthquakeReducer,
  },
});

export default store;
