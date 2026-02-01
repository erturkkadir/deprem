# Earthquake Prediction Matching System

## Terminology

- **Prediction Time**: When the prediction is made (e.g., `2026-02-01 09:56:00`)
- **Predicted Event Time**: When the earthquake is expected to occur (e.g., `2026-02-01 10:56:00`)
- **Prediction Window**: The interval between Prediction Time and Predicted Event Time (minimum 10 minutes)

## Matching Criteria

A recorded earthquake matches a prediction if **all** conditions are met:

| Parameter  | Tolerance       | Example                          |
|------------|-----------------|----------------------------------|
| Location   | ≤ 250 km radius | Within 250 km of (51°N, 97°E)    |
| Magnitude  | ± 0.5           | 3.7 – 4.7 for predicted M4.2     |
| Time       | See status flow | Depends on match timing          |

**Multiple Matches**: If multiple events satisfy the criteria, use the **first occurring** event.

## Status Flow
```
┌─────────────────────────────────────────────────────────────────────┐
│                        PREDICTION CREATED                           │
│                    Status: PENDING (countdown active)               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Event occurs within    │
                    │  prediction window?     │
                    └───────────┬─────────────┘
                          │           │
                         YES          NO
                          │           │
                          ▼           ▼
                    ┌─────────┐  ┌─────────────────────────┐
                    │ MATCHED │  │ MISSED                  │
                    │ (green) │  │ (yellow) - continue     │
                    │ (closed)│  │ searching for 72 hours  │
                    └─────────┘  └───────────┬─────────────┘
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │  Event occurs within    │
                                │  72h after predicted    │
                                │  event time?            │
                                └───────────┬─────────────┘
                                      │           │
                                     YES          NO
                                      │           │
                                      ▼           ▼
                                ┌───────────┐  ┌─────────┐
                                │ LATE      │  │ MISSED  │
                                │ CATCH     │  │ (red)   │
                                │ (orange)  │  │ (closed)│
                                └───────────┘  └─────────┘
```

## Status Definitions

| Status       | Color  | Description                                                      | Record State |
|--------------|--------|------------------------------------------------------------------|--------------|
| **PENDING**  | Blue   | Countdown active; displaying time remaining until predicted event | Open         |
| **MATCHED**  | Green  | Event occurred within prediction window                          | Closed*      |
| **MISSED**   | Yellow | Window expired; searching for late match (up to 72h)             | Open         |
| **LATE CATCH** | Orange | Event matched after window; display actual time + latency      | Closed*      |
| **MISSED (final)** | Red | 72h search period ended with no match                       | Closed*      |

_*Closed records remain viewable via click/expand_

## Configuration Parameters

| Parameter                | Value      | Notes                                                     |
|--------------------------|------------|----------------------------------------------------------|
| `MATCH_RADIUS_KM`        | 250        | Haversine distance from predicted coordinates             |
| `MAGNITUDE_TOLERANCE`    | 0.5        | Inclusive range: `[predicted - 0.5, predicted + 0.5]`     |
| `MIN_PREDICTION_WINDOW`  | 10 minutes | Reject predictions with window < 10 min                   |
| `LATE_SEARCH_DURATION`   | 72 hours   | Continue searching after MISSED before closing as final   |

## UI Behavior

### Countdown Display (PENDING)
- Show `HH:MM:SS` remaining until predicted event time
- Update every second

### MATCHED Display
- Actual event time
- Distance from predicted location (km)
- Magnitude delta (e.g., `+0.2` or `-0.3`)

### LATE CATCH Display
- Actual event time
- Latency: `+HH:MM:SS` (or `+Xd HH:MM:SS` for multi-day)
- Distance from predicted location (km)
- Magnitude delta

### MISSED (searching) Display
- Time since window closed
- Search time remaining (e.g., `68h 42m left`)

### MISSED (final) Display
- Mark as closed
- Show "No matching event within 72h"

### Visual Hierarchy
| Status           | Background | Border/Badge | Notes                        |
|------------------|------------|--------------|------------------------------|
| PENDING          | Light blue | Blue         | Pulse animation optional     |
| MATCHED          | Light green| Green        | Prominent, celebratory       |
| MISSED (searching)| Light yellow| Yellow      | Subtle warning               |
| LATE CATCH       | Light orange| Orange      | Notable but secondary        |
| MISSED (final)   | Light red  | Red          | Clear failure indicator      |

### Closed Record Access
- Collapsed by default in list view
- Expandable on click to show full prediction + outcome details
- Filter/sort options: by status, by date, by accuracy

## Validation Rules

1. **Minimum Window**: Reject prediction if `(Predicted Event Time - Prediction Time) < 10 minutes`
2. **Future Event**: Reject prediction if `Predicted Event Time ≤ Prediction Time`
3. **Valid Coordinates**: Latitude must be [-90, 90], Longitude must be [-180, 180]
4. **Valid Magnitude**: Must be positive (typically 0.0 – 10.0 range)

## Example Timeline
```
09:56:00  Prediction created (M4.2 at 51°N, 97°E expected at 10:56:00)
          Status: PENDING, countdown: 01:00:00

10:30:00  Countdown: 00:26:00
          Status: PENDING

10:56:00  Window closes, no event detected
          Status: MISSED (yellow), late search begins

10:58:30  M4.0 earthquake detected at 51.2°N, 96.8°E
          Distance: ~25 km ✓, Magnitude: 4.0 (delta: -0.2) ✓
          Status: LATE CATCH
          Display: "Event at 10:58:30 | +00:02:30 late | 25 km away | M4.0 (-0.2)"
```