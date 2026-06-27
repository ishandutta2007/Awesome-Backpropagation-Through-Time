# High-Frequency Continuous Time-Series & Telemetry Prediction

LSTMs trained with BPTT are widely used to ingest and predict high-frequency continuous telemetry streams (e.g. cardiac ICU tracking, aerospace telemetry).

## System Overview
The model takes continuous sensor signals, runs them through recurrent networks to learn temporal dependencies, and outputs future predictions or anomaly detections.

```mermaid
graph LR
    Sensors[Continuous Sensor Data] --> Preprocess[Sliding Window Preprocessing]
    Preprocess --> LSTM[BPTT-Trained LSTM Layers]
    LSTM --> Anomaly[Anomaly Detection / Forecasts]
```

[Back to README](../README.md)
