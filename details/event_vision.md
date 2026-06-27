# Neuromorphic Event-Based Computer Vision Tracking

**Event-Based Cameras** output sparse, asynchronous pixel intensity changes (events) rather than full frame grids. Spiking Neural Networks (SNNs) trained with surrogate-gradient BPTT process these event streams with high temporal precision.

```mermaid
graph TD
    DVS[Dynamic Vision Sensor (Events)] --> SNN[Spiking Neural Network]
    SNN --> Out[Surrogate-Gradient BPTT Target Tracking]
```

## Features
- **Microsecond Latency:** Dynamic vision updates occur asynchronously.
- **Sparse Activations:** Spiking neurons only activate when a threshold is met, saving power.

[Back to README](../README.md)
