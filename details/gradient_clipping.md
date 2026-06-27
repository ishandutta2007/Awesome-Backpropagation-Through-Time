# Hard Gradient Norm Clipping

**Hard Gradient Norm Clipping** is a stabilization technique designed to prevent the exploding gradient problem in deep recurrent neural networks.

## Mechanism
Before the optimizer updates the weights, the total gradient vector $g$ is clipped if its norm exceeds a threshold:
$$g \leftarrow g \cdot \frac{\text{threshold}}{\max(\text{threshold}, \|g\|)}$$

```mermaid
graph TD
    Grad[Compute Gradient g] --> Check{||g|| > threshold?}
    Check -- Yes --> Scale[Scale: g * threshold/||g||]
    Check -- No --> Keep[Keep g unchanged]
    Scale --> Update[Update Weights]
    Keep --> Update[Update Weights]
```

[Back to README](../README.md)
