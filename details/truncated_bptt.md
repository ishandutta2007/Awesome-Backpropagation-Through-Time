# Truncated BPTT ($k_1, k_2$)

**Truncated BPTT ($k_1, k_2$)** is defined by two hyperparameters:
- $k_1$: The number of forward pass steps between parameter updates.
- $k_2$: The number of backward pass steps to propagate gradients.

## Computational Flow
Gradients are clipped at step $t - k_2$, stopping the backward flow of error signals beyond that window.

```mermaid
graph LR
    h_prev(...) --> h_tk2(h_{t-k_2})
    h_tk2 -- x Gradient Blocked x --> h_tk2_1(h_{t-k_2+1})
    h_tk2_1 --> h_t(h_t)
    h_t --> Loss(Loss)
    
    style h_tk2 fill:#ff9999,stroke:#333
    style h_tk2_1 fill:#99ff99,stroke:#333
```

[Back to README](../README.md)
