# Full Backpropagation Through Time (Full BPTT)

**Full BPTT** processes the entire sequence timeline from $t=1$ to $t=T$ and backpropagates errors across all steps.

## Mechanism
1. **Forward Pass:** Compute $h_t$ for $t = 1, \dots, T$. Keep all $h_t$ in memory.
2. **Backward Pass:** Calculate gradient starting from loss $\mathcal{L}_T$ all the way back to $t=1$.
3. **Update:** Apply gradient update to parameters once per sequence.

```mermaid
sequenceDiagram
    participant Input as Input Sequence (1..T)
    participant Hidden as Hidden States
    participant Loss as Loss / Output
    Note over Input,Loss: Forward Pass (1 to T)
    Input->>Hidden: Compute states h_1...h_T
    Hidden->>Loss: Compute Loss L
    Note over Input,Loss: Backward Pass (T to 1)
    Loss->>Hidden: Propagate gradients dL/dh_t
    Hidden->>Input: Accumulate parameter gradients
```

[Back to README](../README.md)
