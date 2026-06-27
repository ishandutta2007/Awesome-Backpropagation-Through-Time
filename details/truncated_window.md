# The Truncated Window Era (Truncated BPTT / TBPTT)

The **Truncated Window Era** represents the practical shift towards handling long sequences by restricting the backward pass to a fixed size window.

## Concept
Instead of unrolling over the full length $T$, we partition the sequence into chunks of length $k_1$ (forward steps) and execute backpropagation for $k_2$ steps backwards (typically $k_2 = k_1$).

```mermaid
graph TD
    subgraph Window 1
        x1[x_1] --> h1(h_1)
        x2[x_2] --> h2(h_2)
        h1 --> h2
    end
    subgraph Window 2
        x3[x_3] --> h3(h_3)
        x4[x_4] --> h4(h_4)
        h2 -. Carry State .-> h3
        h3 --> h4
    end
    
    style h1 fill:#ccf,stroke:#333
    style h2 fill:#ccf,stroke:#333
    style h3 fill:#fcf,stroke:#333
    style h4 fill:#fcf,stroke:#333
```

## Benefits
- **Memory Bounded:** Limits the activation caching to $O(k_2)$ instead of $O(T)$.
- **Feasible Training:** Allows training on infinite streams by propagating the hidden state forward while discarding old computational graphs.

[Back to README](../README.md)
