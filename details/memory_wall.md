# The Hidden State Memory Wall

In deep neural networks trained with BPTT over long sequences, saving all hidden state activations for the backward pass requires massive memory, leading to out-of-memory (OOM) errors.

## Solution: Activation Checkpointing
Instead of caching every hidden state, we only cache boundary states (checkpoints) and recompute the intermediate states on-the-fly during the backward pass.

```mermaid
graph LR
    H0[State 0 (Saved)] --> H1(State 1)
    H1 --> H2(State 2)
    H2 --> H3[State 3 (Saved)]
    H3 --> H4(State 4)
    H4 --> H5(State 5)
    H5 --> H6[State 6 (Saved)]
    
    style H0 fill:#9f9,stroke:#333
    style H3 fill:#9f9,stroke:#333
    style H6 fill:#9f9,stroke:#333
```

[Back to README](../README.md)
