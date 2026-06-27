# Gated Memory Cell Integration (LSTM / GRU Blocks)

**Gated Memory Cells** (like LSTMs and GRUs) use internal gating mechanisms to regulate information flow, creating a linear error carousel that allows gradients to flow backwards through time without vanishing.

## LSTM Gates
- **Forget Gate ($f_t$):** Controls how much of the cell state to throw away.
- **Input Gate ($i_t$):** Decides which new values to update in the state.
- **Output Gate ($o_t$):** Dictates what the next hidden state will be.

```mermaid
graph LR
    H_prev[h_{t-1}] --> Forget[Forget Gate]
    C_prev[C_{t-1}] --> CellState[Cell State C_t]
    Forget --> CellState
    Input[Input Gate] --> CellState
    CellState --> Output[Output Gate]
    Output --> H_curr[h_t]
```

[Back to README](../README.md)
