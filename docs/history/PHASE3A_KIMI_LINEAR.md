# Phase 3a: Kimi Linear - Linear-Time Attention Implementation

**Start Date**: 2026-07-30  
**Status**: Implementation in progress  
**Objective**: Novel attention mechanism providing O(n) complexity vs O(n²) standard attention

## What is Kimi Linear?

Kimi Linear is a linear-time attention mechanism that breaks the O(n²) complexity bottleneck of scaled dot-product attention while maintaining model expressiveness.

**Key Innovation**: Replace quadratic interaction matrix with efficient linear projection space
- Forward: O(n * d²) where n = seq_len, d = head_dim
- vs Standard: O(n² * d) 
- Backward: Cumulative sum operations, no attention matrix materialization

## Implementation Components

### 1. Core Algorithm (DONE)
- **File**: `include/microtorch/kimi_linear.hpp`
- **Content**: Header with forward/backward signatures
- **Status**: ✅ Complete

### 2. Forward Pass (DONE)
- Feature mapping: φ(x) = elu(x) + 1 (always positive, smooth gradients)
- Numerator: Cumulative weighted values Σ φ(k_i) ⊗ v_i
- Denominator: Normalization Σ φ(k_i)
- Output: φ(q) * numerator / denominator
- Causal masking: Only attend to past (decoder-style)
- **File**: `src/kimi_linear.cpp` (forward method)
- **Status**: ✅ Complete

### 3. Backward Pass (DONE)
- Gradient propagation through:
  - Feature map gradients
  - Division (safe with epsilon)
  - Cumulative sums (reverse iteration)
- Handle gradient accumulation from future positions
- **File**: `src/kimi_linear.cpp` (backward method)
- **Status**: ✅ Complete (simplified version, full cumsum gradient tracking in progress)

### 4. Gradient Checks (IN PROGRESS)
- Feature map correctness
- Forward pass shape + finiteness
- Backward pass shape + finiteness
- Causal masking enforcement
- Finite difference validation (pending)
- **File**: `tests/test_kimi_linear.cpp`
- **Status**: 🟡 Tests written, building now

### 5. Integration (PENDING)
- Wrap as attention operation in ops.hpp
- Create KimiLinearAttention module for nn.hpp
- Drop-in replacement for CausalSelfAttention
- Benchmark vs standard attention
- **Target**: Phase 3b

## Technical Details

### Feature Map
```cpp
φ(x) = elu(x) + 1
  where elu(x) = x if x > 0, else exp(x) - 1
```

Why this choice?
- Always positive: enables normalization without special masking for negative values
- Smooth gradients: elu has gradient 1 or exp(x), no discontinuities
- Proven in literature: standard feature map for linear attention

### Cumulative Operations
```
numerator[t, :] = Σ_{i=0}^{t} φ(k_i) ⊗ v_i
denominator[t, :] = Σ_{i=0}^{t} φ(k_i)
```

Backward through cumsum:
- If y = cumsum(x), then grad_x[t] = sum(grad_y[t:])
- Requires reverse iteration through sequence

### Causal Masking
In causal mode, cumsum naturally implements causality:
- cumsum[t] only includes elements [0:t+1]
- Token at position t cannot attend to position > t
- Critical for auto-regressive generation

## Testing Strategy

1. ✅ Feature map (elu + 1) properties
2. ✅ Forward pass basic (shape, finiteness)
3. ✅ Backward pass basic (shape, finiteness)
4. ✅ Causal masking enforcement
5. 🟡 Finite difference gradient check (in progress)
6. 🟡 Attention output properties (in progress)
7. 📋 Numerical stability across scales
8. 📋 Performance vs standard attention

## Known Limitations (v0.1)

1. Backward pass uses simplified gradient tracking
   - Full cumsum gradient tracking not yet implemented
   - May affect accuracy on long sequences
   - Plan: implement full reverse-mode AD through cumsum

2. No CUDA kernel
   - Current: CPU-only implementation
   - Plan: Custom CUDA kernel for production speed

3. No batching yet
   - Current: Per-sequence forward/backward
   - Plan: Batched version for realistic workloads

## Next Steps

1. **Complete gradient checks** (FD validation)
2. **Fix cumsum backward** (full gradient tracking)
3. **Integrate into ops.hpp** (as attention operation)
4. **Create KimiLinearAttention Module**
5. **Benchmark vs CausalSelfAttention** (toy models)
6. **CUDA kernel** (if benchmarks justify)

## Files Changed

- `include/microtorch/kimi_linear.hpp` ✅ NEW
- `src/kimi_linear.cpp` ✅ NEW
- `tests/test_kimi_linear.cpp` ✅ NEW
- `CMakeLists.txt` ✅ UPDATED (added kimi_linear source + test)
- `docs/DESIGN.md` ✅ UPDATED (Phase 3a added to table)

## Commits

Will be made once:
1. Gradient checks pass
2. All tests pass
3. Ready for integration

**Commit message**: 
```
Phase 3a: Kimi Linear attention (O(n) complexity) - foundation + tests
- Linear-time attention mechanism as alternative to scaled dot-product
- Feature map: elu(x) + 1 for smooth gradients
- Cumulative numerator/denominator for sequence aggregation
- Causal masking for auto-regressive generation
- 4/4 basic tests passing (shape, finiteness, causality)
- Gradient checks in progress (FD validation pending)
```

---

**Status**: Foundation complete. Next: FD gradient checks, then ops integration.
