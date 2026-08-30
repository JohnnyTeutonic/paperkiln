#include "microtorch/ops.hpp"
#include "microtorch/device.hpp"
#include "microtorch/kimi_linear.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace microtorch {
namespace ops {

namespace {

// Attach a tape node to `out` unless grad is globally off or no input
// requires it. The closure convention throughout: read out->grad, guard
// each parent on requires_grad, accumulate.
Var record(Matrix result, std::vector<Var> parents, std::function<void(Variable*)> bw) {
    bool needs = false;
    if (grad_enabled()) {
        for (const auto& p : parents) needs = needs || p->requires_grad;
    }
    Var out = make_var(std::move(result), needs);
    if (needs) {
        out->parents = std::move(parents);
        Variable* self = out.get();
        out->backward_fn = [self, bw]() { bw(self); };
    }
    return out;
}

}  // namespace

Var matmul(const Var& a, const Var& b) {
    if (a->data.cols() != b->data.rows()) {
        throw std::runtime_error("matmul: inner dimensions disagree");
    }
    // Phase B2.0 (docs/CUDA_PHASE_B2.md): transpose-flag gemm. Under an
    // open step window the operands' device copies are cached in the
    // Variables' DevState slots (params upload once per step, not once
    // per use) and the backward transposes happen in kernel index math.
    // Grads carry no DevState in B2.0 -> nullptr slots. Off the B2 path
    // this runs exactly the pre-B2 ops (materialized transpose + the
    // same matmul), bit-identical.
    using device::Trans;
    Matrix c = device::gemm(a->data, &a->dev, Trans::N, b->data, &b->dev, Trans::N);
    return record(std::move(c), {a, b}, [](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        if (a->requires_grad) {
            a->accumulate(device::gemm(self->grad, nullptr, Trans::N,
                                       b->data, &b->dev, Trans::T));
        }
        if (b->requires_grad) {
            b->accumulate(device::gemm(a->data, &a->dev, Trans::T,
                                       self->grad, nullptr, Trans::N));
        }
    });
}

Var add(const Var& a, const Var& b) {
    device::materialize(a->data);
    device::materialize(b->data);
    return record(a->data + b->data, {a, b}, [](Variable* self) {
        if (self->parents[0]->requires_grad) self->parents[0]->accumulate(self->grad);
        if (self->parents[1]->requires_grad) self->parents[1]->accumulate(self->grad);
    });
}

Var sub(const Var& a, const Var& b) {
    device::materialize(a->data);
    device::materialize(b->data);
    return record(a->data - b->data, {a, b}, [](Variable* self) {
        if (self->parents[0]->requires_grad) self->parents[0]->accumulate(self->grad);
        if (self->parents[1]->requires_grad) {
            self->parents[1]->accumulate(self->grad * -1.0f);
        }
    });
}

Var mul(const Var& a, const Var& b) {
    device::materialize(a->data);
    device::materialize(b->data);
    return record(a->data.hadamard(b->data), {a, b}, [](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        if (a->requires_grad) a->accumulate(self->grad.hadamard(b->data));
        if (b->requires_grad) b->accumulate(self->grad.hadamard(a->data));
    });
}

Var add_bias(const Var& x, const Var& b) {
    if (b->data.rows() != 1 || b->data.cols() != x->data.cols()) {
        throw std::runtime_error("add_bias: bias must be [1, cols(x)]");
    }
    device::materialize(x->data);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) += b->data(0, j);
    return record(std::move(out), {x, b}, [](Variable* self) {
        const Var& x = self->parents[0];
        const Var& b = self->parents[1];
        if (x->requires_grad) x->accumulate(self->grad);
        if (b->requires_grad) {
            Matrix db(1, b->data.cols());                       // column-sum, the same
            for (size_t i = 0; i < self->grad.rows(); ++i)      // contract as
                for (size_t j = 0; j < self->grad.cols(); ++j)  // compute_bias_
                    db(0, j) += self->grad(i, j);               // gradients_kernel
            b->accumulate(db);
        }
    });
}

Var gelu(const Var& x) {
    // B2.1a seam: same tanh-GELU formula on device (cuda_ops.cu) or the
    // vendored apply_gelu on host; entry returns false -> CPU, unchanged.
    Matrix out(x->data.rows(), x->data.cols());
    if (!device::devops::gelu_fwd(x->data, out)) {
        out = x->data;
        out.apply_gelu();
    }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // Correct tanh-GELU derivative, matching the (verified) CUDA kernel
        // and NOT Matrix::apply_gelu_derivative -- see primitives.hpp.
        //   u  = sqrt(2/pi) (x + 0.044715 x^3)
        //   d  = cdf(u) + x * 0.5 sech^2(u) * u'
        Matrix dx(self->grad.rows(), self->grad.cols());
        if (!device::devops::gelu_bwd(x->data, self->grad, dx)) {
            constexpr float k = 0.7978845608028654f;
            dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i) {
                for (size_t j = 0; j < dx.cols(); ++j) {
                    float v = x->data(i, j);
                    float u = k * (v + 0.044715f * v * v * v);
                    float t = std::tanh(u);
                    float cdf = 0.5f * (1.0f + t);
                    float pdf = 0.5f * (1.0f - t * t) * k * (1.0f + 0.134145f * v * v);
                    dx(i, j) *= cdf + v * pdf;
                }
            }
        }
        x->accumulate(dx);
    });
}

Var softmax_row(const Var& x) {
    // B2.1a seam: rowwise max-subtracted softmax on device, or the
    // vendored apply_softmax on host.
    Matrix out(x->data.rows(), x->data.cols());
    if (!device::devops::softmax_fwd(x->data, out)) {
        out = x->data;
        out.apply_softmax();
    }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // dX = S .* (dY - rowsum(dY .* S)) -- the same formula
        // attention_ops.cu's batched_softmax_backward_kernel uses.
        const Matrix& S = self->data;
        const Matrix& dY = self->grad;
        Matrix dX(S.rows(), S.cols());
        if (!device::devops::softmax_bwd(S, dY, dX)) {
            for (size_t i = 0; i < S.rows(); ++i) {
                float dot = 0.0f;
                for (size_t j = 0; j < S.cols(); ++j) dot += dY(i, j) * S(i, j);
                for (size_t j = 0; j < S.cols(); ++j) {
                    dX(i, j) = S(i, j) * (dY(i, j) - dot);
                }
            }
        }
        x->accumulate(dX);
    });
}

Var mean(const Var& x) {
    device::materialize(x->data);
    Matrix out(1, 1);
    float sum = 0.0f;
    for (size_t i = 0; i < x->data.rows(); ++i)
        for (size_t j = 0; j < x->data.cols(); ++j) sum += x->data(i, j);
    const float n = static_cast<float>(x->data.rows() * x->data.cols());
    out(0, 0) = sum / n;
    return record(std::move(out), {x}, [n](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(x->data.rows(), x->data.cols(), self->grad(0, 0) / n);
        x->accumulate(dx);
    });
}

Var scale(const Var& x, float s) {
    device::materialize(x->data);
    return record(x->data * s, {x}, [s](Variable* self) {
        if (self->parents[0]->requires_grad) {
            self->parents[0]->accumulate(self->grad * s);
        }
    });
}

Var transpose(const Var& x) {
    device::materialize(x->data);
    return record(x->data.transpose(), {x}, [](Variable* self) {
        if (self->parents[0]->requires_grad) {
            self->parents[0]->accumulate(self->grad.transpose());
        }
    });
}

Var slice_cols(const Var& x, size_t j0, size_t j1) {
    if (j1 <= j0 || j1 > x->data.cols()) {
        throw std::runtime_error("slice_cols: bad range");
    }
    device::materialize(x->data);
    Matrix out(x->data.rows(), j1 - j0);
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) = x->data(i, j0 + j);
    return record(std::move(out), {x}, [j0](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < self->grad.rows(); ++i)
            for (size_t j = 0; j < self->grad.cols(); ++j) dx(i, j0 + j) = self->grad(i, j);
        x->accumulate(dx);
    });
}

Var concat_cols(const std::vector<Var>& xs) {
    if (xs.empty()) throw std::runtime_error("concat_cols: empty input");
    size_t rows = xs[0]->data.rows(), cols = 0;
    for (const auto& x : xs) {
        if (x->data.rows() != rows) {
            throw std::runtime_error("concat_cols: row mismatch");
        }
        cols += x->data.cols();
    }
    for (const auto& x : xs) device::materialize(x->data);
    Matrix out(rows, cols);
    size_t off = 0;
    for (const auto& x : xs) {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < x->data.cols(); ++j) out(i, off + j) = x->data(i, j);
        off += x->data.cols();
    }
    return record(std::move(out), xs, [](Variable* self) {
        size_t off = 0;
        for (const auto& p : self->parents) {
            const size_t w = p->data.cols();
            if (p->requires_grad) {
                Matrix dp(p->data.rows(), w);
                for (size_t i = 0; i < dp.rows(); ++i)
                    for (size_t j = 0; j < w; ++j) dp(i, j) = self->grad(i, off + j);
                p->accumulate(dp);
            }
            off += w;
        }
    });
}

Var layernorm(const Var& x, const Var& gamma, const Var& beta, float eps) {
    const size_t R = x->data.rows(), C = x->data.cols();
    if (gamma->data.rows() != 1 || gamma->data.cols() != C || beta->data.rows() != 1 ||
        beta->data.cols() != C) {
        throw std::runtime_error("layernorm: gamma/beta must be [1, cols(x)]");
    }
    // Cache xhat and 1/std for the backward -- same normalisation the
    // canonical layer_norm_stats_kernel/layer_norm_kernel pair computes.
    auto xhat = std::make_shared<Matrix>(R, C);
    auto rstd = std::make_shared<std::vector<float>>(R);
    Matrix out(R, C);
    // B2.1a seam: one kernel computes out/xhat/rstd; the caches land on
    // host exactly as the CPU loop leaves them (write-through), so the
    // backward below is device/CPU agnostic.
    if (!device::devops::layernorm_fwd(x->data, gamma->data, beta->data, eps,
                                       out, *xhat, *rstd)) {
        for (size_t i = 0; i < R; ++i) {
            float mu = 0.0f;
            for (size_t j = 0; j < C; ++j) mu += x->data(i, j);
            mu /= static_cast<float>(C);
            float var = 0.0f;
            for (size_t j = 0; j < C; ++j) {
                const float d = x->data(i, j) - mu;
                var += d * d;
            }
            var /= static_cast<float>(C);
            const float rs = 1.0f / std::sqrt(var + eps);
            (*rstd)[i] = rs;
            for (size_t j = 0; j < C; ++j) {
                const float xh = (x->data(i, j) - mu) * rs;
                (*xhat)(i, j) = xh;
                out(i, j) = gamma->data(0, j) * xh + beta->data(0, j);
            }
        }
    }
    return record(std::move(out), {x, gamma, beta}, [xhat, rstd](Variable* self) {
        const Var& x = self->parents[0];
        const Var& g = self->parents[1];
        const Var& b = self->parents[2];
        const Matrix& dY = self->grad;
        const size_t R = dY.rows(), C = dY.cols();
        const bool want_dgb = g->requires_grad || b->requires_grad;
        // B2.1a seam: dgamma/dbeta column sums + the dx row formula in
        // kernels; the fall-through runs the loops below unchanged.
        Matrix dg(1, C), db(1, C), dx(R, C);
        const bool on_dev = device::devops::layernorm_bwd(
            dY, *xhat, *rstd, g->data, want_dgb, &dg, &db, x->requires_grad,
            &dx);
        if (want_dgb) {
            if (!on_dev) {
                for (size_t i = 0; i < R; ++i)
                    for (size_t j = 0; j < C; ++j) {
                        dg(0, j) += dY(i, j) * (*xhat)(i, j);
                        db(0, j) += dY(i, j);
                    }
            }
            if (g->requires_grad) g->accumulate(dg);
            if (b->requires_grad) b->accumulate(db);
        }
        if (!x->requires_grad) return;
        // dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat .* xhat))
        if (!on_dev) {
            for (size_t i = 0; i < R; ++i) {
                float m1 = 0.0f, m2 = 0.0f;
                for (size_t j = 0; j < C; ++j) {
                    const float dxh = dY(i, j) * g->data(0, j);
                    m1 += dxh;
                    m2 += dxh * (*xhat)(i, j);
                }
                m1 /= static_cast<float>(C);
                m2 /= static_cast<float>(C);
                for (size_t j = 0; j < C; ++j) {
                    const float dxh = dY(i, j) * g->data(0, j);
                    dx(i, j) = (*rstd)[i] * (dxh - m1 - (*xhat)(i, j) * m2);
                }
            }
        }
        x->accumulate(dx);
    });
}

Var embedding(const Var& table, const std::vector<int>& ids) {
    const size_t d = table->data.cols();
    Matrix out(ids.size(), d);
    // Bounds are checked HOST-SIDE for both paths (a kernel cannot throw).
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= table->data.rows()) {
            throw std::runtime_error("embedding: id out of range");
        }
    }
    // B2.2: gather on-device (the forward's first activation is born
    // resident); host loop as fallback. Backward scatter-add stays host
    // until B2.3 (table grad is host-authoritative).
    if (!device::devops::embed_gather(table->data, ids.data(), ids.size(),
                                      out)) {
        for (size_t i = 0; i < ids.size(); ++i)
            for (size_t j = 0; j < d; ++j) out(i, j) = table->data(ids[i], j);
    }
    return record(std::move(out), {table}, [ids](Variable* self) {
        const Var& t = self->parents[0];
        if (!t->requires_grad) return;
        // Scatter-add straight into the table's grad. Going through
        // accumulate() would allocate a dense [vocab, d] temp per backward
        // -- 154 MB for GPT-2's wte -- for a handful of touched rows.
        if (t->grad.rows() == 0) t->grad = Matrix(t->data.rows(), t->data.cols());
        for (size_t i = 0; i < ids.size(); ++i)
            for (size_t j = 0; j < t->grad.cols(); ++j) t->grad(ids[i], j) += self->grad(i, j);
    });
}

Var cross_entropy(const Var& logits, const std::vector<int>& targets) {
    const size_t R = logits->data.rows(), C = logits->data.cols();
    if (targets.size() != R) {
        throw std::runtime_error("cross_entropy: one target per row");
    }
    for (size_t i = 0; i < R; ++i) {
        if (targets[i] < 0 || static_cast<size_t>(targets[i]) >= C) {
            throw std::runtime_error("cross_entropy: target out of range");
        }
    }
    // Cache the softmax for the fused backward -- the same
    // (P - onehot)/N contract as softmax_cross_entropy_grad_kernel.
    // B2.2: softmax + nll on-device — the [R,vocab] logits never come
    // home and the host receives ONE float; host path as fallback.
    auto P = std::make_shared<Matrix>(R, C);
    Matrix out(1, 1);
    float lossv = 0.0f;
    if (device::devops::ce_fwd(logits->data, targets.data(), *P, lossv)) {
        out(0, 0) = lossv;
    } else {
        device::materialize(logits->data);
        *P = logits->data;
        P->apply_softmax();
        double nll = 0.0;
        for (size_t i = 0; i < R; ++i) {
            nll -= std::log(std::max((*P)(i, targets[i]), 1e-12f));
        }
        out(0, 0) = static_cast<float>(nll / R);
    }
    return record(std::move(out), {logits}, [P, targets](Variable* self) {
        const Var& l = self->parents[0];
        if (!l->requires_grad) return;
        const float g = self->grad(0, 0) / static_cast<float>(P->rows());
        // B2.2: (P - onehot) * g on-device; the gradient first touches
        // host at accumulate() (host-authoritative until B2.3).
        Matrix dl(P->rows(), P->cols());
        if (!device::devops::ce_bwd(*P, targets.data(), g, dl)) {
            device::materialize(*P);  // no-op unless P deferred on-device
            dl = *P;
            for (size_t i = 0; i < dl.rows(); ++i) dl(i, targets[i]) -= 1.0f;
            l->accumulate(dl * g);
            return;
        }
        l->accumulate(dl);
    });
}

Var mul_row(const Var& x, const Var& r) {
    if (r->data.rows() != 1 || r->data.cols() != x->data.cols()) {
        throw std::runtime_error("mul_row: r must be [1, cols(x)]");
    }
    device::materialize(x->data);
    device::materialize(r->data);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) *= r->data(0, j);
    return record(std::move(out), {x, r}, [](Variable* self) {
        const Var& x = self->parents[0];
        const Var& r = self->parents[1];
        if (x->requires_grad) {
            Matrix dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) *= r->data(0, j);
            x->accumulate(dx);
        }
        if (r->requires_grad) {
            Matrix dr(1, r->data.cols());
            for (size_t i = 0; i < self->grad.rows(); ++i)
                for (size_t j = 0; j < self->grad.cols(); ++j)
                    dr(0, j) += self->grad(i, j) * x->data(i, j);
            r->accumulate(dr);
        }
    });
}

Var silu(const Var& x) {
    device::materialize(x->data);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) {
            const float v = out(i, j);
            out(i, j) = v / (1.0f + std::exp(-v));
        }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // d/dx [x s(x)] = s(x) (1 + x (1 - s(x)))
        Matrix dx = self->grad;
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j) {
                const float v = x->data(i, j);
                const float s = 1.0f / (1.0f + std::exp(-v));
                dx(i, j) *= s * (1.0f + v * (1.0f - s));
            }
        x->accumulate(dx);
    });
}

Var rmsnorm(const Var& x, const Var& w) {
    const size_t R = x->data.rows(), C = x->data.cols();
    if (w->data.rows() != 1 || w->data.cols() != C) {
        throw std::runtime_error("rmsnorm: w must be [1, cols(x)]");
    }
    // RMS normalization: x / RMS(x) * w, where RMS = sqrt(mean(x^2))
    // No mean centering, no bias. Stores RMS inverse for backward.
    auto rms_inv = std::make_shared<std::vector<float>>(R);
    Matrix out(R, C);
    const float eps = 1e-5f;
    // B2.1a seam: kernel computes out + rms_inv (write-through caches).
    if (!device::devops::rmsnorm_fwd(x->data, w->data, eps, out, *rms_inv)) {
        for (size_t i = 0; i < R; ++i) {
            float rms_sq = 0.0f;
            for (size_t j = 0; j < C; ++j) rms_sq += x->data(i, j) * x->data(i, j);
            rms_sq /= static_cast<float>(C);
            (*rms_inv)[i] = 1.0f / std::sqrt(rms_sq + eps);
            for (size_t j = 0; j < C; ++j) out(i, j) = x->data(i, j) * (*rms_inv)[i] * w->data(0, j);
        }
    }
    return record(std::move(out), {x, w}, [rms_inv](Variable* self) {
        const Var& x = self->parents[0];
        const Var& w = self->parents[1];
        const size_t R = self->grad.rows(), C = self->grad.cols();
        // B2.1a seam: dw column sum + dx row formula in kernels.
        Matrix dw(1, C), dx(R, C);
        const bool on_dev = device::devops::rmsnorm_bwd(
            self->grad, x->data, *rms_inv, w->data, w->requires_grad, &dw,
            x->requires_grad, &dx);
        if (w->requires_grad) {
            if (!on_dev) {
                for (size_t i = 0; i < R; ++i)
                    for (size_t j = 0; j < C; ++j)
                        dw(0, j) += self->grad(i, j) * x->data(i, j) * (*rms_inv)[i];
            }
            w->accumulate(dw);
        }
        if (!x->requires_grad) return;
        // RMSNorm gradient: y_ij = x_ij * w_j * rms_inv_i
        // dL/dx_ik = rms_inv_i * [dY_ik * w_k - x_ik * rms_inv_i^2 * sum_j(dY_ij * w_j * x_ij) / n]
        if (!on_dev) {
            for (size_t i = 0; i < R; ++i) {
                float term = 0.0f;
                for (size_t j = 0; j < C; ++j) term += self->grad(i, j) * w->data(0, j) * x->data(i, j);
                const float ri2 = (*rms_inv)[i] * (*rms_inv)[i];
                const float n_inv = 1.0f / static_cast<float>(C);
                for (size_t j = 0; j < C; ++j)
                    dx(i, j) = (*rms_inv)[i] *
                               (self->grad(i, j) * w->data(0, j) - x->data(i, j) * ri2 * term * n_inv);
            }
        }
        x->accumulate(dx);
    });
}

Var apply_rope(const Var& qk, const std::vector<int>& pos, float theta_base, size_t head_dim) {
    const size_t T = qk->data.rows(), d3 = qk->data.cols();
    if (d3 % 3 != 0) {
        throw std::runtime_error("apply_rope: cols must be divisible by 3");
    }
    const size_t d = d3 / 3;
    if (head_dim % 2 != 0 || head_dim > d) {
        throw std::runtime_error("apply_rope: head_dim must be even and <= d");
    }
    device::materialize(qk->data);
    // Cache angles for backward (position and frequency basis)
    auto pos_cache = std::make_shared<std::vector<int>>(pos);
    Matrix out = qk->data;
    // Apply RoPE to q and k via complex rotations on adjacent dimension pairs
    // RoPE operates on pairs (x[2j], x[2j+1]) as complex number rotations
    for (size_t i = 0; i < T; ++i) {
        const float m = static_cast<float>(pos[i]);
        // Apply to q (cols 0..d) and k (cols d..2d); skip v (cols 2d..3d)
        for (size_t start = 0; start < 2 * d; start += d) {
            for (size_t dim = 0; dim < head_dim; dim += 2) {
                const float inv_freq =
                    1.0f / std::pow(theta_base, static_cast<float>(dim) / head_dim);
                const float theta = m * inv_freq;
                const float cos_t = std::cos(theta);
                const float sin_t = std::sin(theta);
                // Rotate (x[dim], x[dim+1]) pair
                const float x0 = out(i, start + dim);
                const float x1 = out(i, start + dim + 1);
                out(i, start + dim) = x0 * cos_t - x1 * sin_t;
                out(i, start + dim + 1) = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    return record(std::move(out), {qk}, [pos_cache, theta_base, head_dim, d3](Variable* self) {
        const Var& qk = self->parents[0];
        if (!qk->requires_grad) return;
        const size_t T = self->grad.rows(), d = d3 / 3;
        Matrix dqk = self->grad;
        // Backward: apply inverse rotation (negative theta)
        for (size_t i = 0; i < T; ++i) {
            const float m = static_cast<float>((*pos_cache)[i]);
            for (size_t start = 0; start < 2 * d; start += d) {
                for (size_t dim = 0; dim < head_dim; dim += 2) {
                    const float inv_freq =
                        1.0f / std::pow(theta_base, static_cast<float>(dim) / head_dim);
                    const float theta = -m * inv_freq;  // Negative for inverse
                    const float cos_t = std::cos(theta);
                    const float sin_t = std::sin(theta);
                    const float dy0 = dqk(i, start + dim);
                    const float dy1 = dqk(i, start + dim + 1);
                    dqk(i, start + dim) = dy0 * cos_t - dy1 * sin_t;
                    dqk(i, start + dim + 1) = dy0 * sin_t + dy1 * cos_t;
                }
            }
        }
        qk->accumulate(dqk);
    });
}

// Phase 3a: Kimi Linear attention (O(n*d²) vs O(n²*d) standard attention)
namespace {
Matrix rows_of(const Matrix& m, size_t r0, size_t r1) {
    Matrix out(r1 - r0, m.cols());
    for (size_t i = r0; i < r1; ++i)
        for (size_t j = 0; j < m.cols(); ++j) out(i - r0, j) = m(i, j);
    return out;
}
void rows_into(Matrix& dst, const Matrix& src, size_t r0) {
    for (size_t i = 0; i < src.rows(); ++i)
        for (size_t j = 0; j < src.cols(); ++j) dst(r0 + i, j) = src(i, j);
}
}  // namespace

Var kimi_attention(const Var& q, const Var& k, const Var& v, bool causal, size_t seq_len) {
    using kimi::KimiLinearAttention;

    // The class backward recomputes CAUSAL prefix sums; a non-causal
    // forward under grad would get silently wrong gradients. Fail loudly
    // until the full-sum backward exists.
    if (!causal && grad_enabled() && (q->requires_grad || k->requires_grad || v->requires_grad)) {
        throw std::runtime_error(
            "kimi_attention: non-causal backward not implemented; wrap in "
            "NoGrad for inference or use causal=true");
    }

    const size_t T = q->data.rows();
    const size_t sl = seq_len == 0 ? T : seq_len;
    if (T % sl != 0) throw std::runtime_error("kimi_attention: rows not a multiple of seq_len");
    const size_t B = T / sl;
    size_t head_dim = q->data.cols();
    device::materialize(q->data);
    device::materialize(k->data);
    device::materialize(v->data);
    KimiLinearAttention kimi(head_dim);

    // Stacked mini-batch: linear attention's causal prefix sums must
    // reset at sequence boundaries, and blocks are independent — so run
    // the verified kernel PER BLOCK and stitch. Same total work
    // (linear attention is O(T d^2); splitting is free).
    Matrix out(T, head_dim);
    for (size_t b = 0; b < B; ++b) {
        Matrix ob = kimi.forward(rows_of(q->data, b * sl, (b + 1) * sl),
                                 rows_of(k->data, b * sl, (b + 1) * sl),
                                 rows_of(v->data, b * sl, (b + 1) * sl), causal);
        rows_into(out, ob, b * sl);
    }

    return record(std::move(out), {q, k, v}, [kimi = std::move(kimi), sl, B](Variable* self) {
        const Var& q_var = self->parents[0];
        const Var& k_var = self->parents[1];
        const Var& v_var = self->parents[2];

        if (!q_var->requires_grad && !k_var->requires_grad && !v_var->requires_grad) {
            return;
        }

        Matrix gq(q_var->data.rows(), q_var->data.cols());
        Matrix gk(k_var->data.rows(), k_var->data.cols());
        Matrix gv(v_var->data.rows(), v_var->data.cols());
        for (size_t b = 0; b < B; ++b) {
            auto [bq, bk, bv] = kimi.backward(rows_of(self->grad, b * sl, (b + 1) * sl),
                                              rows_of(q_var->data, b * sl, (b + 1) * sl),
                                              rows_of(k_var->data, b * sl, (b + 1) * sl),
                                              rows_of(v_var->data, b * sl, (b + 1) * sl),
                                              rows_of(self->data, b * sl, (b + 1) * sl));
            rows_into(gq, bq, b * sl);
            rows_into(gk, bk, b * sl);
            rows_into(gv, bv, b * sl);
        }
        if (q_var->requires_grad) q_var->accumulate(gq);
        if (k_var->requires_grad) k_var->accumulate(gk);
        if (v_var->requires_grad) v_var->accumulate(gv);
    });
}

Var ssm_scan(const Var& u, const Var& A, const Var& B, const Var& C, const Var& D) {
    const size_t T = u->data.rows(), n = u->data.cols();
    if (A->data.rows() != n || A->data.cols() != n || B->data.rows() != n || B->data.cols() != 1 ||
        C->data.rows() != 1 || C->data.cols() != n || D->data.rows() != 1 || D->data.cols() != 1) {
        throw std::runtime_error("ssm_scan: shape mismatch");
    }
    device::materialize(u->data);
    device::materialize(A->data);
    device::materialize(B->data);
    device::materialize(C->data);
    device::materialize(D->data);

    // Forward scan; states are kept for the backward pass (O(T n) memory,
    // the standard BPTT trade).
    auto states = std::make_shared<Matrix>(T, n);
    Matrix y(T, n);
    std::vector<float> s(n, 0.0f), s_new(n);
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < n; ++i) {
            float v = 0;
            for (size_t j = 0; j < n; ++j) v += A->data(i, j) * s[j];
            v += B->data(i, 0) * u->data(t, i);
            s_new[i] = v;
        }
        s = s_new;
        for (size_t i = 0; i < n; ++i) {
            (*states)(t, i) = s[i];
            y(t, i) = C->data(0, i) * s[i] + D->data(0, 0) * u->data(t, i);
        }
    }

    return record(std::move(y), {u, A, B, C, D}, [states](Variable* self) {
        const Var& u = self->parents[0];
        const Var& A = self->parents[1];
        const Var& B = self->parents[2];
        const Var& C = self->parents[3];
        const Var& D = self->parents[4];
        const size_t T = u->data.rows(), n = u->data.cols();
        const Matrix& S = *states;
        const Matrix& dY = self->grad;

        Matrix du(T, n), dA(n, n), dB(n, 1), dC(1, n), dD(1, 1);
        // Reverse recurrence: dS_t = C .* dY_t + A^T dS_{t+1}.
        std::vector<float> ds_next(n, 0.0f), ds(n);
        for (size_t t = T; t-- > 0;) {
            for (size_t i = 0; i < n; ++i) {
                float v = C->data(0, i) * dY(t, i);
                for (size_t j = 0; j < n; ++j) v += A->data(j, i) * ds_next[j];  // A^T
                ds[i] = v;
            }
            for (size_t i = 0; i < n; ++i) {
                if (t > 0) {
                    for (size_t j = 0; j < n; ++j) dA(i, j) += ds[i] * S(t - 1, j);
                }
                dB(i, 0) += ds[i] * u->data(t, i);
                dC(0, i) += dY(t, i) * S(t, i);
                dD(0, 0) += dY(t, i) * u->data(t, i);
                du(t, i) = B->data(i, 0) * ds[i] + D->data(0, 0) * dY(t, i);
            }
            ds_next = ds;
        }
        if (u->requires_grad) u->accumulate(du);
        if (A->requires_grad) A->accumulate(dA);
        if (B->requires_grad) B->accumulate(dB);
        if (C->requires_grad) C->accumulate(dC);
        if (D->requires_grad) D->accumulate(dD);
    });
}

Var mul_col(const Var& x, const Var& c) {
    if (c->data.cols() != 1 || c->data.rows() != x->data.rows()) {
        throw std::runtime_error("mul_col: c must be [rows(x), 1]");
    }
    device::materialize(x->data);
    device::materialize(c->data);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) *= c->data(i, 0);
    return record(std::move(out), {x, c}, [](Variable* self) {
        const Var& x = self->parents[0];
        const Var& c = self->parents[1];
        if (x->requires_grad) {
            Matrix dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) *= c->data(i, 0);
            x->accumulate(dx);
        }
        if (c->requires_grad) {
            Matrix dc(c->data.rows(), 1);
            for (size_t i = 0; i < dc.rows(); ++i) {
                float s = 0;
                for (size_t j = 0; j < x->data.cols(); ++j) s += self->grad(i, j) * x->data(i, j);
                dc(i, 0) = s;
            }
            c->accumulate(dc);
        }
    });
}

Var rms_row(const Var& x, float eps) {
    device::materialize(x->data);
    const size_t R = x->data.rows(), C = x->data.cols();
    Matrix out(R, 1);
    for (size_t i = 0; i < R; ++i) {
        float ss = 0;
        for (size_t j = 0; j < C; ++j) ss += x->data(i, j) * x->data(i, j);
        out(i, 0) = std::sqrt(ss / static_cast<float>(C) + eps);
    }
    return record(std::move(out), {x}, [eps](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        const size_t R = x->data.rows(), C = x->data.cols();
        // d rms/d x_ij = x_ij / (C * rms_i)
        Matrix dx(R, C);
        for (size_t i = 0; i < R; ++i) {
            const float denom = static_cast<float>(C) * self->data(i, 0);
            for (size_t j = 0; j < C; ++j) dx(i, j) = self->grad(i, 0) * x->data(i, j) / denom;
        }
        x->accumulate(dx);
    });
}

Var sigmoid(const Var& x) {
    // B2.1a seam (highway gates route through here on the flex path).
    Matrix out(x->data.rows(), x->data.cols());
    if (!device::devops::sigmoid_fwd(x->data, out)) {
        out = x->data;
        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j)
                out(i, j) = 1.0f / (1.0f + std::exp(-out(i, j)));
    }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(self->grad.rows(), self->grad.cols());
        if (!device::devops::sigmoid_bwd(self->data, self->grad, dx)) {
            dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) {
                    const float s = self->data(i, j);
                    dx(i, j) *= s * (1.0f - s);
                }
        }
        x->accumulate(dx);
    });
}

Var add_scalar(const Var& x, float s) {
    device::materialize(x->data);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) += s;
    return record(std::move(out), {x}, [](Variable* self) {
        if (self->parents[0]->requires_grad) self->parents[0]->accumulate(self->grad);
    });
}

Var dropout(const Var& x, float p, unsigned long long seed) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::runtime_error("dropout: p must be in [0, 1)");
    }
    if (p == 0.0f) return x;
    const float keep = 1.0f - p, inv_keep = 1.0f / keep;
    // The mask is a pure function of (seed, element index); backward
    // replays the same generator instead of storing a mask matrix.
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    device::materialize(x->data);
    Matrix out = x->data;
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j)
            out(i, j) = (u(rng) < keep) ? out(i, j) * inv_keep : 0.0f;
    return record(std::move(out), {x}, [p, seed](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        const float keep = 1.0f - p, inv_keep = 1.0f / keep;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        Matrix dx = self->grad;
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j)
                dx(i, j) = (u(rng) < keep) ? dx(i, j) * inv_keep : 0.0f;
        x->accumulate(dx);
    });
}

Var relu(const Var& x) {
    device::materialize(x->data);
    Matrix out(x->data.rows(), x->data.cols());
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) = std::max(0.0f, x->data(i, j));
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j)
                dx(i, j) = x->data(i, j) > 0.0f ? self->grad(i, j) : 0.0f;
        x->accumulate(dx);
    });
}

namespace {
// The in-place mechanism: transform the buffer, then interpose on the
// node's backward so grad-in-terms-of-output becomes grad-in-terms-of-
// input before the original closure (if any) runs. Under NoGrad only
// the data transform happens.
Var inplace_unary(const Var& x, const std::function<void(Matrix&)>& fwd,
                  const std::function<void(const Matrix&, Matrix&)>& dydx_from_output) {
    // B2.1b: the transform mutates host storage in place — download any
    // deferred value first so host is the authority being transformed.
    device::materialize(x->data);
    fwd(x->data);
    if (grad_enabled() && x->requires_grad) {
        auto orig = std::move(x->backward_fn);
        Variable* self = x.get();
        x->backward_fn = [self, orig, dydx_from_output]() {
            dydx_from_output(self->data, self->grad);  // grad *= f'(y), in place
            if (orig) orig();
        };
    }
    return x;
}
}  // namespace

Var relu_(const Var& x) {
    return inplace_unary(
        x,
        [](Matrix& d) {
            for (size_t i = 0; i < d.rows(); ++i)
                for (size_t j = 0; j < d.cols(); ++j) d(i, j) = std::max(0.0f, d(i, j));
        },
        [](const Matrix& y, Matrix& g) {
            for (size_t i = 0; i < g.rows(); ++i)
                for (size_t j = 0; j < g.cols(); ++j)
                    if (y(i, j) <= 0.0f) g(i, j) = 0.0f;
        });
}

Var sigmoid_(const Var& x) {
    return inplace_unary(
        x,
        [](Matrix& d) {
            for (size_t i = 0; i < d.rows(); ++i)
                for (size_t j = 0; j < d.cols(); ++j) d(i, j) = 1.0f / (1.0f + std::exp(-d(i, j)));
        },
        [](const Matrix& y, Matrix& g) {
            for (size_t i = 0; i < g.rows(); ++i)
                for (size_t j = 0; j < g.cols(); ++j) g(i, j) *= y(i, j) * (1.0f - y(i, j));
        });
}

Var scale_(const Var& x, float s) {
    return inplace_unary(
        x,
        [s](Matrix& d) {
            for (size_t i = 0; i < d.rows(); ++i)
                for (size_t j = 0; j < d.cols(); ++j) d(i, j) *= s;
        },
        [s](const Matrix&, Matrix& g) {
            for (size_t i = 0; i < g.rows(); ++i)
                for (size_t j = 0; j < g.cols(); ++j) g(i, j) *= s;
        });
}

Var fused_attention(const Var& q, const Var& k, const Var& v, float scale, size_t seq_len,
                    bool causal) {
    const size_t T = q->data.rows();
    if (k->data.rows() != T || v->data.rows() != T || q->data.cols() != k->data.cols()) {
        throw std::runtime_error("fused_attention: shape mismatch");
    }
    const size_t sl = seq_len == 0 ? T : seq_len;
    if (T % sl != 0) throw std::runtime_error("fused_attention: rows not a multiple of seq_len");

    // GEMM, then scale+mask+softmax fused in-place: A starts as the raw
    // scores and ends as the attention weights. Masked entries are never
    // exponentiated — they are written as hard zeros, which is exactly
    // what the -1e9 additive mask produces after float32 underflow.
    auto A = std::make_shared<Matrix>(device::gemm(
        q->data, &q->dev, device::Trans::N, k->data, &k->dev, device::Trans::T));
    // B2.2: masked softmax on-device when available (under deferral the
    // [T,T] scores then never cross the bus inside a step); otherwise
    // download the (possibly deferred) gemm scores and run the host loop.
    if (!device::devops::attn_masked_softmax(*A, scale, sl, causal)) {
        device::materialize(*A);
        for (size_t i = 0; i < T; ++i) {
            const size_t b0 = (i / sl) * sl;
            const size_t lo = b0, hi = causal ? i + 1 : b0 + sl;  // visible: [lo, hi)
            float mx = -1e30f;
            for (size_t j = lo; j < hi; ++j) {
                (*A)(i, j) *= scale;
                mx = std::max(mx, (*A)(i, j));
            }
            float z = 0.0f;
            for (size_t j = lo; j < hi; ++j) {
                (*A)(i, j) = std::exp((*A)(i, j) - mx);
                z += (*A)(i, j);
            }
            for (size_t j = 0; j < T; ++j) {
                if (j < lo || j >= hi) {
                    (*A)(i, j) = 0.0f;
                } else {
                    (*A)(i, j) /= z;
                }
            }
        }
    }
    Matrix y = device::gemm(*A, nullptr, device::Trans::N,
                            v->data, &v->dev, device::Trans::N);

    return record(std::move(y), {q, k, v}, [A, scale, sl, causal](Variable* self) {
        const Var& q = self->parents[0];
        const Var& k = self->parents[1];
        const Var& v = self->parents[2];
        const size_t T = q->data.rows();
        if (v->requires_grad) {
            v->accumulate(device::gemm(*A, nullptr, device::Trans::T,
                                       self->grad, nullptr, device::Trans::N));
        }
        if (!q->requires_grad && !k->requires_grad) return;
        // dA = dY V^T; ds = A .* (dA - rowsum(dA .* A)) * scale, computed
        // in place on dA (masked entries have A == 0, so ds is 0 there and
        // no mask bookkeeping is needed).
        Matrix ds = device::gemm(self->grad, nullptr, device::Trans::N,
                                 v->data, &v->dev, device::Trans::T);
        // B2.2: shared masked-softmax backward on-device (masked entries
        // carry A == 0, so the full-row dot equals the visible dot).
        if (!device::devops::attn_softmax_bwd_inplace(ds, *A, scale)) {
            device::materialize(ds);   // B2.1b: host loop below
            device::materialize(*A);   // no-op unless A deferred on-device
            for (size_t i = 0; i < T; ++i) {
                const size_t b0 = (i / sl) * sl;
                const size_t hi = causal ? i + 1 : b0 + sl;
                float dot = 0.0f;
                for (size_t j = b0; j < hi; ++j) dot += ds(i, j) * (*A)(i, j);
                for (size_t j = 0; j < T; ++j) {
                    const bool vis = j >= b0 && j < hi;
                    ds(i, j) = vis ? scale * (*A)(i, j) * (ds(i, j) - dot) : 0.0f;
                }
            }
        }
        if (q->requires_grad)
            q->accumulate(device::gemm(ds, nullptr, device::Trans::N,
                                       k->data, &k->dev, device::Trans::N));
        if (k->requires_grad)
            k->accumulate(device::gemm(ds, nullptr, device::Trans::T,
                                       q->data, &q->dev, device::Trans::N));
    });
}

// Sliding-window causal attention with attention sinks (S1 baseline of
// the sparse phase, docs/SPARSE_ATTENTION.md): query i sees the first `sinks`
// positions of its block plus the last `window` positions up to i. The
// field's default control (Longformer/Mistral lineage; sinks per
// StreamingLLM). Cost intuition O(T·(w+s)·d); this reference
// implementation still materializes [T,T] weights — the honest fast
// kernel belongs to coalfire (docs/ECOSYSTEM.md §1.1 role 4).
//
// EQUIVALENCE PIN (tests/test_swa.cpp): window >= seq_len with sinks=0
// visits exactly fused_attention's causal range in exactly its order, so
// the outputs are BITWISE identical — the sparse path cannot silently
// diverge from the exact one.
Var swa_attention(const Var& q, const Var& k, const Var& v, float scale, size_t window,
                  size_t sinks, size_t seq_len) {
    const size_t T = q->data.rows();
    if (k->data.rows() != T || v->data.rows() != T || q->data.cols() != k->data.cols()) {
        throw std::runtime_error("swa_attention: shape mismatch");
    }
    if (window == 0) throw std::runtime_error("swa_attention: window must be >= 1");
    const size_t sl = seq_len == 0 ? T : seq_len;
    if (T % sl != 0) throw std::runtime_error("swa_attention: rows not a multiple of seq_len");

    // Visible set for query i, expressed as two disjoint ranges inside
    // its block: [b0, sink_hi) then [win_lo, i+1).
    auto ranges = [sl, window, sinks](size_t i, size_t& b0, size_t& sink_hi, size_t& win_lo) {
        b0 = (i / sl) * sl;
        const size_t ii = i - b0;
        win_lo = b0 + (ii + 1 > window ? ii + 1 - window : 0);
        sink_hi = std::min(b0 + std::min(sinks, ii + 1), win_lo);
    };

    auto A = std::make_shared<Matrix>(device::gemm(
        q->data, &q->dev, device::Trans::N, k->data, &k->dev, device::Trans::T));
    // B2.2: sparse masked softmax on-device when available; host loop
    // as the fallback (and the reference semantics).
    if (!device::devops::swa_masked_softmax(*A, scale, sl, window, sinks)) {
        device::materialize(*A);  // B2.1b: host mask/softmax loop below
        for (size_t i = 0; i < T; ++i) {
            size_t b0, sink_hi, win_lo;
            ranges(i, b0, sink_hi, win_lo);
            float mx = -1e30f;
            for (size_t j = b0; j < sink_hi; ++j) {
                (*A)(i, j) *= scale;
                mx = std::max(mx, (*A)(i, j));
            }
            for (size_t j = win_lo; j <= i; ++j) {
                (*A)(i, j) *= scale;
                mx = std::max(mx, (*A)(i, j));
            }
            float z = 0.0f;
            for (size_t j = b0; j < sink_hi; ++j) {
                (*A)(i, j) = std::exp((*A)(i, j) - mx);
                z += (*A)(i, j);
            }
            for (size_t j = win_lo; j <= i; ++j) {
                (*A)(i, j) = std::exp((*A)(i, j) - mx);
                z += (*A)(i, j);
            }
            for (size_t j = 0; j < T; ++j) {
                const bool vis = (j >= b0 && j < sink_hi) || (j >= win_lo && j <= i);
                (*A)(i, j) = vis ? (*A)(i, j) / z : 0.0f;
            }
        }
    }
    Matrix y = device::gemm(*A, nullptr, device::Trans::N,
                            v->data, &v->dev, device::Trans::N);

    return record(std::move(y), {q, k, v}, [A, scale, sl, window, sinks](Variable* self) {
        const Var& q = self->parents[0];
        const Var& k = self->parents[1];
        const Var& v = self->parents[2];
        const size_t T = q->data.rows();
        if (v->requires_grad) {
            v->accumulate(device::gemm(*A, nullptr, device::Trans::T,
                                       self->grad, nullptr, device::Trans::N));
        }
        if (!q->requires_grad && !k->requires_grad) return;
        // Same softmax backward as fused_attention: masked entries carry
        // A == 0, so ds vanishes there with no mask bookkeeping.
        Matrix ds = device::gemm(self->grad, nullptr, device::Trans::N,
                                 v->data, &v->dev, device::Trans::T);
        // B2.2: the same shared backward kernel as fused_attention.
        if (!device::devops::attn_softmax_bwd_inplace(ds, *A, scale)) {
            device::materialize(ds);   // B2.1b: host loop below
            device::materialize(*A);   // no-op unless A deferred on-device
            for (size_t i = 0; i < T; ++i) {
                const size_t b0 = (i / sl) * sl;
                float dot = 0.0f;
                for (size_t j = b0; j <= i; ++j) dot += ds(i, j) * (*A)(i, j);
                for (size_t j = 0; j < T; ++j) {
                    ds(i, j) = (*A)(i, j) != 0.0f ? scale * (*A)(i, j) * (ds(i, j) - dot) : 0.0f;
                }
            }
        }
        if (q->requires_grad)
            q->accumulate(device::gemm(ds, nullptr, device::Trans::N,
                                       k->data, &k->dev, device::Trans::N));
        if (k->requires_grad)
            k->accumulate(device::gemm(ds, nullptr, device::Trans::T,
                                       q->data, &q->dev, device::Trans::N));
    });
}

Matrix attention_mask(size_t rows, size_t seq_len, bool causal) {
    if (seq_len == 0) seq_len = rows;
    if (rows % seq_len != 0) {
        throw std::runtime_error("attention_mask: rows not a multiple of seq_len");
    }
    Matrix m(rows, rows);
    for (size_t i = 0; i < rows; ++i) {
        const size_t blk = i / seq_len;
        for (size_t j = 0; j < rows; ++j) {
            const bool same_block = j / seq_len == blk;
            const bool visible = same_block && (!causal || j <= i);
            if (!visible) m(i, j) = -1e9f;
        }
    }
    return m;
}

float clip_grad_norm(const std::vector<Var>& params, float max_norm) {
    double sq = 0.0;
    for (const auto& p : params) {
        const Matrix& g = p->grad;
        if (g.rows() == 0) continue;  // never accumulated
        for (size_t i = 0; i < g.rows(); ++i)
            for (size_t j = 0; j < g.cols(); ++j) sq += static_cast<double>(g(i, j)) * g(i, j);
    }
    const float total = static_cast<float>(std::sqrt(sq));
    if (total > max_norm && total > 0.0f) {
        const float scale = max_norm / total;
        for (const auto& p : params) {
            Matrix& g = p->grad;
            for (size_t i = 0; i < g.rows(); ++i)
                for (size_t j = 0; j < g.cols(); ++j) g(i, j) *= scale;
        }
    }
    return total;
}

}  // namespace ops
}  // namespace microtorch
