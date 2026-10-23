#include <cstdlib>
#include "microtorch/ops.hpp"
#include "microtorch/device.hpp"
#include "microtorch/kimi_linear.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

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

// Forward-cached matrix (attention weights A, CE softmax P) with a
// self-discarding deleter: under NoGrad (or no-grad inputs) record()
// drops the backward closure, so the cache's only owner dies MID-WINDOW
// — without the discard its deferred entry outlives the buffer and
// step_end()'s materialize_all segfaults into freed memory (T4 gdb
// stack, 30 Aug: step_end -> cudaMemcpyDtoH). Fourth enforcement point
// of the lifetime rule, closing the last ownership class.
std::shared_ptr<Matrix> cached(Matrix&& m) {
    return std::shared_ptr<Matrix>(new Matrix(std::move(m)), [](Matrix* p) {
        device::discard(*p);
        delete p;
    });
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
            a->accumulate(device::gemm(self->grad, nullptr, Trans::N, b->data, &b->dev, Trans::T));
        }
        if (b->requires_grad) {
            b->accumulate(device::gemm(a->data, &a->dev, Trans::T, self->grad, nullptr, Trans::N));
        }
    });
}

// Elementwise add/mul run on the device when an operand is device-fresh
// (accumulate()'s rule): the result then stays in the value cache instead
// of pulling both operands home. One IEEE add or product per element on
// either side, so bit for bit; host-fresh operands (scalars, host-built
// tensors) keep the host loop and its host-fresh result.
namespace {
bool device_side(const Matrix& a, const Matrix& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (device::host_stale(a) || device::host_stale(b));
}
}  // namespace

Var add(const Var& a, const Var& b) {
    if (device_side(a->data, b->data)) {
        Matrix out = device::out_matrix(a->data.rows(), a->data.cols());
        if (device::devops::add(a->data, b->data, out))
            return record(std::move(out), {a, b}, [](Variable* self) {
                if (self->parents[0]->requires_grad) self->parents[0]->accumulate(self->grad);
                if (self->parents[1]->requires_grad) self->parents[1]->accumulate(self->grad);
            });
    }
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
            device::materialize(self->grad);  // B2.3c: host-read below
            self->parents[1]->accumulate(self->grad * -1.0f);
        }
    });
}

Var mul(const Var& a, const Var& b) {
    if (device_side(a->data, b->data)) {
        Matrix out = device::out_matrix(a->data.rows(), a->data.cols());
        if (device::devops::mul(a->data, b->data, out))
            return record(std::move(out), {a, b}, [](Variable* self) {
                const Var& a = self->parents[0];
                const Var& b = self->parents[1];
                // The forward ran on the device, so the operands are (or
                // were) device-fresh: the products stay there too.
                if (a->requires_grad) {
                    Matrix da = device::out_matrix(a->data.rows(), a->data.cols());
                    if (!device::devops::mul(self->grad, b->data, da)) {
                        device::materialize(self->grad);
                        device::materialize(b->data);
                        da = self->grad.hadamard(b->data);
                    }
                    a->accumulate(std::move(da));
                }
                if (b->requires_grad) {
                    Matrix db = device::out_matrix(b->data.rows(), b->data.cols());
                    if (!device::devops::mul(self->grad, a->data, db)) {
                        device::materialize(self->grad);
                        device::materialize(a->data);
                        db = self->grad.hadamard(a->data);
                    }
                    b->accumulate(std::move(db));
                }
            });
    }
    device::materialize(a->data);
    device::materialize(b->data);
    return record(a->data.hadamard(b->data), {a, b}, [](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        device::materialize(self->grad);  // B2.3c: host-read below
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
            device::materialize(self->grad);                    // B2.3c: host-read below
            Matrix db(1, b->data.cols());                       // column-sum, the same
            for (size_t i = 0; i < self->grad.rows(); ++i)      // contract as
                for (size_t j = 0; j < self->grad.cols(); ++j)  // compute_bias_
                    db(0, j) += self->grad(i, j);               // gradients_kernel
            b->accumulate(std::move(db));
        }
    });
}

Var gelu(const Var& x) {
    // B2.1a seam: same tanh-GELU formula on device (cuda_ops.cu) or the
    // vendored apply_gelu on host; entry returns false -> CPU, unchanged.
    Matrix out = device::out_matrix(x->data.rows(), x->data.cols());
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
        Matrix dx = device::out_matrix(self->grad.rows(), self->grad.cols());
        if (!device::devops::gelu_bwd(x->data, self->grad, dx)) {
            device::materialize(self->grad);  // B2.3c: host-read below
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
        x->accumulate(std::move(dx));
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
            device::materialize(self->grad);  // B2.3c: host-read below
            for (size_t i = 0; i < S.rows(); ++i) {
                float dot = 0.0f;
                for (size_t j = 0; j < S.cols(); ++j) dot += dY(i, j) * S(i, j);
                for (size_t j = 0; j < S.cols(); ++j) {
                    dX(i, j) = S(i, j) * (dY(i, j) - dot);
                }
            }
        }
        x->accumulate(std::move(dX));
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
        device::materialize(self->grad);  // B2.3c: host-read below
        Matrix dx(x->data.rows(), x->data.cols(), self->grad(0, 0) / n);
        x->accumulate(std::move(dx));
    });
}

Var scale(const Var& x, float s) {
    device::materialize(x->data);
    return record(x->data * s, {x}, [s](Variable* self) {
        if (self->parents[0]->requires_grad) {
            device::materialize(self->grad);  // B2.3c: host-read below
            self->parents[0]->accumulate(self->grad * s);
        }
    });
}

Var transpose(const Var& x) {
    // Device seam (the tied head's E^T): an exact copy; false -> host.
    Matrix out = device::out_matrix(x->data.cols(), x->data.rows());
    if (!device::devops::transpose(x->data, out)) {
        device::materialize(x->data);
        out = x->data.transpose();
    }
    return record(std::move(out), {x}, [](Variable* self) {
        if (self->parents[0]->requires_grad) {
            Matrix g = device::out_matrix(self->grad.cols(), self->grad.rows());
            if (!device::devops::transpose(self->grad, g)) {
                device::materialize(self->grad);  // B2.3c: host-read below
                g = self->grad.transpose();
            }
            self->parents[0]->accumulate(std::move(g));
        }
    });
}

Var slice_cols(const Var& x, size_t j0, size_t j1) {
    if (j1 <= j0 || j1 > x->data.cols()) {
        throw std::runtime_error("slice_cols: bad range");
    }
    Matrix out = device::out_matrix(x->data.rows(), j1 - j0);
    // Device seam (cuda_ops.cu): an exact copy; false -> the host loop.
    if (!device::devops::slice_cols(x->data, j0, j1 - j0, out)) {
        device::materialize(x->data);
        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j) out(i, j) = x->data(i, j0 + j);
    }
    return record(std::move(out), {x}, [j0](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // Straight into the parent's grad columns: no full-width temporary
        // (a llama block slices 18 heads per layer).
        x->accumulate_cols(j0, self->grad);
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
    Matrix out = device::out_matrix(rows, cols);
    std::vector<const Matrix*> parts;
    parts.reserve(xs.size());
    for (const auto& x : xs) parts.push_back(&x->data);
    // Device seam (cuda_ops.cu): an exact copy; false -> the host loop.
    if (!device::devops::concat_cols(parts, out)) {
        for (const auto& x : xs) device::materialize(x->data);
        size_t off = 0;
        for (const auto& x : xs) {
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < x->data.cols(); ++j) out(i, off + j) = x->data(i, j);
            off += x->data.cols();
        }
    }
    return record(std::move(out), xs, [](Variable* self) {
        size_t off = 0;
        for (const auto& p : self->parents) {
            const size_t w = p->data.cols();
            if (p->requires_grad) {
                Matrix dp = device::out_matrix(p->data.rows(), w);
                if (!device::devops::slice_cols(self->grad, off, w, dp)) {
                    device::materialize(self->grad);  // B2.3c: host-read below
                    for (size_t i = 0; i < dp.rows(); ++i)
                        for (size_t j = 0; j < w; ++j) dp(i, j) = self->grad(i, off + j);
                }
                p->accumulate(std::move(dp));
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
    if (!device::devops::layernorm_fwd(x->data, gamma->data, beta->data, eps, out, *xhat, *rstd)) {
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
        const bool on_dev = device::devops::layernorm_bwd(dY, *xhat, *rstd, g->data, want_dgb, &dg,
                                                          &db, x->requires_grad, &dx);
        if (!on_dev) device::materialize(self->grad);  // B2.3c: host-read below

        if (want_dgb) {
            if (!on_dev) {
                for (size_t i = 0; i < R; ++i)
                    for (size_t j = 0; j < C; ++j) {
                        dg(0, j) += dY(i, j) * (*xhat)(i, j);
                        db(0, j) += dY(i, j);
                    }
            }
            if (g->requires_grad) g->accumulate(std::move(dg));
            if (b->requires_grad) b->accumulate(std::move(db));
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
        x->accumulate(std::move(dx));
    });
}

Var embedding(const Var& table, const std::vector<int>& ids) {
    const size_t d = table->data.cols();
    Matrix out = device::out_matrix(ids.size(), d);
    // Bounds are checked HOST-SIDE for both paths (a kernel cannot throw).
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= table->data.rows()) {
            throw std::runtime_error("embedding: id out of range");
        }
    }
    // B2.2: gather on-device (the forward's first activation is born
    // resident); host loop as fallback. Backward scatter-add stays host
    // until B2.3 (table grad is host-authoritative).
    if (!device::devops::embed_gather(table->data, ids.data(), ids.size(), out)) {
        for (size_t i = 0; i < ids.size(); ++i)
            for (size_t j = 0; j < d; ++j) out(i, j) = table->data(ids[i], j);
    }
    return record(std::move(out), {table}, [ids](Variable* self) {
        const Var& t = self->parents[0];
        if (!t->requires_grad) return;
        // Scatter-add straight into the table's grad. Going through
        // accumulate() would allocate a dense [vocab, d] temp per backward
        // -- 154 MB for GPT-2's wte -- for a handful of touched rows.
        // On the device when the grad already lives there (zero_grad wrote
        // it in a deferring window, or the tied head's gemm deferred it):
        // the same adds in the same order, without downloading both.
        if (t->grad.rows() != 0 && device::host_stale(t->grad) &&
            device::devops::embed_scatter_add(self->grad, ids, t->grad))
            return;
        device::materialize(self->grad);  // B2.3c: host-read below
        device::materialize(t->grad);     // tied weights: gemm may have deferred it
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
    auto P = cached(device::out_matrix(R, C));
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
        device::materialize(self->grad);  // B2.3c: host-read below
        const float g = self->grad(0, 0) / static_cast<float>(P->rows());
        // B2.2: (P - onehot) * g on-device; the gradient first touches
        // host at accumulate() (host-authoritative until B2.3).
        Matrix dl = device::out_matrix(P->rows(), P->cols());
        if (!device::devops::ce_bwd(*P, targets.data(), g, dl)) {
            device::materialize(*P);  // no-op unless P deferred on-device
            dl = *P;
            for (size_t i = 0; i < dl.rows(); ++i) dl(i, targets[i]) -= 1.0f;
            l->accumulate(dl * g);
            device::discard(*P);
            return;
        }
        l->accumulate(std::move(dl));  // rvalue: self-discarding
        device::discard(*P);           // P is never host-read; skip its step_end D2H
    });
}

Var policy_gradient_kl(const Var& logits, const std::vector<int>& targets, const std::vector<float>& w,
                       const std::vector<float>& k, const Matrix& ref_logp) {
    const size_t R = logits->data.rows(), C = logits->data.cols();
    if (targets.size() != R || w.size() != R || k.size() != R)
        throw std::runtime_error("policy_gradient_kl: one target, w and k per row");
    if (ref_logp.rows() != R || ref_logp.cols() != C)
        throw std::runtime_error("policy_gradient_kl: ref_logp must match logits");
    device::materialize(logits->data);
    // log-softmax in double per row, cached for the backward.
    auto LP = cached(Matrix(R, C));
    auto kl = std::make_shared<std::vector<double>>(R, 0.0);
    double total = 0.0;
    for (size_t i = 0; i < R; ++i) {
        if (w[i] == 0.0f && k[i] == 0.0f) continue;
        if (targets[i] < 0 || static_cast<size_t>(targets[i]) >= C)
            throw std::runtime_error("policy_gradient_kl: target out of range");
        double mx = -1e300;
        for (size_t j = 0; j < C; ++j) mx = std::max(mx, static_cast<double>(logits->data(i, j)));
        double z = 0.0;
        for (size_t j = 0; j < C; ++j) z += std::exp(logits->data(i, j) - mx);
        const double lz = mx + std::log(z);
        double kli = 0.0;
        for (size_t j = 0; j < C; ++j) {
            const double lp = logits->data(i, j) - lz;
            (*LP)(i, j) = static_cast<float>(lp);
            kli += std::exp(lp) * (lp - ref_logp(i, j));
        }
        (*kl)[i] = kli;
        total += -static_cast<double>(w[i]) * ((*LP)(i, targets[i])) + static_cast<double>(k[i]) * kli;
    }
    Matrix out(1, 1);
    out(0, 0) = static_cast<float>(total);
    return record(std::move(out), {logits}, [LP, kl, targets, w, k, ref_logp](Variable* self) {
        const Var& l = self->parents[0];
        if (!l->requires_grad) return;
        device::materialize(self->grad);
        const double g = self->grad(0, 0);
        Matrix dl(LP->rows(), LP->cols());
        for (size_t i = 0; i < dl.rows(); ++i) {
            if (w[i] == 0.0f && k[i] == 0.0f) {
                for (size_t j = 0; j < dl.cols(); ++j) dl(i, j) = 0.0f;
                continue;
            }
            for (size_t j = 0; j < dl.cols(); ++j) {
                const double lp = (*LP)(i, j), p = std::exp(lp);
                double d = static_cast<double>(w[i]) * (p - (static_cast<int>(j) == targets[i] ? 1.0 : 0.0));
                d += static_cast<double>(k[i]) * p * (lp - ref_logp(i, j) - (*kl)[i]);
                dl(i, j) = static_cast<float>(g * d);
            }
        }
        l->accumulate(dl);
    });
}

Var dpo_weighted(const Var& logits, const std::vector<int>& targets, const std::vector<int>& row_pair,
                 const std::vector<int8_t>& row_sign, const std::vector<double>& ref_margin,
                 const std::vector<float>& pair_w, float beta) {
    const size_t R = logits->data.rows(), C = logits->data.cols(), P = pair_w.size();
    if (targets.size() != R || row_pair.size() != R || row_sign.size() != R || ref_margin.size() != P)
        throw std::runtime_error("dpo_weighted: one target, pair and sign per row; one ref_margin per pair");
    double W = 0.0;
    for (float w : pair_w) {
        if (w < 0.0f) throw std::runtime_error("dpo_weighted: negative weight");
        W += w;
    }
    if (W <= 0.0) throw std::runtime_error("dpo_weighted: every pair has weight 0");
    device::materialize(logits->data);
    // Row log-softmax (double), cached for the backward; margins per pair.
    auto LP = cached(Matrix(R, C));
    auto margin = std::make_shared<std::vector<double>>(P, 0.0);
    for (size_t p = 0; p < P; ++p) (*margin)[p] = -ref_margin[p];
    for (size_t i = 0; i < R; ++i) {
        const int p = row_pair[i];
        if (p < 0) continue;
        if (static_cast<size_t>(p) >= P || (row_sign[i] != 1 && row_sign[i] != -1))
            throw std::runtime_error("dpo_weighted: bad pair index or sign");
        if (targets[i] < 0 || static_cast<size_t>(targets[i]) >= C)
            throw std::runtime_error("dpo_weighted: target out of range");
        double mx = -1e300, z = 0.0;
        for (size_t j = 0; j < C; ++j) mx = std::max(mx, static_cast<double>(logits->data(i, j)));
        for (size_t j = 0; j < C; ++j) z += std::exp(logits->data(i, j) - mx);
        const double lz = mx + std::log(z);
        for (size_t j = 0; j < C; ++j) (*LP)(i, j) = static_cast<float>(logits->data(i, j) - lz);
        (*margin)[p] += row_sign[i] * static_cast<double>((*LP)(i, targets[i]));
    }
    double loss = 0.0;
    for (size_t p = 0; p < P; ++p) {
        if (pair_w[p] == 0.0f) continue;
        const double x = -beta * (*margin)[p];
        loss += pair_w[p] * (x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)));  // softplus
    }
    Matrix out(1, 1);
    out(0, 0) = static_cast<float>(loss / W);
    return record(std::move(out), {logits},
                  [LP, margin, targets, row_pair, row_sign, pair_w, beta, W](Variable* self) {
                      const Var& l = self->parents[0];
                      if (!l->requires_grad) return;
                      device::materialize(self->grad);
                      const double g = self->grad(0, 0);
                      Matrix dl(LP->rows(), LP->cols());
                      for (size_t i = 0; i < dl.rows(); ++i) {
                          const int p = row_pair[i];
                          if (p < 0 || pair_w[p] == 0.0f) {
                              for (size_t j = 0; j < dl.cols(); ++j) dl(i, j) = 0.0f;
                              continue;
                          }
                          // dL/dmargin_p = -(beta w_p / W) sigmoid(-beta margin_p)
                          const double sig = 1.0 / (1.0 + std::exp(beta * (*margin)[p]));
                          const double dm = -beta * pair_w[p] / W * sig;
                          for (size_t j = 0; j < dl.cols(); ++j) {
                              const double pr = std::exp(static_cast<double>((*LP)(i, j)));
                              const double dlogp = (static_cast<int>(j) == targets[i] ? 1.0 : 0.0) - pr;
                              dl(i, j) = static_cast<float>(g * dm * row_sign[i] * dlogp);
                          }
                      }
                      l->accumulate(dl);
                  });
}

Var cross_entropy_masked(const Var& logits, const std::vector<int>& targets, const std::vector<uint8_t>& mask) {
    const size_t R = logits->data.rows(), C = logits->data.cols();
    if (targets.size() != R || mask.size() != R) {
        throw std::runtime_error("cross_entropy_masked: one target and one mask entry per row");
    }
    size_t kept = 0;
    for (uint8_t m : mask) {
        if (m > 1) throw std::runtime_error("cross_entropy_masked: mask entries are 0 or 1");
        kept += m;
    }
    if (kept == 0) throw std::runtime_error("cross_entropy_masked: every row is masked out");
    // All ones: the plain op itself, so the result is bit-identical on every
    // backend (its device path included), not merely close.
    if (kept == R) return cross_entropy(logits, targets);
    for (size_t i = 0; i < R; ++i) {
        if (mask[i] && (targets[i] < 0 || static_cast<size_t>(targets[i]) >= C)) {
            throw std::runtime_error("cross_entropy_masked: target out of range");
        }
    }
    auto P = cached(device::out_matrix(R, C));
    // Device path (24 Oct 2026): masked rows travel as target -1; softmax,
    // nll and the gradient stay on the device exactly as cross_entropy's,
    // so the [R, vocab] logits and gradient never cross the bus (the host
    // path below cost a 1 GiB download and upload per SFT step at batch 32,
    // T 512, 16k vocab: SFT ran at 3,490 tok/s against pretraining's 25,113).
    std::vector<int> st(R);
    for (size_t i = 0; i < R; ++i) st[i] = mask[i] ? targets[i] : -1;
    float lossv = 0.0f;
    if (device::devops::ce_fwd_masked(logits->data, st.data(), kept, *P, lossv)) {
        Matrix out(1, 1);
        out(0, 0) = lossv;
        return record(std::move(out), {logits}, [P, st, kept](Variable* self) {
            const Var& l = self->parents[0];
            if (!l->requires_grad) return;
            device::materialize(self->grad);
            const float g = self->grad(0, 0) / static_cast<float>(kept);
            Matrix dl = device::out_matrix(P->rows(), P->cols());
            if (!device::devops::ce_bwd_masked(*P, st.data(), g, dl)) {
                device::materialize(*P);
                dl = *P;
                for (size_t i = 0; i < dl.rows(); ++i) {
                    if (st[i] < 0) {
                        for (size_t j = 0; j < dl.cols(); ++j) dl(i, j) = 0.0f;
                    } else {
                        dl(i, st[i]) -= 1.0f;
                    }
                }
                l->accumulate(dl * g);
                device::discard(*P);
                return;
            }
            l->accumulate(std::move(dl));  // rvalue: self-discarding
            device::discard(*P);           // never host-read; skip its step_end D2H
        });
    }
    // Host path, in cross_entropy's own operation order restricted to kept
    // rows, so it equals cross_entropy on the kept rows bit for bit.
    device::materialize(logits->data);
    *P = logits->data;
    P->apply_softmax();
    double nll = 0.0;
    for (size_t i = 0; i < R; ++i) {
        if (mask[i]) nll -= std::log(std::max((*P)(i, targets[i]), 1e-12f));
    }
    Matrix out(1, 1);
    out(0, 0) = static_cast<float>(nll / kept);
    return record(std::move(out), {logits}, [P, targets, mask, kept](Variable* self) {
        const Var& l = self->parents[0];
        if (!l->requires_grad) return;
        device::materialize(self->grad);
        const float g = self->grad(0, 0) / static_cast<float>(kept);
        Matrix dl = *P;
        for (size_t i = 0; i < dl.rows(); ++i) {
            if (!mask[i]) {
                for (size_t j = 0; j < dl.cols(); ++j) dl(i, j) = 0.0f;
            } else {
                dl(i, targets[i]) -= 1.0f;
            }
        }
        l->accumulate(dl * g);
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
            device::materialize(self->grad);  // B2.3c: host-read below
            Matrix dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) *= r->data(0, j);
            x->accumulate(std::move(dx));
        }
        if (r->requires_grad) {
            device::materialize(self->grad);  // B2.3c: host-read below
            Matrix dr(1, r->data.cols());
            for (size_t i = 0; i < self->grad.rows(); ++i)
                for (size_t j = 0; j < self->grad.cols(); ++j)
                    dr(0, j) += self->grad(i, j) * x->data(i, j);
            r->accumulate(std::move(dr));
        }
    });
}

Var silu(const Var& x) {
    // Device seam (18 Oct 2026), as gelu: the same formula in a kernel,
    // the host loop otherwise. Host path unchanged, bit for bit.
    Matrix out = device::out_matrix(x->data.rows(), x->data.cols());
    if (!device::devops::silu_fwd(x->data, out)) {
        device::materialize(x->data);
        out = x->data;
        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j) {
                const float v = out(i, j);
                out(i, j) = v / (1.0f + std::exp(-v));
            }
    }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // d/dx [x s(x)] = s(x) (1 + x (1 - s(x)))
        Matrix dx = device::out_matrix(self->grad.rows(), self->grad.cols());
        if (!device::devops::silu_bwd(x->data, self->grad, dx)) {
            device::materialize(self->grad);  // B2.3c: host-read below
            dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) {
                    const float v = x->data(i, j);
                    const float s = 1.0f / (1.0f + std::exp(-v));
                    dx(i, j) *= s * (1.0f + v * (1.0f - s));
                }
        }
        x->accumulate(std::move(dx));
    });
}

Var swiglu(const Var& gate, const Var& up) {
    // The llama FFN's silu(gate) .* up as one device op (20 Oct 2026): the
    // [rows, d_ff] silu output, a quarter of a dense layer's kept
    // activations, is recomputed from gate in the backward rather than
    // stored. The kernels evaluate silu, its derivative and both products
    // exactly as k_silu_fwd/k_silu_bwd and k_mul do, so the gradients and
    // the step are bit for bit those of mul(silu(gate), up), which is what
    // runs whenever the device op does not (CPU, host-fresh operands).
    if (device_side(gate->data, up->data)) {
        Matrix out = device::out_matrix(gate->data.rows(), gate->data.cols());
        if (device::devops::swiglu_fwd(gate->data, up->data, out))
            return record(std::move(out), {gate, up}, [](Variable* self) {
                const Var& g = self->parents[0];
                const Var& u = self->parents[1];
                const size_t R = self->grad.rows(), C = self->grad.cols();
                Matrix dg = g->requires_grad ? device::out_matrix(R, C) : Matrix();
                Matrix du = u->requires_grad ? device::out_matrix(R, C) : Matrix();
                if (!device::devops::swiglu_bwd(g->data, u->data, self->grad, g->requires_grad ? &dg : nullptr,
                                                u->requires_grad ? &du : nullptr))
                    throw std::runtime_error("swiglu: device backward unavailable");
                if (g->requires_grad) g->accumulate(std::move(dg));
                if (u->requires_grad) u->accumulate(std::move(du));
            });
    }
    return mul(silu(gate), up);
}

Var rmsnorm(const Var& x, const Var& w) {
    const size_t R = x->data.rows(), C = x->data.cols();
    if (w->data.rows() != 1 || w->data.cols() != C) {
        throw std::runtime_error("rmsnorm: w must be [1, cols(x)]");
    }
    // RMS normalization: x / RMS(x) * w, where RMS = sqrt(mean(x^2))
    // No mean centering, no bias. Stores RMS inverse for backward, as an
    // [R, 1] matrix so the device path keeps it resident between forward
    // and backward (a host vector cost a D2H and an H2D, each a device sync).
    auto rms_inv = cached(device::out_matrix(R, 1));
    Matrix out = device::out_matrix(R, C);
    const float eps = kRmsNormEps;
    // B2.1a seam: kernel computes out + rms_inv.
    if (!device::devops::rmsnorm_fwd(x->data, w->data, eps, out, *rms_inv)) {
        for (size_t i = 0; i < R; ++i) {
            float rms_sq = 0.0f;
            for (size_t j = 0; j < C; ++j) rms_sq += x->data(i, j) * x->data(i, j);
            rms_sq /= static_cast<float>(C);
            (*rms_inv)(i, 0) = 1.0f / std::sqrt(rms_sq + eps);
            for (size_t j = 0; j < C; ++j)
                out(i, j) = x->data(i, j) * (*rms_inv)(i, 0) * w->data(0, j);
        }
    }
    return record(std::move(out), {x, w}, [rms_inv](Variable* self) {
        const Var& x = self->parents[0];
        const Var& w = self->parents[1];
        const size_t R = self->grad.rows(), C = self->grad.cols();
        // B2.1a seam: dw column sum + dx row formula in kernels.
        Matrix dw(1, C), dx = device::out_matrix(R, C);
        const bool on_dev = device::devops::rmsnorm_bwd(
            self->grad, x->data, *rms_inv, w->data, w->requires_grad, &dw, x->requires_grad, &dx);
        if (!on_dev) {
            device::materialize(self->grad);  // B2.3c: host-read below
            device::materialize(*rms_inv);
        }
        if (w->requires_grad) {
            if (!on_dev) {
                for (size_t i = 0; i < R; ++i)
                    for (size_t j = 0; j < C; ++j)
                        dw(0, j) += self->grad(i, j) * x->data(i, j) * (*rms_inv)(i, 0);
            }
            w->accumulate(std::move(dw));
        }
        if (!x->requires_grad) return;
        // RMSNorm gradient: y_ij = x_ij * w_j * rms_inv_i
        // dL/dx_ik = rms_inv_i * [dY_ik * w_k - x_ik * rms_inv_i^2 * sum_j(dY_ij * w_j * x_ij) / n]
        if (!on_dev) {
            for (size_t i = 0; i < R; ++i) {
                float term = 0.0f;
                for (size_t j = 0; j < C; ++j)
                    term += self->grad(i, j) * w->data(0, j) * x->data(i, j);
                const float ri2 = (*rms_inv)(i, 0) * (*rms_inv)(i, 0);
                const float n_inv = 1.0f / static_cast<float>(C);
                for (size_t j = 0; j < C; ++j)
                    dx(i, j) = (*rms_inv)(i, 0) * (self->grad(i, j) * w->data(0, j) -
                                                x->data(i, j) * ri2 * term * n_inv);
            }
        }
        x->accumulate(std::move(dx));
    });
}

Var apply_rope(const Var& qk, const std::vector<int>& pos, float theta_base, size_t head_dim,
               bool all_heads) {
    const size_t T = qk->data.rows(), d3 = qk->data.cols();
    if (d3 % 3 != 0) {
        throw std::runtime_error("apply_rope: cols must be divisible by 3");
    }
    const size_t d = d3 / 3;
    if (head_dim == 0 || head_dim % 2 != 0 || head_dim > d) {
        throw std::runtime_error("apply_rope: head_dim must be even and <= d");
    }
    if (all_heads && d % head_dim != 0) {
        throw std::runtime_error("apply_rope: d must be a multiple of head_dim");
    }
    if (pos.size() != T) throw std::runtime_error("apply_rope: one position per row");
    // all_heads: every head h rotates its own columns [h*head_dim,
    // (h+1)*head_dim) of q and of k with the same per-head frequency basis.
    // Legacy (all_heads=false): only head 0's columns, the behaviour every
    // llama-family run before arch.rope_heads was trained with.
    const size_t n_rot = all_heads ? d / head_dim : 1;
    auto pos_cache = std::make_shared<std::vector<int>>(pos);
    Matrix out = device::out_matrix(T, d3);
    // Device seam (cuda_ops.cu): the same rotation from a host-built
    // cos/sin table; false -> the host loop below.
    const bool on_dev = device::devops::rope_fwd(qk->data, pos, head_dim, n_rot, theta_base, out);
    if (!on_dev) {
        device::materialize(qk->data);
        out = qk->data;
    }
    // Adjacent pairs (x[2j], x[2j+1]) within a head, rotated as complex
    // numbers by pos * theta^(-2j/head_dim) (the GGUF/GPT-J interleaved
    // convention).
    for (size_t i = 0; i < (on_dev ? 0 : T); ++i) {
        const float m = static_cast<float>(pos[i]);
        for (size_t dim = 0; dim < head_dim; dim += 2) {
            const float inv_freq = 1.0f / std::pow(theta_base, static_cast<float>(dim) / head_dim);
            const float theta = m * inv_freq;
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            // q heads (cols 0..d) and k heads (cols d..2d); v untouched.
            for (size_t start = 0; start < 2 * d; start += d) {
                for (size_t h = 0; h < n_rot; ++h) {
                    const size_t c = start + h * head_dim + dim;
                    const float x0 = out(i, c);
                    const float x1 = out(i, c + 1);
                    out(i, c) = x0 * cos_t - x1 * sin_t;
                    out(i, c + 1) = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    }
    return record(std::move(out), {qk},
                  [pos_cache, theta_base, head_dim, d3, n_rot](Variable* self) {
                      const Var& qk = self->parents[0];
                      if (!qk->requires_grad) return;
                      const size_t T = self->grad.rows(), d = d3 / 3;
                      Matrix dqk = device::out_matrix(T, d3);
                      if (device::devops::rope_bwd(self->grad, *pos_cache, head_dim, n_rot, theta_base, dqk)) {
                          qk->accumulate(std::move(dqk));
                          return;
                      }
                      device::materialize(self->grad);  // B2.3c: host-read below
                      dqk = self->grad;
                      // Backward: the transpose of a rotation is the inverse rotation.
                      for (size_t i = 0; i < T; ++i) {
                          const float m = static_cast<float>((*pos_cache)[i]);
                          for (size_t dim = 0; dim < head_dim; dim += 2) {
                              const float inv_freq =
                                  1.0f / std::pow(theta_base, static_cast<float>(dim) / head_dim);
                              const float theta = -m * inv_freq;
                              const float cos_t = std::cos(theta);
                              const float sin_t = std::sin(theta);
                              for (size_t start = 0; start < 2 * d; start += d) {
                                  for (size_t h = 0; h < n_rot; ++h) {
                                      const size_t c = start + h * head_dim + dim;
                                      const float dy0 = dqk(i, c);
                                      const float dy1 = dqk(i, c + 1);
                                      dqk(i, c) = dy0 * cos_t - dy1 * sin_t;
                                      dqk(i, c + 1) = dy0 * sin_t + dy1 * cos_t;
                                  }
                              }
                          }
                      }
                      qk->accumulate(std::move(dqk));
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
        device::materialize(self->grad);  // B2.3c: host-read below
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
        if (q_var->requires_grad) q_var->accumulate(std::move(gq));
        if (k_var->requires_grad) k_var->accumulate(std::move(gk));
        if (v_var->requires_grad) v_var->accumulate(std::move(gv));
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
        device::materialize(self->grad);  // B2.3c: host-read below
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
        if (u->requires_grad) u->accumulate(std::move(du));
        if (A->requires_grad) A->accumulate(std::move(dA));
        if (B->requires_grad) B->accumulate(std::move(dB));
        if (C->requires_grad) C->accumulate(std::move(dC));
        if (D->requires_grad) D->accumulate(std::move(dD));
    });
}

Var mul_col(const Var& x, const Var& c) {
    if (c->data.cols() != 1 || c->data.rows() != x->data.rows()) {
        throw std::runtime_error("mul_col: c must be [rows(x), 1]");
    }
    // Device seam (the MoE gate product): the same per-element product, and
    // dc's per-row sum in the host's column order.
    Matrix out(x->data.rows(), x->data.cols());
    if (!device::devops::mul_col_fwd(x->data, c->data, out)) {
        device::materialize(x->data);
        device::materialize(c->data);
        out = x->data;
        for (size_t i = 0; i < out.rows(); ++i)
            for (size_t j = 0; j < out.cols(); ++j) out(i, j) *= c->data(i, 0);
    }
    return record(std::move(out), {x, c}, [](Variable* self) {
        const Var& x = self->parents[0];
        const Var& c = self->parents[1];
        if (x->requires_grad) {
            Matrix dx(self->grad.rows(), self->grad.cols());
            if (!device::devops::mul_col_dx(self->grad, c->data, dx)) {
                device::materialize(self->grad);  // B2.3c: host-read below
                device::materialize(c->data);
                dx = self->grad;
                for (size_t i = 0; i < dx.rows(); ++i)
                    for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) *= c->data(i, 0);
            }
            x->accumulate(std::move(dx));
        }
        if (c->requires_grad) {
            Matrix dc(c->data.rows(), 1);
            if (!device::devops::mul_col_dc(self->grad, x->data, dc)) {
                device::materialize(self->grad);  // B2.3c: host-read below
                device::materialize(x->data);
                for (size_t i = 0; i < dc.rows(); ++i) {
                    float s = 0;
                    for (size_t j = 0; j < x->data.cols(); ++j) s += self->grad(i, j) * x->data(i, j);
                    dc(i, 0) = s;
                }
            }
            c->accumulate(std::move(dc));
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
        device::materialize(self->grad);  // B2.3c: host-read below
        // d rms/d x_ij = x_ij / (C * rms_i)
        Matrix dx(R, C);
        for (size_t i = 0; i < R; ++i) {
            const float denom = static_cast<float>(C) * self->data(i, 0);
            for (size_t j = 0; j < C; ++j) dx(i, j) = self->grad(i, 0) * x->data(i, j) / denom;
        }
        x->accumulate(std::move(dx));
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
            device::materialize(self->grad);  // B2.3c: host-read below
            dx = self->grad;
            for (size_t i = 0; i < dx.rows(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) {
                    const float s = self->data(i, j);
                    dx(i, j) *= s * (1.0f - s);
                }
        }
        x->accumulate(std::move(dx));
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
        device::materialize(self->grad);  // B2.3c: host-read below (dropout)
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        Matrix dx = self->grad;
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j)
                dx(i, j) = (u(rng) < keep) ? dx(i, j) * inv_keep : 0.0f;
        x->accumulate(std::move(dx));
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
        device::materialize(self->grad);  // B2.3c: host-read below
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j)
                dx(i, j) = x->data(i, j) > 0.0f ? self->grad(i, j) : 0.0f;
        x->accumulate(std::move(dx));
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
            device::materialize(self->grad);           // B2.3c: host-read below
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

bool blocked_attention_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("MICROTORCH_ATTN_BLOCKED");
        return e && std::string(e) == "1";
    }();
    return on;
}

// Batched attention, one sequence block at a time (17 Oct 2026). The rows of
// q, k, v are B stacked sequences of length sl. The single-[T,T] path used
// to compute scores for every pair of rows across the whole batch and zero
// the cross-sequence ones, so memory and work grew as B^2: on an L4 a batch
// of 64 at T=256 needed a 16384^2 score matrix per head and was killed, and
// throughput FELL as the batch grew (experiments/chat_sizing/). Per block
// the visible scores, softmax and products are the same arithmetic (the
// masked entries were exact zeros), at B*sl^2 cost. Backward per block:
// dV = A^T dY, dA = dY V^T, ds = scale * A .* (dA - rowsum(dA .* A)),
// dQ = ds K, dK = ds^T Q.
// OPT-IN (MICROTORCH_ATTN_BLOCKED=1): it equals the per-sequence path bit for
// bit (tests/test_attention_blocked.cpp), but against the stacked [T,T] path
// GEMMs over smaller shapes sum in a different order: 17/100 losses of the
// banked Atlas bridge moved by at most 1e-7 relative. The default stays
// bit-identical to every banked receipt; mtstudio records the choice in the
// model event (attention_impl).
static Matrix block_rows(const Matrix& m, size_t r0, size_t n) {
    Matrix out(n, m.cols());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m.cols(); ++j) out(i, j) = m(r0 + i, j);
    return out;
}

Var attention_heads(const Var& qkv, size_t heads, float scale, size_t seq_len, bool causal) {
    const size_t T = qkv->data.rows(), d3 = qkv->data.cols();
    if (heads == 0 || d3 % (3 * heads) != 0) throw std::runtime_error("attention_heads: bad head split");
    const size_t d = d3 / 3, dk = d / heads, sl = seq_len == 0 ? T : seq_len;
    // All heads in one device op (20 Oct 2026) where the per-head path would
    // take the blocked device kernels: the same kernels with head strides,
    // reading q, k, v straight out of qkv. Per-head slices, their column
    // accumulates and the head concat go, as do ~13 launches per head.
    if (sl < T && T % sl == 0 && blocked_attention_enabled() && device::device_ops_enabled() &&
        device::get() == device::Device::CUDA) {
        auto Ab = cached(device::out_matrix(heads * T, sl));
        Matrix y = device::out_matrix(T, d);
        if (device::devops::attn_heads_fwd(qkv->data, heads, scale, sl, causal, *Ab, y)) {
            return record(std::move(y), {qkv}, [Ab, heads, scale, sl, causal](Variable* self) {
                const Var& qkv = self->parents[0];
                if (!qkv->requires_grad) return;
                Matrix g = device::out_matrix(qkv->data.rows(), qkv->data.cols());
                if (!device::devops::attn_heads_bwd(qkv->data, heads, *Ab, self->grad, scale, sl, causal, g))
                    throw std::runtime_error("attention_heads: device backward unavailable");
                qkv->accumulate(std::move(g));
            });
        }
    }
    // Per head: slice q, k, v, attend, concatenate. The slices are read by
    // no backward, so their device copies go once the heads are sliced
    // (device::release; no-op unless deferring).
    Var q = slice_cols(qkv, 0, d), k = slice_cols(qkv, d, 2 * d), v = slice_cols(qkv, 2 * d, 3 * d);
    std::vector<Var> hs;
    hs.reserve(heads);
    for (size_t h = 0; h < heads; ++h) {
        Var qh = slice_cols(q, h * dk, (h + 1) * dk);
        Var kh = slice_cols(k, h * dk, (h + 1) * dk);
        Var vh = slice_cols(v, h * dk, (h + 1) * dk);
        hs.push_back(fused_attention(qh, kh, vh, scale, seq_len, causal));
    }
    for (const Var* t : {&q, &k, &v}) device::release((*t)->data);
    // The attention backward reads q, k, v and its weights, never its output.
    Var out = concat_cols(hs);
    for (const Var& hv : hs) device::release(hv->data);
    return out;
}

Var fused_attention_blocked(const Var& q, const Var& k, const Var& v, float scale, size_t sl, bool causal) {
    const size_t T = q->data.rows(), nb = T / sl;
    // Device path (18 Oct 2026): the same formulas in kernels, with A kept
    // block-local [T, sl] and resident between forward and backward. The
    // host loops below are the reference and the fallback.
    if (device::device_ops_enabled() && device::get() == device::Device::CUDA) {
        auto Ab = cached(device::out_matrix(T, sl));
        Matrix yd = device::out_matrix(T, v->data.cols());
        if (device::devops::attn_blocked_fwd(q->data, k->data, v->data, scale, sl, causal, *Ab, yd)) {
            return record(std::move(yd), {q, k, v}, [Ab, scale, sl, causal](Variable* self) {
                const Var& q = self->parents[0];
                const Var& k = self->parents[1];
                const Var& v = self->parents[2];
                Matrix dq = device::out_matrix(q->data.rows(), q->data.cols()),
                       dkm = device::out_matrix(k->data.rows(), k->data.cols()),
                       dvm = device::out_matrix(v->data.rows(), v->data.cols());
                if (!device::devops::attn_blocked_bwd(q->data, k->data, v->data, *Ab, self->grad, scale, sl, causal,
                                                      dq, dkm, dvm))
                    throw std::runtime_error("fused_attention_blocked: device backward unavailable");
                if (v->requires_grad) v->accumulate(std::move(dvm));
                else device::discard(dvm);
                if (q->requires_grad) q->accumulate(std::move(dq));
                else device::discard(dq);
                if (k->requires_grad) k->accumulate(std::move(dkm));
                else device::discard(dkm);
            });
        }
    }
    device::materialize(q->data);
    device::materialize(k->data);
    device::materialize(v->data);
    auto As = std::make_shared<std::vector<Matrix>>();
    As->reserve(nb);
    Matrix y(T, v->data.cols());
    for (size_t b = 0; b < nb; ++b) {
        const Matrix qb = block_rows(q->data, b * sl, sl), kb = block_rows(k->data, b * sl, sl),
                     vb = block_rows(v->data, b * sl, sl);
        Matrix A = device::gemm(qb, nullptr, device::Trans::N, kb, nullptr, device::Trans::T);
        device::materialize(A);
        for (size_t i = 0; i < sl; ++i) {
            const size_t hi = causal ? i + 1 : sl;
            float mx = -1e30f;
            for (size_t j = 0; j < hi; ++j) {
                A(i, j) *= scale;
                mx = std::max(mx, A(i, j));
            }
            float z = 0.0f;
            for (size_t j = 0; j < hi; ++j) {
                A(i, j) = std::exp(A(i, j) - mx);
                z += A(i, j);
            }
            for (size_t j = 0; j < sl; ++j) A(i, j) = j < hi ? A(i, j) / z : 0.0f;
        }
        Matrix yb = device::gemm(A, nullptr, device::Trans::N, vb, nullptr, device::Trans::N);
        device::materialize(yb);
        for (size_t i = 0; i < sl; ++i)
            for (size_t j = 0; j < yb.cols(); ++j) y(b * sl + i, j) = yb(i, j);
        As->push_back(std::move(A));
    }
    return record(std::move(y), {q, k, v}, [As, scale, sl, causal](Variable* self) {
        const Var& q = self->parents[0];
        const Var& k = self->parents[1];
        const Var& v = self->parents[2];
        device::materialize(self->grad);
        const size_t nb = As->size(), dk = q->data.cols(), dv = v->data.cols();
        Matrix dq(q->data.rows(), dk), dkm(k->data.rows(), dk), dvm(v->data.rows(), dv);
        for (size_t b = 0; b < nb; ++b) {
            const Matrix& A = (*As)[b];
            const Matrix gy = block_rows(self->grad, b * sl, sl), qb = block_rows(q->data, b * sl, sl),
                         kb = block_rows(k->data, b * sl, sl), vb = block_rows(v->data, b * sl, sl);
            if (v->requires_grad) {
                Matrix g = device::gemm(A, nullptr, device::Trans::T, gy, nullptr, device::Trans::N);
                device::materialize(g);
                for (size_t i = 0; i < sl; ++i)
                    for (size_t j = 0; j < dv; ++j) dvm(b * sl + i, j) = g(i, j);
            }
            if (!q->requires_grad && !k->requires_grad) continue;
            Matrix ds = device::gemm(gy, nullptr, device::Trans::N, vb, nullptr, device::Trans::T);
            device::materialize(ds);
            for (size_t i = 0; i < sl; ++i) {
                const size_t hi = causal ? i + 1 : sl;
                float dot = 0.0f;
                for (size_t j = 0; j < hi; ++j) dot += ds(i, j) * A(i, j);
                for (size_t j = 0; j < sl; ++j) ds(i, j) = j < hi ? scale * A(i, j) * (ds(i, j) - dot) : 0.0f;
            }
            if (q->requires_grad) {
                Matrix g = device::gemm(ds, nullptr, device::Trans::N, kb, nullptr, device::Trans::N);
                device::materialize(g);
                for (size_t i = 0; i < sl; ++i)
                    for (size_t j = 0; j < dk; ++j) dq(b * sl + i, j) = g(i, j);
            }
            if (k->requires_grad) {
                Matrix g = device::gemm(ds, nullptr, device::Trans::T, qb, nullptr, device::Trans::N);
                device::materialize(g);
                for (size_t i = 0; i < sl; ++i)
                    for (size_t j = 0; j < dk; ++j) dkm(b * sl + i, j) = g(i, j);
            }
        }
        if (v->requires_grad) v->accumulate(dvm);
        if (q->requires_grad) q->accumulate(dq);
        if (k->requires_grad) k->accumulate(dkm);
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
    if (sl < T && blocked_attention_enabled()) return fused_attention_blocked(q, k, v, scale, sl, causal);

    // GEMM, then scale+mask+softmax fused in-place: A starts as the raw
    // scores and ends as the attention weights. Masked entries are never
    // exponentiated — they are written as hard zeros, which is exactly
    // what the -1e9 additive mask produces after float32 underflow.
    auto A = cached(
        device::gemm(q->data, &q->dev, device::Trans::N, k->data, &k->dev, device::Trans::T));
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
    Matrix y = device::gemm(*A, nullptr, device::Trans::N, v->data, &v->dev, device::Trans::N);

    return record(std::move(y), {q, k, v}, [A, scale, sl, causal](Variable* self) {
        const Var& q = self->parents[0];
        const Var& k = self->parents[1];
        const Var& v = self->parents[2];
        const size_t T = q->data.rows();
        if (v->requires_grad) {
            v->accumulate(
                device::gemm(*A, nullptr, device::Trans::T, self->grad, nullptr, device::Trans::N));
        }
        if (!q->requires_grad && !k->requires_grad) {
            device::discard(*A);
            return;
        }
        // dA = dY V^T; ds = A .* (dA - rowsum(dA .* A)) * scale, computed
        // in place on dA (masked entries have A == 0, so ds is 0 there and
        // no mask bookkeeping is needed).
        Matrix ds =
            device::gemm(self->grad, nullptr, device::Trans::N, v->data, &v->dev, device::Trans::T);
        // B2.2: shared masked-softmax backward on-device (masked entries
        // carry A == 0, so the full-row dot equals the visible dot).
        if (!device::devops::attn_softmax_bwd_inplace(ds, *A, scale)) {
            device::materialize(ds);  // B2.1b: host loop below
            device::materialize(*A);  // no-op unless A deferred on-device
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
            q->accumulate(
                device::gemm(ds, nullptr, device::Trans::N, k->data, &k->dev, device::Trans::N));
        if (k->requires_grad)
            k->accumulate(
                device::gemm(ds, nullptr, device::Trans::T, q->data, &q->dev, device::Trans::N));
        // ds dies HERE while its deferred entry would still be stale;
        // A is never host-read again. Discard both or step_end() D2Hs
        // into freed memory (the leg-4 heap-corruption bug, 30 Aug).
        device::discard(ds);
        device::discard(*A);
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

    auto A = cached(
        device::gemm(q->data, &q->dev, device::Trans::N, k->data, &k->dev, device::Trans::T));
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
    Matrix y = device::gemm(*A, nullptr, device::Trans::N, v->data, &v->dev, device::Trans::N);

    return record(std::move(y), {q, k, v}, [A, scale, sl, window, sinks](Variable* self) {
        const Var& q = self->parents[0];
        const Var& k = self->parents[1];
        const Var& v = self->parents[2];
        const size_t T = q->data.rows();
        if (v->requires_grad) {
            v->accumulate(
                device::gemm(*A, nullptr, device::Trans::T, self->grad, nullptr, device::Trans::N));
        }
        if (!q->requires_grad && !k->requires_grad) {
            device::discard(*A);
            return;
        }
        // Same softmax backward as fused_attention: masked entries carry
        // A == 0, so ds vanishes there with no mask bookkeeping.
        Matrix ds =
            device::gemm(self->grad, nullptr, device::Trans::N, v->data, &v->dev, device::Trans::T);
        // B2.2: the same shared backward kernel as fused_attention.
        if (!device::devops::attn_softmax_bwd_inplace(ds, *A, scale)) {
            device::materialize(ds);  // B2.1b: host loop below
            device::materialize(*A);  // no-op unless A deferred on-device
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
            q->accumulate(
                device::gemm(ds, nullptr, device::Trans::N, k->data, &k->dev, device::Trans::N));
        if (k->requires_grad)
            k->accumulate(
                device::gemm(ds, nullptr, device::Trans::T, q->data, &q->dev, device::Trans::N));
        // Same lifetime rule as fused_attention's backward: ds dies here.
        device::discard(ds);
        device::discard(*A);
    });
}

// ---- mixture-of-experts dispatch (see ops.hpp) ----

Var gather_rows(const Var& x, const std::vector<int>& idx) {
    const size_t R = x->data.rows(), C = x->data.cols();
    if (idx.empty()) throw std::runtime_error("gather_rows: empty index list");
    for (int i : idx)
        if (i < 0 || static_cast<size_t>(i) >= R) throw std::runtime_error("gather_rows: index out of range");
    // Device seam (cuda_ops.cu, the MoE dispatch): an exact copy, and the
    // backward's scatter adds each row's sources in the host loop's order.
    Matrix out(idx.size(), C);
    if (!device::devops::gather_rows(x->data, idx, C, out)) {
        device::materialize(x->data);
        for (size_t i = 0; i < idx.size(); ++i)
            for (size_t j = 0; j < C; ++j) out(i, j) = x->data(idx[i], j);
    }
    return record(std::move(out), {x}, [idx](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        // On the device the rows add straight into x's gradient: an MoE
        // layer gathers once per expert, and a dense [rows(x), cols] dx per
        // expert was most of its backward's host allocation. Same roundings
        // as dx-then-accumulate (each row's sum starts from 0.0f).
        if (x->grad.rows() == 0) {
            // First gradient: the kernel writes a fresh one whole (a vector
            // move keeps the buffer, so its value-cache entry moves too).
            Matrix g(x->data.rows(), x->data.cols());
            if (device::devops::scatter_rows_into(self->grad, idx, g, true)) {
                x->grad = std::move(g);
                return;
            }
        } else if (device::devops::scatter_rows_into(self->grad, idx, x->grad, false)) {
            return;
        }
        device::materialize(self->grad);  // host-read below
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < idx.size(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j) dx(idx[i], j) += self->grad(i, j);
        x->accumulate(std::move(dx));
    });
}

Var scatter_add_rows(const Var& x, const std::vector<int>& idx, size_t n_rows) {
    const size_t C = x->data.cols();
    if (idx.size() != x->data.rows()) throw std::runtime_error("scatter_add_rows: one index per row of x");
    for (int i : idx)
        if (i < 0 || static_cast<size_t>(i) >= n_rows)
            throw std::runtime_error("scatter_add_rows: index out of range");
    Matrix out(n_rows, C);
    // First write to a row copies, later ones add: a one-to-one scatter is
    // then an exact copy (0.0f + v would turn a -0.0 into +0.0). The device
    // seam keeps that rule and the ascending source order.
    if (!device::devops::scatter_rows(x->data, idx, n_rows, C, true, out)) {
        device::materialize(x->data);
        std::vector<char> touched(n_rows, 0);
        for (size_t i = 0; i < idx.size(); ++i) {
            const size_t r = static_cast<size_t>(idx[i]);
            if (touched[r]) {
                for (size_t j = 0; j < C; ++j) out(r, j) += x->data(i, j);
            } else {
                for (size_t j = 0; j < C; ++j) out(r, j) = x->data(i, j);
                touched[r] = 1;
            }
        }
    }
    return record(std::move(out), {x}, [idx](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(idx.size(), self->grad.cols());
        if (!device::devops::gather_rows(self->grad, idx, dx.cols(), dx)) {
            device::materialize(self->grad);  // host-read below
            for (size_t i = 0; i < idx.size(); ++i)
                for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) = self->grad(idx[i], j);
        }
        x->accumulate(std::move(dx));
    });
}

Var scatter_add_rows(const std::vector<Var>& xs, const std::vector<std::vector<int>>& idx, size_t n_rows) {
    if (xs.empty() || idx.size() != xs.size()) throw std::runtime_error("scatter_add_rows: one index list per input");
    const size_t C = xs[0]->data.cols();
    std::vector<int> all;  // the index lists concatenated, in input order
    std::vector<const Matrix*> parts;
    for (size_t e = 0; e < xs.size(); ++e) {
        if (xs[e]->data.cols() != C || idx[e].size() != xs[e]->data.rows())
            throw std::runtime_error("scatter_add_rows: one index per row of each input, equal widths");
        for (int i : idx[e])
            if (i < 0 || static_cast<size_t>(i) >= n_rows)
                throw std::runtime_error("scatter_add_rows: index out of range");
        all.insert(all.end(), idx[e].begin(), idx[e].end());
        parts.push_back(&xs[e]->data);
    }
    Matrix out(n_rows, C);
    // The single-input op's rule over the stacked inputs: first write
    // copies, later ones add, sources in input order.
    if (!device::devops::scatter_rows_multi(parts, all, n_rows, C, out)) {
        std::vector<char> touched(n_rows, 0);
        for (size_t e = 0; e < xs.size(); ++e) {
            device::materialize(xs[e]->data);
            const Matrix& x = xs[e]->data;
            for (size_t i = 0; i < idx[e].size(); ++i) {
                const size_t r = static_cast<size_t>(idx[e][i]);
                if (touched[r]) {
                    for (size_t j = 0; j < C; ++j) out(r, j) += x(i, j);
                } else {
                    for (size_t j = 0; j < C; ++j) out(r, j) = x(i, j);
                    touched[r] = 1;
                }
            }
        }
    }
    return record(std::move(out), xs, [idx](Variable* self) {
        for (size_t e = 0; e < self->parents.size(); ++e) {
            const Var& x = self->parents[e];
            if (!x->requires_grad) continue;
            Matrix dx(idx[e].size(), self->grad.cols());
            if (!device::devops::gather_rows(self->grad, idx[e], dx.cols(), dx)) {
                device::materialize(self->grad);  // host-read below
                for (size_t i = 0; i < idx[e].size(); ++i)
                    for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) = self->grad(idx[e][i], j);
            }
            x->accumulate(std::move(dx));
        }
    });
}

Var gather_elems(const Var& x, const std::vector<int>& rows, const std::vector<int>& cols, size_t out_cols) {
    const size_t n = rows.size();
    if (n == 0 || cols.size() != n || out_cols == 0 || n % out_cols != 0)
        throw std::runtime_error("gather_elems: rows/cols must be equal, non-empty, a multiple of out_cols");
    for (size_t i = 0; i < n; ++i)
        if (rows[i] < 0 || static_cast<size_t>(rows[i]) >= x->data.rows() || cols[i] < 0 ||
            static_cast<size_t>(cols[i]) >= x->data.cols())
            throw std::runtime_error("gather_elems: index out of range");
    // Device seam: the C = 1 row ops over flat element indices (the matrix
    // storage is row-major), so the backward keeps the host's add order.
    std::vector<int> flat(n);
    for (size_t i = 0; i < n; ++i) flat[i] = rows[i] * static_cast<int>(x->data.cols()) + cols[i];
    Matrix out(n / out_cols, out_cols);
    if (!device::devops::gather_rows(x->data, flat, 1, out)) {
        device::materialize(x->data);
        for (size_t i = 0; i < n; ++i) out(i / out_cols, i % out_cols) = x->data(rows[i], cols[i]);
    }
    return record(std::move(out), {x}, [rows, cols, out_cols, flat](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        Matrix dx(x->data.rows(), x->data.cols());
        if (!device::devops::scatter_rows(self->grad, flat, dx.rows() * dx.cols(), 1, false, dx)) {
            device::materialize(self->grad);  // host-read below
            for (size_t i = 0; i < rows.size(); ++i)
                dx(rows[i], cols[i]) += self->grad(i / out_cols, i % out_cols);
        }
        x->accumulate(std::move(dx));
    });
}

Var logsumexp_row(const Var& x) {
    const size_t R = x->data.rows(), C = x->data.cols();
    Matrix out(R, 1);
    // Device seam: one thread per row running this loop (the double sum's
    // exp/log are the device libm's, so agreement is to rounding).
    if (!device::devops::logsumexp_fwd(x->data, out)) {
        device::materialize(x->data);
        for (size_t i = 0; i < R; ++i) {
            float mx = x->data(i, 0);
            for (size_t j = 1; j < C; ++j) mx = std::max(mx, x->data(i, j));
            double s = 0.0;
            for (size_t j = 0; j < C; ++j) s += std::exp(static_cast<double>(x->data(i, j)) - mx);
            out(i, 0) = mx + static_cast<float>(std::log(s));
        }
    }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        const size_t R = x->data.rows(), C = x->data.cols();
        Matrix dx(R, C);
        if (!device::devops::logsumexp_bwd(x->data, self->data, self->grad, dx)) {
            device::materialize(self->grad);  // host-read below
            device::materialize(self->data);
            device::materialize(x->data);
            for (size_t i = 0; i < R; ++i) {
                const float lse = self->data(i, 0), g = self->grad(i, 0);
                for (size_t j = 0; j < C; ++j) dx(i, j) = g * std::exp(x->data(i, j) - lse);
            }
        }
        x->accumulate(std::move(dx));
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
    // Grads still on the device (the optimizer runs inside the step window:
    // mtstudio's dense path) reduce there, one double per parameter, and
    // are scaled there; host-fresh grads keep the host loops. A device
    // partial is summed in a different order from the host loop, so the
    // norm agrees to double rounding (in practice the same float).
    std::vector<const Matrix*> dev;
    for (const auto& p : params)
        if (p->grad.rows() != 0 && device::host_stale(p->grad)) dev.push_back(&p->grad);
    std::vector<double> part;
    if (!dev.empty() && !device::devops::sumsq(dev, part)) {
        for (const Matrix* g : dev) device::materialize(*g);
        dev.clear();
    }
    double sq = 0.0;
    size_t k = 0;
    for (const auto& p : params) {
        const Matrix& g = p->grad;
        if (g.rows() == 0) continue;  // never accumulated
        if (k < dev.size() && dev[k] == &g) {
            sq += part[k++];
            continue;
        }
        device::materialize(g);
        for (size_t i = 0; i < g.rows(); ++i)
            for (size_t j = 0; j < g.cols(); ++j) sq += static_cast<double>(g(i, j)) * g(i, j);
    }
    const float total = static_cast<float>(std::sqrt(sq));
    if (total > max_norm && total > 0.0f) {
        const float scale = max_norm / total;
        for (const auto& p : params) {
            Matrix& g = p->grad;
            if (g.rows() == 0) continue;
            if (device::host_stale(g) && device::devops::scale(g, scale, g)) continue;
            device::materialize(g);
            for (size_t i = 0; i < g.rows(); ++i)
                for (size_t j = 0; j < g.cols(); ++j) g(i, j) *= scale;
        }
    }
    return total;
}

// ---- runtime-defined ops' primitives (ops::expr) ----
// Host loops: the forward is one f32 operation per element, the backward
// reads the input (x) and the output (y) of each element.
namespace {

template <class F, class D>
Var unary_host(const Var& x, F f, D dydx) {
    device::materialize(x->data);
    Matrix out(x->data.rows(), x->data.cols());
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) = f(x->data(i, j));
    return record(std::move(out), {x}, [dydx](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        device::materialize(self->grad);  // B2.3c: host-read below
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j)
                dx(i, j) = self->grad(i, j) * dydx(x->data(i, j), self->data(i, j));
        x->accumulate(std::move(dx));
    });
}

void same_shape(const char* op, const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::runtime_error(std::string(op) + ": operands must have one shape");
}

// f(a, b) per element; da(a, b, y) and db(a, b, y) are the partial
// derivatives, each multiplied into the upstream gradient.
template <class F, class DA, class DB>
Var binary_host(const char* op, const Var& a, const Var& b, F f, DA da, DB db) {
    same_shape(op, a->data, b->data);
    device::materialize(a->data);
    device::materialize(b->data);
    Matrix out(a->data.rows(), a->data.cols());
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) out(i, j) = f(a->data(i, j), b->data(i, j));
    return record(std::move(out), {a, b}, [da, db](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        device::materialize(self->grad);  // B2.3c: host-read below
        const size_t R = self->data.rows(), C = self->data.cols();
        if (a->requires_grad) {
            Matrix g(R, C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j)
                    g(i, j) = self->grad(i, j) * da(a->data(i, j), b->data(i, j), self->data(i, j));
            a->accumulate(std::move(g));
        }
        if (b->requires_grad) {
            Matrix g(R, C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j)
                    g(i, j) = self->grad(i, j) * db(a->data(i, j), b->data(i, j), self->data(i, j));
            b->accumulate(std::move(g));
        }
    });
}

}  // namespace

Var exp(const Var& x) {
    return unary_host(x, [](float v) { return std::exp(v); }, [](float, float y) { return y; });
}

Var log(const Var& x) {
    return unary_host(x, [](float v) { return std::log(v); }, [](float v, float) { return 1.0f / v; });
}

Var sqrt(const Var& x) {
    return unary_host(x, [](float v) { return std::sqrt(v); }, [](float, float y) { return 0.5f / y; });
}

Var tanh(const Var& x) {
    return unary_host(x, [](float v) { return std::tanh(v); }, [](float, float y) { return 1.0f - y * y; });
}

Var erf(const Var& x) {
    return unary_host(x, [](float v) { return std::erf(v); },
                      [](float v, float) { return 1.1283791670955126f * std::exp(-v * v); });  // 2/sqrt(pi)
}

Var abs(const Var& x) {
    return unary_host(x, [](float v) { return std::fabs(v); },
                      [](float v, float) { return v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f); });
}

Var div(const Var& a, const Var& b) {
    return binary_host(
        "div", a, b, [](float p, float q) { return p / q; }, [](float, float q, float) { return 1.0f / q; },
        [](float, float q, float y) { return -y / q; });
}

Var pow(const Var& a, const Var& b) {
    return binary_host(
        "pow", a, b, [](float p, float q) { return std::pow(p, q); },
        [](float p, float q, float) { return q == 0.0f ? 0.0f : q * std::pow(p, q - 1.0f); },
        [](float p, float, float y) { return y == 0.0f ? 0.0f : y * std::log(p); });
}

Var maximum(const Var& a, const Var& b) {
    return binary_host(
        "maximum", a, b, [](float p, float q) { return p >= q ? p : q; },
        [](float p, float q, float) { return p >= q ? 1.0f : 0.0f; },
        [](float p, float q, float) { return p >= q ? 0.0f : 1.0f; });
}

Var minimum(const Var& a, const Var& b) {
    return binary_host(
        "minimum", a, b, [](float p, float q) { return p <= q ? p : q; },
        [](float p, float q, float) { return p <= q ? 1.0f : 0.0f; },
        [](float p, float q, float) { return p <= q ? 0.0f : 1.0f; });
}

Var where(const Var& c, const Var& a, const Var& b) {
    same_shape("where", c->data, a->data);
    same_shape("where", a->data, b->data);
    device::materialize(c->data);
    device::materialize(a->data);
    device::materialize(b->data);
    const size_t R = a->data.rows(), C = a->data.cols();
    Matrix out(R, C);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j) out(i, j) = c->data(i, j) > 0.0f ? a->data(i, j) : b->data(i, j);
    // The condition is a parent (it must outlive the closure) but takes no
    // gradient: selection is not differentiable.
    return record(std::move(out), {a, b, c}, [](Variable* self) {
        const Var& a = self->parents[0];
        const Var& b = self->parents[1];
        const Var& c = self->parents[2];
        device::materialize(self->grad);  // B2.3c: host-read below
        const size_t R = self->data.rows(), C = self->data.cols();
        for (int side = 0; side < 2; ++side) {
            const Var& p = side == 0 ? a : b;
            if (!p->requires_grad) continue;
            Matrix g(R, C);
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j)
                    if ((c->data(i, j) > 0.0f) == (side == 0)) g(i, j) = self->grad(i, j);
            p->accumulate(std::move(g));
        }
    });
}

Var rowsum(const Var& x) {
    device::materialize(x->data);
    const size_t R = x->data.rows(), C = x->data.cols();
    Matrix out(R, 1);
    for (size_t i = 0; i < R; ++i) {
        float s = 0.0f;
        for (size_t j = 0; j < C; ++j) s += x->data(i, j);
        out(i, 0) = s;
    }
    return record(std::move(out), {x}, [](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        device::materialize(self->grad);  // B2.3c: host-read below
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < dx.rows(); ++i)
            for (size_t j = 0; j < dx.cols(); ++j) dx(i, j) = self->grad(i, 0);
        x->accumulate(std::move(dx));
    });
}

Var rowmax(const Var& x) {
    device::materialize(x->data);
    const size_t R = x->data.rows(), C = x->data.cols();
    if (C == 0) throw std::runtime_error("rowmax: no columns");
    Matrix out(R, 1);
    std::vector<size_t> arg(R, 0);
    for (size_t i = 0; i < R; ++i) {
        float m = x->data(i, 0);
        for (size_t j = 1; j < C; ++j)
            if (x->data(i, j) > m) {
                m = x->data(i, j);
                arg[i] = j;
            }
        out(i, 0) = m;
    }
    return record(std::move(out), {x}, [arg](Variable* self) {
        const Var& x = self->parents[0];
        if (!x->requires_grad) return;
        device::materialize(self->grad);  // B2.3c: host-read below
        Matrix dx(x->data.rows(), x->data.cols());
        for (size_t i = 0; i < dx.rows(); ++i) dx(i, arg[i]) = self->grad(i, 0);
        x->accumulate(std::move(dx));
    });
}

Var broadcast_cols(const Var& c, size_t n) {
    if (c->data.cols() != 1) throw std::runtime_error("broadcast_cols: c must be [rows, 1]");
    device::materialize(c->data);
    const size_t R = c->data.rows();
    Matrix out(R, n);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < n; ++j) out(i, j) = c->data(i, 0);
    return record(std::move(out), {c}, [](Variable* self) {
        const Var& c = self->parents[0];
        if (!c->requires_grad) return;
        device::materialize(self->grad);  // B2.3c: host-read below
        Matrix dc(self->grad.rows(), 1);
        for (size_t i = 0; i < dc.rows(); ++i) {
            float s = 0.0f;
            for (size_t j = 0; j < self->grad.cols(); ++j) s += self->grad(i, j);
            dc(i, 0) = s;
        }
        c->accumulate(std::move(dc));
    });
}

}  // namespace ops
}  // namespace microtorch
