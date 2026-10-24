// ops::depth_attend: attention over depth as one tape node (see
// include/microtorch/depth_attend.hpp for the formulas and why).
#include "microtorch/depth_attend.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "microtorch/device_cache.hpp"

namespace microtorch {
namespace ops {

namespace {

// As ops.cpp's: an aux table that discards its value-cache entry when it
// dies (a deferred buffer must never outlive its host storage).
std::shared_ptr<Matrix> cached(Matrix&& m) {
    return std::shared_ptr<Matrix>(new Matrix(std::move(m)), [](Matrix* p) {
        device::discard(*p);
        delete p;
    });
}

}  // namespace

Var depth_attend(const std::vector<Var>& sources, const Var& w, float eps, float scale) {
    if (sources.empty()) throw std::runtime_error("depth_attend: needs at least one source");
    if (sources.size() == 1) {
        // Softmax over one source: the source, bit for bit (and no node).
        if (scale != 1.0f) throw std::runtime_error("depth_attend: a scaled mixture needs two or more sources");
        return sources[0];
    }
    const size_t S = sources.size(), T = sources[0]->data.rows(), d = sources[0]->data.cols();
    if (w->data.rows() != 1 || w->data.cols() != d)
        throw std::runtime_error("depth_attend: the query must be [1, d]");
    for (const auto& s : sources)
        if (s->data.rows() != T || s->data.cols() != d)
            throw std::runtime_error("depth_attend: every source must be [T, d]");
    std::vector<const Matrix*> src(S);
    for (size_t i = 0; i < S; ++i) src[i] = &sources[i]->data;
    Matrix out = device::out_matrix(T, d);
    auto alpha = cached(device::out_matrix(T, S));
    auto rinv = cached(device::out_matrix(T, S));
    auto logit = cached(device::out_matrix(T, S));
    if (!device::devops::depth_attend_fwd(src, w->data, eps, scale, out, *alpha, *rinv, *logit)) {
        for (const Matrix* m : src) device::materialize(*m);
        device::materialize(w->data);
        const float* wq = w->data.get_data();
        const float fd = static_cast<float>(d);
#pragma omp parallel for schedule(static)
        for (long long tt = 0; tt < static_cast<long long>(T); ++tt) {
            const size_t t = static_cast<size_t>(tt);
            std::vector<float> l(S), e(S);
            for (size_t i = 0; i < S; ++i) {
                const float* row = src[i]->get_data() + t * d;
                float ss = 0.0f;
                for (size_t j = 0; j < d; ++j) ss += row[j] * row[j];
                const float r = 1.0f / std::sqrt(ss / fd + eps);
                float acc = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    const float n = row[j] * r;
                    acc += n * wq[j];
                }
                (*rinv)(t, i) = r;
                (*logit)(t, i) = l[i] = acc;
            }
            float mx = l[0];
            for (size_t i = 1; i < S; ++i) mx = std::max(mx, l[i]);
            float z = 0.0f;
            for (size_t i = 0; i < S; ++i) z += (e[i] = std::exp(l[i] - mx));
            for (size_t i = 0; i < S; ++i) (*alpha)(t, i) = e[i] / z;
            float* o = out.get_data() + t * d;
            const float* r0 = src[0]->get_data() + t * d;
            const float a0 = (*alpha)(t, 0);
            for (size_t j = 0; j < d; ++j) o[j] = r0[j] * a0;
            for (size_t i = 1; i < S; ++i) {
                const float* row = src[i]->get_data() + t * d;
                const float ai = (*alpha)(t, i);
                for (size_t j = 0; j < d; ++j) {
                    const float p = row[j] * ai;
                    o[j] = o[j] + p;
                }
            }
            if (scale != 1.0f)
                for (size_t j = 0; j < d; ++j) o[j] = o[j] * scale;
        }
    }
    std::vector<Var> parents(sources.begin(), sources.end());
    parents.push_back(w);
    bool needs = false;
    if (grad_enabled())
        for (const auto& p : parents) needs = needs || p->requires_grad;
    Var res = make_var(std::move(out), needs);
    if (!needs) return res;
    res->parents = std::move(parents);
    Variable* self = res.get();
    res->backward_fn = [self, alpha, rinv, logit, S, T, d, scale]() {
        const Var& wv = self->parents[S];
        std::vector<const Matrix*> src(S);
        for (size_t i = 0; i < S; ++i) src[i] = &self->parents[i]->data;
        std::vector<Matrix> ds(S);
        std::vector<Matrix*> dsp(S, nullptr);
        for (size_t i = 0; i < S; ++i)
            if (self->parents[i]->requires_grad) {
                ds[i] = device::out_matrix(T, d);
                dsp[i] = &ds[i];
            }
        Matrix dw;
        if (wv->requires_grad) dw = device::out_matrix(1, d);
        if (!device::devops::depth_attend_bwd(src, wv->data, scale, *alpha, *rinv, *logit, self->grad, dsp,
                                              wv->requires_grad ? &dw : nullptr)) {
            device::materialize(self->grad);
            for (const Matrix* m : src) device::materialize(*m);
            device::materialize(wv->data);
            for (const Matrix* m : {alpha.get(), rinv.get(), logit.get()}) device::materialize(*m);
            const float* wq = wv->data.get_data();
            const float fd = static_cast<float>(d);
            // coef(t, i) = dl_i r_i, kept for the query's gradient below.
            Matrix coef(T, S);
#pragma omp parallel for schedule(static)
            for (long long tt = 0; tt < static_cast<long long>(T); ++tt) {
                const size_t t = static_cast<size_t>(tt);
                // The mixture's gradient: scale * dY (dY itself when unscaled).
                std::vector<float> dyv(self->grad.get_data() + t * d, self->grad.get_data() + (t + 1) * d);
                if (scale != 1.0f)
                    for (auto& v : dyv) v = v * scale;
                const float* dy = dyv.data();
                std::vector<float> g(S);
                float gbar = 0.0f;
                for (size_t i = 0; i < S; ++i) {
                    const float* row = src[i]->get_data() + t * d;
                    float acc = 0.0f;
                    for (size_t j = 0; j < d; ++j) acc += dy[j] * row[j];
                    g[i] = acc;
                    gbar += (*alpha)(t, i) * acc;
                }
                for (size_t i = 0; i < S; ++i) {
                    const float a = (*alpha)(t, i), r = (*rinv)(t, i);
                    const float dl = a * (g[i] - gbar);
                    coef(t, i) = dl * r;
                    if (!dsp[i]) continue;
                    const float* row = src[i]->get_data() + t * d;
                    float* o = dsp[i]->get_data() + t * d;
                    const float k = ((*logit)(t, i) * r) / fd;
                    for (size_t j = 0; j < d; ++j) o[j] = a * dy[j] + (dl * r) * (wq[j] - k * row[j]);
                }
            }
            if (wv->requires_grad) {
                // Column-parallel, each column summed over (t, i) in one fixed
                // order, so the thread count never changes the result.
#pragma omp parallel for schedule(static)
                for (long long jj = 0; jj < static_cast<long long>(d); ++jj) {
                    const size_t j = static_cast<size_t>(jj);
                    float acc = 0.0f;
                    for (size_t t = 0; t < T; ++t)
                        for (size_t i = 0; i < S; ++i) acc += coef(t, i) * (*src[i])(t, j);
                    dw(0, j) = acc;
                }
            }
        }
        for (size_t i = 0; i < S; ++i)
            if (dsp[i]) self->parents[i]->accumulate(std::move(ds[i]));
        if (wv->requires_grad) wv->accumulate(std::move(dw));
    };
    return res;
}

}  // namespace ops

#ifndef MICROTORCH_CUDA
namespace device {
namespace devops {
bool depth_attend_fwd(const std::vector<const Matrix*>&, const Matrix&, float, float, Matrix&, Matrix&, Matrix&,
                      Matrix&) {
    return false;
}
bool depth_attend_bwd(const std::vector<const Matrix*>&, const Matrix&, float, const Matrix&, const Matrix&,
                      const Matrix&, const Matrix&, const std::vector<Matrix*>&, Matrix*) {
    return false;
}
}  // namespace devops
}  // namespace device
#endif

}  // namespace microtorch
