// pybind11 surface for microtorch. Exposes the tape (Variable/backward),
// the op set, the layer zoo, optimizers + schedulers, and safetensors IO.
//
// Numpy interop: Matrix <-> numpy.ndarray (float32, 2-D) copies at the
// boundary. Zero-copy views are deliberately NOT offered -- the tape owns
// its buffers and a leaked view into a freed Matrix is the exact class of
// bug this library exists to avoid.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "microtorch/nn.hpp"
#include "microtorch/safetensors.hpp"

namespace py = pybind11;
using namespace microtorch;

namespace {

Matrix numpy_to_matrix(const py::array_t<float, py::array::c_style | py::array::forcecast>& a) {
    if (a.ndim() == 1) {
        Matrix m(1, static_cast<size_t>(a.shape(0)));
        std::memcpy(&m(0, 0), a.data(), sizeof(float) * a.shape(0));
        return m;
    }
    if (a.ndim() != 2) throw std::runtime_error("expected a 1-D or 2-D float32 array");
    Matrix m(static_cast<size_t>(a.shape(0)), static_cast<size_t>(a.shape(1)));
    std::memcpy(&m(0, 0), a.data(), sizeof(float) * a.shape(0) * a.shape(1));
    return m;
}

py::array_t<float> matrix_to_numpy(const Matrix& m) {
    py::array_t<float> a({m.rows(), m.cols()});
    std::memcpy(a.mutable_data(), &m(0, 0), sizeof(float) * m.rows() * m.cols());
    return a;
}

}  // namespace

PYBIND11_MODULE(_microtorch, mod) {
    mod.doc() = "microtorch: research-grade autograd + novel attention mechanisms";

    // ---- tape ----
    py::class_<Variable, Var>(mod, "Variable")
        .def_property_readonly("data", [](const Variable& v) { return matrix_to_numpy(v.data); })
        .def_property_readonly("grad", [](const Variable& v) {
            if (v.grad.rows() == 0) throw std::runtime_error("grad not populated; call backward()");
            return matrix_to_numpy(v.grad);
        })
        .def_readonly("requires_grad", &Variable::requires_grad)
        .def_property_readonly("shape", [](const Variable& v) {
            return py::make_tuple(v.data.rows(), v.data.cols());
        })
        .def("__repr__", [](const Variable& v) {
            return "<Variable [" + std::to_string(v.data.rows()) + ", " +
                   std::to_string(v.data.cols()) + "] requires_grad=" +
                   (v.requires_grad ? "True" : "False") + ">";
        });

    mod.def("tensor", [](py::array_t<float, py::array::c_style | py::array::forcecast> a,
                         bool requires_grad) {
        return make_var(numpy_to_matrix(a), requires_grad);
    }, py::arg("array"), py::arg("requires_grad") = false,
       "Create a Variable from a numpy float32 array (copies).");
    mod.def("backward", &backward, "Reverse pass from a scalar [1,1] root.");
    mod.def("zero_grad", &zero_grad);

    // ---- ops ----
    auto ops_mod = mod.def_submodule("ops", "differentiable op set (all gradchecked)");
    ops_mod.def("matmul", &ops::matmul);
    ops_mod.def("add", &ops::add);
    ops_mod.def("sub", &ops::sub);
    ops_mod.def("mul", &ops::mul);
    ops_mod.def("add_bias", &ops::add_bias);
    ops_mod.def("gelu", &ops::gelu);
    ops_mod.def("silu", &ops::silu);
    ops_mod.def("softmax", &ops::softmax_row);
    ops_mod.def("mean", &ops::mean);
    ops_mod.def("scale", &ops::scale);
    ops_mod.def("transpose", &ops::transpose);
    ops_mod.def("layernorm", &ops::layernorm,
                py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    ops_mod.def("rmsnorm", &ops::rmsnorm);
    ops_mod.def("embedding", &ops::embedding);
    ops_mod.def("cross_entropy", &ops::cross_entropy);
    ops_mod.def("dropout", &ops::dropout, py::arg("x"), py::arg("p"), py::arg("seed"));
    ops_mod.def("clip_grad_norm", &ops::clip_grad_norm, py::arg("params"), py::arg("max_norm"));
    ops_mod.def("kimi_attention", &ops::kimi_attention,
                py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = true,
                py::arg("seq_len") = 0);

    // ---- layers ----
    auto nn_mod = mod.def_submodule("nn", "layer zoo over the op set");
    py::class_<nn::Module, std::shared_ptr<nn::Module>>(nn_mod, "Module")
        .def("parameters", &nn::Module::parameters)
        .def("state_dict", [](const nn::Module& m) {
            py::dict d;
            for (const auto& [k, v] : m.state_dict()) d[py::str(k)] = matrix_to_numpy(v);
            return d;
        })
        .def("train", &nn::Module::train)
        .def("eval", &nn::Module::eval);

    py::class_<nn::Linear, nn::Module, std::shared_ptr<nn::Linear>>(nn_mod, "Linear")
        .def(py::init<size_t, size_t, bool, unsigned>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("bias") = true, py::arg("seed") = 0)
        .def("__call__", &nn::Linear::forward)
        .def("forward", &nn::Linear::forward);

    py::class_<nn::LayerNorm, nn::Module, std::shared_ptr<nn::LayerNorm>>(nn_mod, "LayerNorm")
        .def(py::init<size_t, float>(), py::arg("d"), py::arg("eps") = 1e-5f)
        .def("__call__", &nn::LayerNorm::forward)
        .def("forward", &nn::LayerNorm::forward);

    py::class_<nn::Dropout, nn::Module, std::shared_ptr<nn::Dropout>>(nn_mod, "Dropout")
        .def(py::init<float, unsigned long long>(), py::arg("p"), py::arg("seed") = 0)
        .def("__call__", &nn::Dropout::forward)
        .def("forward", &nn::Dropout::forward);

    py::class_<nn::CausalSelfAttention, nn::Module,
               std::shared_ptr<nn::CausalSelfAttention>>(nn_mod, "CausalSelfAttention")
        .def(py::init<size_t, size_t, unsigned, bool>(),
             py::arg("d"), py::arg("n_heads"), py::arg("seed") = 0, py::arg("causal") = true)
        .def("__call__", &nn::CausalSelfAttention::forward)
        .def("forward", &nn::CausalSelfAttention::forward);

    py::class_<nn::KimiLinearAttention, nn::Module,
               std::shared_ptr<nn::KimiLinearAttention>>(nn_mod, "KimiLinearAttention")
        .def(py::init<size_t, size_t, unsigned, bool>(),
             py::arg("d"), py::arg("n_heads"), py::arg("seed") = 0, py::arg("causal") = true)
        .def("__call__", &nn::KimiLinearAttention::forward)
        .def("forward", &nn::KimiLinearAttention::forward);

    py::class_<nn::GPT2Config>(nn_mod, "GPT2Config")
        .def(py::init<>())
        .def_readwrite("vocab", &nn::GPT2Config::vocab)
        .def_readwrite("n_ctx", &nn::GPT2Config::n_ctx)
        .def_readwrite("d", &nn::GPT2Config::d)
        .def_readwrite("n_layers", &nn::GPT2Config::n_layers)
        .def_readwrite("n_heads", &nn::GPT2Config::n_heads);

    py::class_<nn::GPT2, nn::Module, std::shared_ptr<nn::GPT2>>(nn_mod, "GPT2")
        .def(py::init<const nn::GPT2Config&, unsigned>(), py::arg("config"), py::arg("seed") = 0)
        .def("__call__", &nn::GPT2::forward)
        .def("forward", &nn::GPT2::forward);

    // ---- optimizers + schedulers ----
    py::class_<nn::SGD>(nn_mod, "SGD")
        .def(py::init<std::vector<Var>, float, float>(),
             py::arg("params"), py::arg("lr"), py::arg("momentum") = 0.0f)
        .def("step", &nn::SGD::step)
        .def("zero_grad", &nn::SGD::zero_grad)
        .def_readwrite("lr", &nn::SGD::lr);

    py::class_<nn::AdamW>(nn_mod, "AdamW")
        .def(py::init<std::vector<Var>, float, float, float, float, float>(),
             py::arg("params"), py::arg("lr") = 1e-3f, py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.01f)
        .def("step", &nn::AdamW::step)
        .def("zero_grad", &nn::AdamW::zero_grad)
        .def_readwrite("lr", &nn::AdamW::lr);

    py::class_<nn::CosineWarmupLR<nn::AdamW>>(nn_mod, "CosineWarmupLR")
        .def(py::init<nn::AdamW&, size_t, size_t, float>(),
             py::arg("optimizer"), py::arg("warmup"), py::arg("total"),
             py::arg("min_lr") = 0.0f, py::keep_alive<1, 2>())
        .def("step", &nn::CosineWarmupLR<nn::AdamW>::step);

    py::class_<nn::StepLR<nn::AdamW>>(nn_mod, "StepLR")
        .def(py::init<nn::AdamW&, size_t, float>(),
             py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f,
             py::keep_alive<1, 2>())
        .def("step", &nn::StepLR<nn::AdamW>::step);

    // ---- checkpoint IO ----
    mod.def("load_safetensors", [](const std::string& path) {
        py::dict d;
        for (const auto& [k, v] : load_safetensors(path)) d[py::str(k)] = matrix_to_numpy(v);
        return d;
    }, py::arg("path"), "Load an HF-format safetensors file as {name: ndarray}.");
    mod.def("save_safetensors", [](const std::string& path, const py::dict& tensors) {
        std::map<std::string, Matrix> m;
        for (const auto& item : tensors) {
            m.emplace(item.first.cast<std::string>(),
                      numpy_to_matrix(item.second.cast<
                          py::array_t<float, py::array::c_style | py::array::forcecast>>()));
        }
        save_safetensors(path, m);
    }, py::arg("path"), py::arg("tensors"), "Write {name: ndarray} as F32 safetensors.");
}
