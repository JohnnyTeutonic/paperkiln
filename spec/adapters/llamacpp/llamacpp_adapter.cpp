// A keramos conformance adapter for llama.cpp (IR_SPEC.md section 11.3): JSON Lines on stdin and stdout,
// one request per line, over llama.h (no llama.cpp source is changed).
//
// llama.cpp reads the GGUF, never the IR document: it builds its own graph from general.architecture
// and the <arch>.* metadata, and ignores paperkiln.ir. That is what this adapter measures.
//
// What it answers (FINDINGS.txt beside it has the details, the build and the results):
//   hello, open, forward (logits; kv_rows and kv_positions read from llama.cpp's own sequence state;
//   recurrent tensors of Mamba-1 read the same way), truncate (llama_memory_seq_rm), snapshot and
//   restore (llama_state_seq_get_data / set_data), close, dequantize (ggml's to_float).
// What it skips (llama.cpp has no path for it): check (IR validation), open of a file whose
//   general.architecture llama.cpp has no graph for ("paperkiln" among them), open with a replacement
//   document (ir) or with evict true, run_node, run_node_grad, digest, render_chat, the depth report,
//   and the kv reports of a cache with sliding-window layers (llama.cpp keeps those layers in a second
//   cache whose layer assignment llama.h does not expose).
//
// Adapter-side rewrites, each reported in open's result as "_rewrites" and on stderr:
//   - the suite's tokenizer "word" is not a llama.cpp tokenizer, so tokenizer.ggml.model is overridden
//     to "no_vocab" with <arch>.vocab_size (llama_model_kv_override, as llama.cpp's --override-kv
//     does). The suite feeds token ids, so no tokenizer is used. The file on disk is read unchanged.
//   - --alias FROM=TO (off by default) loads a file whose general.architecture is FROM as
//     architecture TO, by writing a temporary GGUF with general.architecture and the FROM.* keys
//     renamed (tensors unchanged), as llama.cpp's converter would have named it.
//
// Options: --threads N (2), --kv-type f32|f16 (f32), --flash-attn 0|1 (0), --swa-full 0|1 (1, the
//   llama.cpp default), --n-ubatch N (512), --n-ctx N (0: the model's context_length),
//   --alias FROM=TO, --rewind-guard (re-prefill after a truncate that left a sliding-window cache
//   without the rows the next token reads, as llama-server does), --tmp DIR (/tmp), --verbose.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#ifndef LC_LLAMA_COMMIT
#define LC_LLAMA_COMMIT "unknown"
#endif

using json = nlohmann::json;

namespace {

struct Skip : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    int threads = 2;
    ggml_type kv_type = GGML_TYPE_F32;
    bool flash_attn = false;
    bool swa_full = true;
    int n_ubatch = 512;
    int n_ctx = 0;
    std::map<std::string, std::string> alias;
    bool rewind_guard = false;
    std::string tmp = "/tmp";
    bool verbose = false;
} opt;

std::string g_last_error;

void log_cb(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            g_last_error += text;
        }
    }
    if (opt.verbose || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

// ------------------------------------------------------------------ tensors (IR_SPEC 11.2)

json enc(const std::vector<float> & v, std::vector<int64_t> shape) {
    static const char * hexd = "0123456789abcdef";
    std::string s;
    s.reserve(v.size() * 8);
    for (float f : v) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        for (int k = 7; k >= 0; --k) {
            s.push_back(hexd[(u >> (4 * k)) & 15]);
        }
    }
    return json{{"shape", shape}, {"f32", s}};
}

std::vector<uint8_t> unhex(const std::string & h) {
    if (h.size() % 2) {
        throw std::runtime_error("odd hex length");
    }
    std::vector<uint8_t> out(h.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = (uint8_t) std::stoul(h.substr(2 * i, 2), nullptr, 16);
    }
    return out;
}

// ------------------------------------------------------------------ GGUF metadata

struct GgufMeta {
    std::string arch, tokenizer;
    int64_t n_tokens = -1;
    std::map<std::string, int64_t> ints;
};

GgufMeta read_meta(const std::string & path) {
    gguf_init_params p{true, nullptr};
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        throw std::runtime_error("gguf_init_from_file failed on " + path);
    }
    GgufMeta m;
    int64_t k = gguf_find_key(g, "general.architecture");
    if (k >= 0) {
        m.arch = gguf_get_val_str(g, k);
    }
    k = gguf_find_key(g, "tokenizer.ggml.model");
    if (k >= 0) {
        m.tokenizer = gguf_get_val_str(g, k);
    }
    k = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (k >= 0) {
        m.n_tokens = (int64_t) gguf_get_arr_n(g, k);
    }
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        const auto t = gguf_get_kv_type(g, i);
        if (t == GGUF_TYPE_UINT32) {
            m.ints[gguf_get_key(g, i)] = gguf_get_val_u32(g, i);
        } else if (t == GGUF_TYPE_INT32) {
            m.ints[gguf_get_key(g, i)] = gguf_get_val_i32(g, i);
        } else if (t == GGUF_TYPE_UINT64) {
            m.ints[gguf_get_key(g, i)] = (int64_t) gguf_get_val_u64(g, i);
        }
    }
    gguf_free(g);
    return m;
}

// A copy of the file with general.architecture FROM renamed TO and every FROM.* key renamed TO.*;
// the tensors are copied unchanged.
std::string write_alias(const std::string & path, const std::string & from, const std::string & to) {
    ggml_context * data = nullptr;
    gguf_init_params p{false, &data};
    gguf_context * src = gguf_init_from_file(path.c_str(), p);
    if (!src) {
        throw std::runtime_error("gguf_init_from_file failed on " + path);
    }
    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    std::vector<std::string> keys;
    for (int64_t i = 0; i < gguf_get_n_kv(src); ++i) {
        keys.emplace_back(gguf_get_key(src, i));
    }
    for (const auto & key : keys) {
        if (key.rfind(from + ".", 0) != 0) {
            continue;
        }
        const std::string nk = to + key.substr(from.size());
        const int64_t i = gguf_find_key(src, key.c_str());
        switch (gguf_get_kv_type(src, i)) {
            case GGUF_TYPE_UINT32: gguf_set_val_u32(dst, nk.c_str(), gguf_get_val_u32(src, i)); break;
            case GGUF_TYPE_INT32: gguf_set_val_i32(dst, nk.c_str(), gguf_get_val_i32(src, i)); break;
            case GGUF_TYPE_UINT64: gguf_set_val_u64(dst, nk.c_str(), gguf_get_val_u64(src, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32(dst, nk.c_str(), gguf_get_val_f32(src, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64(dst, nk.c_str(), gguf_get_val_f64(src, i)); break;
            case GGUF_TYPE_BOOL: gguf_set_val_bool(dst, nk.c_str(), gguf_get_val_bool(src, i)); break;
            case GGUF_TYPE_STRING: gguf_set_val_str(dst, nk.c_str(), gguf_get_val_str(src, i)); break;
            default: throw std::runtime_error("alias: key " + key + " has a type this adapter does not copy");
        }
        gguf_remove_key(dst, key.c_str());
    }
    gguf_set_val_str(dst, "general.architecture", to.c_str());
    for (int64_t t = 0; t < gguf_get_n_tensors(src); ++t) {
        gguf_add_tensor(dst, ggml_get_tensor(data, gguf_get_tensor_name(src, t)));
    }
    char tmpl[512];
    std::snprintf(tmpl, sizeof tmpl, "%s/lc_alias_XXXXXX", opt.tmp.c_str());
    const int fd = mkstemp(tmpl);
    if (fd < 0) {
        throw std::runtime_error("mkstemp failed in " + opt.tmp);
    }
    close(fd);
    const bool ok = gguf_write_to_file(dst, tmpl, false);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(data);
    if (!ok) {
        throw std::runtime_error("alias: writing the temporary GGUF failed");
    }
    return tmpl;
}

// ------------------------------------------------------------------ llama.cpp's sequence state

struct Reader {
    const std::vector<uint8_t> & b;
    size_t off = 0;
    template <typename T> T get() {
        if (off + sizeof(T) > b.size()) {
            throw std::runtime_error("state: read past the end");
        }
        T v;
        std::memcpy(&v, b.data() + off, sizeof(T));
        off += sizeof(T);
        return v;
    }
    void skip(size_t n) {
        if (off + n > b.size()) {
            throw std::runtime_error("state: skip past the end");
        }
        off += n;
    }
};

struct KvCache {
    std::vector<int32_t> positions;  // stream 0
    uint32_t n_layer = 0;
};

// llama_kv_cache_unified::state_write (src/llama-kv-cache-unified.cpp, b6000).
KvCache parse_kv(Reader & r) {
    KvCache c;
    const uint32_t n_stream = r.get<uint32_t>();
    for (uint32_t s = 0; s < n_stream; ++s) {
        const uint32_t cells = r.get<uint32_t>();
        if (cells == 0) {
            continue;
        }
        for (uint32_t i = 0; i < cells; ++i) {
            const int32_t pos = r.get<int32_t>();
            const uint32_t n_seq = r.get<uint32_t>();
            r.skip(4 * (size_t) n_seq);
            if (s == 0) {
                c.positions.push_back(pos);
            }
        }
        const uint32_t v_trans = r.get<uint32_t>();
        const uint32_t n_layer = r.get<uint32_t>();
        c.n_layer = n_layer;
        for (uint32_t l = 0; l < n_layer; ++l) {
            r.get<int32_t>();
            r.skip(cells * r.get<uint64_t>());
        }
        for (uint32_t l = 0; l < n_layer; ++l) {
            r.get<int32_t>();
            if (!v_trans) {
                r.skip(cells * r.get<uint64_t>());
            } else {
                const uint32_t el = r.get<uint32_t>();
                const uint32_t n_embd = r.get<uint32_t>();
                r.skip((size_t) n_embd * cells * el);
            }
        }
    }
    std::sort(c.positions.begin(), c.positions.end());
    return c;
}

struct RecState {
    std::vector<std::vector<float>> r, s;  // per layer, llama.cpp's layout
};

// llama_memory_recurrent::state_write (src/llama-memory-recurrent.cpp, b6000).
RecState parse_recurrent(Reader & rd) {
    RecState st;
    const uint32_t cells = rd.get<uint32_t>();
    if (cells != 1) {
        throw std::runtime_error("recurrent state: " + std::to_string(cells) + " cells for one sequence");
    }
    rd.get<int32_t>();
    rd.skip(4 * (size_t) rd.get<uint32_t>());
    const uint32_t s_trans = rd.get<uint32_t>();
    const uint32_t n_layer = rd.get<uint32_t>();
    if (s_trans) {
        throw std::runtime_error("recurrent state: transposed s is not read by this adapter");
    }
    for (int part = 0; part < 2; ++part) {
        for (uint32_t l = 0; l < n_layer; ++l) {
            const int32_t type = rd.get<int32_t>();
            const uint64_t row = rd.get<uint64_t>();
            if (type != GGML_TYPE_F32 || row % 4) {
                throw std::runtime_error("recurrent state: a tensor that is not f32");
            }
            std::vector<float> v(row / 4);
            std::memcpy(v.data(), rd.b.data() + rd.off, row);
            rd.skip(row);
            (part == 0 ? st.r : st.s).push_back(std::move(v));
        }
    }
    return st;
}

// ------------------------------------------------------------------ sessions

struct Snapshot {
    std::vector<uint8_t> data;
    std::vector<llama_token> ids;
};

struct Session {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::vector<llama_token> ids;
    std::map<std::string, Snapshot> snaps;
    int n_layer = 0, n_vocab = 0, n_swa = 0;
    bool recurrent = false;
    int64_t conv_k = 0, d_inner = 0, d_state = 0;  // Mamba-1, from the GGUF metadata
    std::string arch, tmp_file;
    std::vector<std::string> notes;  // how truncate and restore were done, for stderr

    ~Session() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
        if (!tmp_file.empty()) {
            std::remove(tmp_file.c_str());
        }
    }
};

std::map<std::string, std::unique_ptr<Session>> sessions;
int n_sessions = 0;

Session & get_session(const json & p) {
    auto it = sessions.find(p.at("session").get<std::string>());
    if (it == sessions.end()) {
        throw std::runtime_error("no such session");
    }
    return *it->second;
}

std::vector<uint8_t> seq_state(Session & s) {
    const size_t n = llama_state_seq_get_size(s.ctx, 0);
    std::vector<uint8_t> b(n);
    const size_t got = llama_state_seq_get_data(s.ctx, b.data(), n, 0);
    b.resize(got);
    return b;
}

bool wants(const json & report, const char * what) {
    for (const auto & r : report) {
        if (r == what) {
            return true;
        }
    }
    return false;
}

// Decided before anything runs (IR_SPEC 11.3: a skip covers the whole call).
void check_reports(const Session & s, const json & report) {
    for (const auto & r : report) {
        const std::string x = r.get<std::string>();
        if (x == "depth") {
            throw Skip("llama.cpp has no depth_route (the depth report)");
        }
        if ((x == "kv_rows" || x == "kv_positions") && !s.recurrent && s.n_swa > 0) {
            throw Skip("kv rows of a sliding-window model: llama.cpp keeps its windowed layers in a second cache, "
                       "and llama.h does not say which layers it holds");
        }
        if (x == "recurrent" && s.recurrent && s.arch != "mamba") {
            throw Skip("recurrent report: only Mamba-1's state layout is mapped by this adapter");
        }
    }
}

json reports(Session & s, const json & report) {
    json out = json::object();
    const bool kv = wants(report, "kv_rows") || wants(report, "kv_positions");
    const bool rec = wants(report, "recurrent");
    if (!kv && !rec) {
        return out;
    }
    const std::vector<uint8_t> b = seq_state(s);
    Reader rd{b};
    if (kv) {
        json rows = json::object(), pos = json::object();
        if (!s.recurrent) {
            KvCache c = parse_kv(rd);
            json r_apps = json::array(), p_apps = json::array();
            for (int l = 0; l < s.n_layer; ++l) {
                r_apps.push_back(c.positions.size());
                p_apps.push_back(c.positions);
            }
            rows["kv"] = r_apps;
            pos["kv"] = p_apps;
        }
        if (wants(report, "kv_rows")) {
            out["kv_rows"] = rows;
        }
        if (wants(report, "kv_positions")) {
            out["kv_positions"] = pos;
        }
    }
    if (rec) {
        json r = json::object();
        if (s.recurrent) {
            const int64_t K1 = s.conv_k - 1, Di = s.d_inner, N = s.d_state;
            json conv = json::array(), ssm = json::array();
            if (s.ids.empty()) {
                for (int l = 0; l < s.n_layer; ++l) {
                    conv.push_back(json{{"x", enc(std::vector<float>(K1 * Di, 0.0f), {K1, Di})}});
                    ssm.push_back(json{{"s", enc(std::vector<float>(Di * N, 0.0f), {Di, N})}});
                }
            } else {
                RecState st = parse_recurrent(rd);
                for (int l = 0; l < s.n_layer; ++l) {
                    if ((int64_t) st.r[l].size() != K1 * Di || (int64_t) st.s[l].size() != Di * N) {
                        throw std::runtime_error("recurrent state: sizes differ from the GGUF's ssm.* metadata");
                    }
                    // llama.cpp's conv state is [d_inner][d_conv - 1] (ggml ne0 = d_conv - 1); the IR's
                    // conv slot tensor x is [d_conv - 1, d_inner] row-major, oldest input first.
                    std::vector<float> x(K1 * Di);
                    for (int64_t t = 0; t < K1; ++t) {
                        for (int64_t c = 0; c < Di; ++c) {
                            x[t * Di + c] = st.r[l][c * K1 + t];
                        }
                    }
                    conv.push_back(json{{"x", enc(x, {K1, Di})}});
                    // ssm state is [d_inner][d_state] (ne0 = d_state): the IR's s [Di, N] row-major.
                    ssm.push_back(json{{"s", enc(st.s[l], {Di, N})}});
                }
            }
            r["conv"] = conv;
            r["ssm"] = ssm;
        }
        out["recurrent"] = r;
    }
    return out;
}

// Decode ids at positions len .. len + n - 1, one llama_batch per n_batch tokens; logits rows when asked.
std::vector<float> decode(Session & s, const std::vector<llama_token> & ids, bool want_logits) {
    std::vector<float> logits;
    const int n_batch = (int) llama_n_batch(s.ctx);
    size_t at = 0;
    while (at < ids.size()) {
        const int n = (int) std::min<size_t>(n_batch, ids.size() - at);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int j = 0; j < n; ++j) {
            b.token[j] = ids[at + j];
            b.pos[j] = (llama_pos) (s.ids.size() + at + j);
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = want_logits || (at + j + 1 == ids.size());
        }
        b.n_tokens = n;
        g_last_error.clear();
        const int rc = llama_decode(s.ctx, b);
        if (rc != 0) {
            llama_batch_free(b);
            throw std::runtime_error("llama_decode returned " + std::to_string(rc) + ": " + g_last_error);
        }
        if (want_logits) {
            for (int j = 0; j < n; ++j) {
                const float * row = llama_get_logits_ith(s.ctx, j);
                logits.insert(logits.end(), row, row + s.n_vocab);
            }
        }
        llama_batch_free(b);
        at += n;
    }
    return logits;
}

// ------------------------------------------------------------------ methods

json m_hello() {
    return json{{"engine", "llama.cpp"},
                {"version", std::string("llama.cpp ") + LC_LLAMA_COMMIT},
                {"backend", "cpu"},
                {"exact", false},
                {"profiles", json::array({"core", "state", "thirdparty"})},
                {"methods", json::array({"hello", "open", "forward", "truncate", "snapshot", "restore", "close",
                                         "dequantize"})},
                {"settings", json{{"threads", opt.threads}, {"kv_type", ggml_type_name(opt.kv_type)},
                                  {"flash_attn", opt.flash_attn}, {"swa_full", opt.swa_full},
                                  {"n_ubatch", opt.n_ubatch}, {"n_ctx", opt.n_ctx},
                                  {"rewind_guard", opt.rewind_guard}, {"alias", opt.alias}}}};
}

json m_open(const json & p) {
    if (p.contains("ir") && !p["ir"].is_null()) {
        throw Skip("llama.cpp ignores the IR document and cannot run a replacement one");
    }
    if (p.contains("evict") && p["evict"].is_boolean() && p["evict"].get<bool>()) {
        throw Skip("llama.cpp has no turn_state eviction");
    }
    const std::string path = p.at("gguf").get<std::string>();
    GgufMeta m = read_meta(path);
    auto s = std::make_unique<Session>();
    json rewrites = json::array();
    std::string load_path = path, arch = m.arch;
    if (opt.alias.count(m.arch)) {
        arch = opt.alias[m.arch];
        s->tmp_file = write_alias(path, m.arch, arch);
        load_path = s->tmp_file;
        rewrites.push_back("general.architecture " + m.arch + " -> " + arch + " and " + m.arch + ".* -> " + arch +
                           ".* (a temporary copy; tensors unchanged)");
    }
    std::vector<llama_model_kv_override> ov;
    static const char * known[] = {"llama", "bert", "gpt2", "t5", "rwkv", "plamo2", "no_vocab", "none"};
    if (std::find_if(std::begin(known), std::end(known), [&](const char * k) { return m.tokenizer == k; }) ==
        std::end(known)) {
        if (m.n_tokens <= 0) {
            throw std::runtime_error("tokenizer " + m.tokenizer + " without tokenizer.ggml.tokens");
        }
        llama_model_kv_override a{};
        a.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        std::snprintf(a.key, sizeof a.key, "tokenizer.ggml.model");
        std::snprintf(a.val_str, sizeof a.val_str, "no_vocab");
        ov.push_back(a);
        llama_model_kv_override b{};
        b.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::snprintf(b.key, sizeof b.key, "%s.vocab_size", arch.c_str());
        b.val_i64 = m.n_tokens;
        ov.push_back(b);
        rewrites.push_back("tokenizer.ggml.model " + m.tokenizer + " -> no_vocab, " + arch + ".vocab_size = " +
                           std::to_string(m.n_tokens) + " (kv overrides)");
    }
    llama_model_kv_override end{};
    ov.push_back(end);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.kv_overrides = ov.size() > 1 ? ov.data() : nullptr;
    g_last_error.clear();
    s->model = llama_model_load_from_file(load_path.c_str(), mp);
    if (!s->model) {
        std::string e = g_last_error;
        while (!e.empty() && (e.back() == '\n' || e.back() == ' ')) {
            e.pop_back();
        }
        if (e.find("unknown model architecture") != std::string::npos) {
            // No graph for this architecture: no path for the case (nothing was loaded, so the session
            // set is as it was). "paperkiln" is IR_SPEC 9.1's architecture of files no llama.cpp graph
            // computes, so refusing it is the right behaviour, not a failure.
            throw Skip("llama.cpp has no graph for general.architecture '" + arch + "'");
        }
        throw std::runtime_error("llama.cpp did not load the file: " + e);
    }
    s->arch = arch;
    s->n_layer = llama_model_n_layer(s->model);
    s->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(s->model));
    s->n_swa = llama_model_n_swa(s->model);
    s->recurrent = llama_model_is_recurrent(s->model);
    auto mi = [&](const std::string & k) { return m.ints.count(m.arch + "." + k) ? m.ints[m.arch + "." + k] : 0; };
    s->conv_k = mi("ssm.conv_kernel");
    s->d_inner = mi("ssm.inner_size");
    s->d_state = mi("ssm.state_size");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = opt.n_ctx > 0 ? opt.n_ctx : std::max(64, llama_model_n_ctx_train(s->model));
    cp.n_batch = 512;
    cp.n_ubatch = std::min(512, opt.n_ubatch);
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = opt.threads;
    cp.type_k = cp.type_v = opt.kv_type;
    cp.flash_attn = opt.flash_attn;
    cp.swa_full = opt.swa_full;
    cp.no_perf = true;
    s->ctx = llama_init_from_model(s->model, cp);
    if (!s->ctx) {
        throw std::runtime_error("llama_init_from_model failed: " + g_last_error);
    }
    const std::string sid = "s" + std::to_string(++n_sessions);
    for (const auto & r : rewrites) {
        std::fprintf(stderr, "[lc] %s: %s\n", sid.c_str(), r.get<std::string>().c_str());
    }
    json res{{"session", sid}, {"vocab", s->n_vocab}, {"applications", s->n_layer}, {"state_k", 0},
             {"_rewrites", rewrites}, {"_arch", arch}, {"_n_swa", s->n_swa}, {"_recurrent", s->recurrent}};
    sessions[sid] = std::move(s);
    return res;
}

json m_forward(const json & p) {
    Session & s = get_session(p);
    const json report = p.value("report", json::array());
    std::vector<llama_token> ids = p.at("ids").get<std::vector<llama_token>>();
    if (ids.empty()) {
        throw std::runtime_error("forward: ids is empty (IR_SPEC 11.3)");
    }
    check_reports(s, report);
    for (auto t : ids) {
        if (t < 0 || t >= s.n_vocab) {
            throw std::runtime_error("forward: id " + std::to_string(t) + " outside the vocabulary");
        }
    }
    const bool lg = wants(report, "logits");
    std::vector<float> logits = decode(s, ids, lg);
    s.ids.insert(s.ids.end(), ids.begin(), ids.end());
    json out = reports(s, report);
    out["len"] = s.ids.size();
    if (lg) {
        out["logits"] = enc(logits, {(int64_t) ids.size(), s.n_vocab});
    }
    return out;
}

void reprefill(Session & s, size_t n) {
    llama_memory_t mem = llama_get_memory(s.ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);
    std::vector<llama_token> keep(s.ids.begin(), s.ids.begin() + n);
    s.ids.clear();
    if (n) {
        decode(s, keep, false);
    }
    s.ids = keep;
}

json m_truncate(const json & p) {
    Session & s = get_session(p);
    const json report = p.value("report", json::array());
    const long n = p.at("n").get<long>();
    if (n < 0 || n > (long) s.ids.size()) {
        throw std::runtime_error("truncate to " + std::to_string(n) + ", outside 0 .. len " +
                                 std::to_string(s.ids.size()) + " (IR_SPEC 11.3)");
    }
    check_reports(s, report);
    std::string how = "unchanged";
    if (n < (long) s.ids.size()) {
        llama_memory_t mem = llama_get_memory(s.ctx);
        // llama-server's test (tools/server/server.cpp, b6000 lines 3255-3267), made before it cuts the
        // cache: the cache must still hold position max(0, n - n_swa); otherwise it re-processes the
        // prompt. llama_memory_seq_pos_min reads the sliding-window cache of an iSWA model.
        const llama_pos pmin = llama_memory_seq_pos_min(mem, 0);
        const bool guard = opt.rewind_guard && s.n_swa > 0 && n > 0 &&
                           pmin > std::max<llama_pos>(0, (llama_pos) n - s.n_swa);
        if (guard) {
            reprefill(s, n);
            how = "re-prefill (rewind guard: the sliding-window cache no longer holds position max(0, n - n_swa))";
        } else if (llama_memory_seq_rm(mem, 0, (llama_pos) n, -1)) {
            how = "llama_memory_seq_rm";
            s.ids.resize(n);
        } else {
            // llama.cpp refuses to cut a recurrent state at an earlier position: re-prefill the first
            // n ids, which the session owns (IR_SPEC 6.9, 11.3).
            reprefill(s, n);
            how = "re-prefill (llama_memory_seq_rm refused)";
        }
    }
    std::fprintf(stderr, "[lc] truncate to %ld: %s\n", n, how.c_str());
    json out = reports(s, report);
    out["len"] = s.ids.size();
    out["_how"] = how;
    return out;
}

json m_snapshot(const json & p) {
    Session & s = get_session(p);
    Snapshot sn;
    sn.data = seq_state(s);
    sn.ids = s.ids;
    s.snaps[p.at("tag").get<std::string>()] = std::move(sn);
    return json{{"len", s.ids.size()}};
}

json m_restore(const json & p) {
    Session & s = get_session(p);
    const json report = p.value("report", json::array());
    auto it = s.snaps.find(p.at("tag").get<std::string>());
    if (it == s.snaps.end()) {
        throw std::runtime_error("restore: no snapshot under that tag in this session");
    }
    check_reports(s, report);
    llama_memory_t mem = llama_get_memory(s.ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (!it->second.ids.empty()) {
        g_last_error.clear();
        const size_t n = llama_state_seq_set_data(s.ctx, it->second.data.data(), it->second.data.size(), 0);
        if (n == 0) {
            throw std::runtime_error("llama_state_seq_set_data failed: " + g_last_error);
        }
    }
    s.ids = it->second.ids;
    json out = reports(s, report);
    out["len"] = s.ids.size();
    return out;
}

json m_close(const json & p) {
    sessions.erase(p.at("session").get<std::string>());
    return json::object();
}

json m_dequantize(const json & p) {
    const std::string name = p.at("type").get<std::string>();
    const int64_t n = p.at("n").get<int64_t>();
    std::vector<uint8_t> raw = unhex(p.at("bytes").get<std::string>());
    int found = -1;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (!tn) {
            continue;
        }
        std::string a(tn), b(name);
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        if (a == b) {
            found = t;
            break;
        }
    }
    if (found < 0) {
        throw Skip("ggml has no storage type " + name);
    }
    const ggml_type t = (ggml_type) found;
    if (raw.size() != ggml_row_size(t, n)) {
        throw std::runtime_error("dequantize: " + std::to_string(raw.size()) + " bytes for " + std::to_string(n) +
                                 " elements of " + name);
    }
    std::vector<float> out(n);
    if (t == GGML_TYPE_F32) {
        std::memcpy(out.data(), raw.data(), raw.size());
    } else {
        const ggml_type_traits * tr = ggml_get_type_traits(t);
        if (!tr->to_float) {
            throw Skip("ggml has no to_float for " + name);
        }
        tr->to_float(raw.data(), out.data(), n);
    }
    return json{{"values", enc(out, {n})}};
}

json dispatch(const std::string & method, const json & p) {
    if (method == "hello") return m_hello();
    if (method == "open") return m_open(p);
    if (method == "forward") return m_forward(p);
    if (method == "truncate") return m_truncate(p);
    if (method == "snapshot") return m_snapshot(p);
    if (method == "restore") return m_restore(p);
    if (method == "close") return m_close(p);
    if (method == "dequantize") return m_dequantize(p);
    if (method == "check") {
        throw Skip("llama.cpp does not validate IR documents (IR_SPEC section 10): it ignores paperkiln.ir");
    }
    if (method == "run_node" || method == "run_node_grad") {
        throw Skip("llama.cpp runs whole graphs it builds itself, not single IR nodes");
    }
    if (method == "digest") {
        throw Skip("llama.cpp has no IR digest (IR_SPEC 3.10)");
    }
    if (method == "render_chat") {
        throw Skip("llama.cpp renders its own built-in or Jinja templates, not IR template records");
    }
    throw Skip("unknown method " + method);
}

void parse_args(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(a + " needs a value");
            }
            return argv[++i];
        };
        if (a == "--threads") opt.threads = std::stoi(next());
        else if (a == "--kv-type") {
            const std::string v = next();
            if (v == "f32") opt.kv_type = GGML_TYPE_F32;
            else if (v == "f16") opt.kv_type = GGML_TYPE_F16;
            else throw std::runtime_error("--kv-type f32 or f16");
        }
        else if (a == "--flash-attn") opt.flash_attn = next() == "1";
        else if (a == "--swa-full") opt.swa_full = next() == "1";
        else if (a == "--n-ubatch") opt.n_ubatch = std::stoi(next());
        else if (a == "--n-ctx") opt.n_ctx = std::stoi(next());
        else if (a == "--alias") {
            const std::string v = next();
            const auto eq = v.find('=');
            if (eq == std::string::npos) throw std::runtime_error("--alias FROM=TO");
            opt.alias[v.substr(0, eq)] = v.substr(eq + 1);
        }
        else if (a == "--rewind-guard") opt.rewind_guard = true;
        else if (a == "--tmp") opt.tmp = next();
        else if (a == "--verbose") opt.verbose = true;
        else if (a == "--conformance") {}
        else throw std::runtime_error("unknown option " + a);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        parse_args(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llamacpp_adapter: %s\n", e.what());
        return 2;
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json reply;
        json req;
        try {
            req = json::parse(line);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "llamacpp_adapter: bad request line: %s\n", e.what());
            continue;
        }
        reply["id"] = req.value("id", 0);
        try {
            reply["result"] = dispatch(req.at("method").get<std::string>(), req.value("params", json::object()));
        } catch (const Skip & e) {
            reply["skip"] = e.what();
        } catch (const std::exception & e) {
            reply["error"] = e.what();
        }
        std::cout << reply.dump() << "\n" << std::flush;
    }
    sessions.clear();
    llama_backend_free();
    return 0;
}
