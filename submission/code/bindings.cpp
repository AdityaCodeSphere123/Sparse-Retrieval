#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ir_config.h"
#include "codecs.h"
#include "engine.h"
#include "analysis.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "Inverted index, BM25+ scoring, optional RM3 expansion";
    m.def("check_codecs", &check_codecs,
          "S9Reader/EFReader/s9_skip must match simple9_unpack/ef_unpack_docs");
    m.def("tokenize", [](const std::string& text, bool remove_stop) {
            TokenizerCfg cfg;
            cfg.remove_stop = remove_stop ? 1 : 0;
            auto toks = cpp_tokenize(text, cfg);
            std::vector<std::string> out;
            out.reserve(toks.size());
            for (const auto& t : toks) out.push_back(t.first);
            return out;
        }, py::arg("text"), py::arg("remove_stop") = true,
        "Same analysis chain as retrieve(): lowercase, alnum, drop length-1, optional stops, Porter.");
    py::class_<CPPEngine>(m, "CPPEngine")
        .def(py::init<>())
        .def("build_index", &CPPEngine::build_index, py::arg("corpus"), py::arg("index_dir"))
        .def("build_index_from_file", &CPPEngine::build_index_from_file,
             py::arg("corpus_path"), py::arg("index_dir"))
        .def("load_index", &CPPEngine::load_index, py::arg("index_dir"))
        .def("retrieve", &CPPEngine::retrieve,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f,
             py::arg("w_title") = 0.0f, py::arg("title_bonus") = 0.0f,
             py::arg("use_rm3") = true, py::arg("fb_docs") = 10, py::arg("fb_terms") = 25,
             py::arg("alpha") = 0.6f, py::arg("mu") = 1000.0f,
             py::arg("rm3_pool") = RM3_POOL,
             py::arg("apply_qlen_heuristics") = true)
        .def("retrieve_bm25", &CPPEngine::retrieve_bm25,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f,
             py::arg("w_title") = 0.0f, py::arg("title_bonus") = 0.0f)
        .def_readonly("N", &CPPEngine::N)
        .def_readonly("avg_doc_len", &CPPEngine::avg_doc_len)
        .def_readonly("total_corpus_tokens", &CPPEngine::total_corpus_tokens);
}
