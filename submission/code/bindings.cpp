#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ir_config.h"
#include "codecs.h"
#include "engine.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "From-scratch inverted index with BM25F+ scoring and RM3 query expansion";
    m.def("check_codecs", &check_codecs,
          "S9Reader/EFReader/s9_skip must match simple9_unpack/ef_unpack_docs");
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
             py::arg("rm3_pool") = RM3_POOL, py::arg("fb_temp") = 20.0f,
             py::arg("fusion_depth") = 100, py::arg("rrf_c") = 60.0f,
             py::arg("rrf_w") = 0.4f, py::arg("use_fusion") = false,
             py::arg("fusion_kind") = 1,
             py::arg("prf2_docs") = 30, py::arg("prf2_terms") = 40,
             py::arg("prf2_w") = 0.25f, py::arg("prf2_support") = 0.0f,
             py::arg("prf2_rank_c") = 10.0f,
             py::arg("rm3_fb_docs") = 50, py::arg("rm3_fb_terms") = 15,
             py::arg("rm3_lambda") = 0.5f, py::arg("rm3_mu") = 1000.0f,
             py::arg("rm3_support") = 0.15f, py::arg("rm3_idf_w") = 0.5f)
        .def("retrieve_bm25", &CPPEngine::retrieve_bm25,
             py::arg("query"), py::arg("k"),
             py::arg("k1") = 1.2f, py::arg("b") = 0.75f, py::arg("delta") = 1.0f,
             py::arg("w_title") = 0.0f, py::arg("title_bonus") = 0.0f)
        .def_readonly("N", &CPPEngine::N)
        .def_readonly("avg_doc_len", &CPPEngine::avg_doc_len)
        .def_readonly("total_corpus_tokens", &CPPEngine::total_corpus_tokens);
}
