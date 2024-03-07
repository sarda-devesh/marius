
#include "common/pybind_headers.h"
#include "data/features_loader.h"

class PyFeaturesLoader : FeaturesLoader {
   public:
    using FeaturesLoader::FeaturesLoader;
    int64_t num_pages_for_nodes(torch::Tensor node_ids) override {
        PYBIND11_OVERRIDE_PURE_NAME(int64_t, FeaturesLoader, "num_pages_for_nodes", num_pages_for_nodes, node_ids);
    }
};

void init_features_loader(py::module &m) {
    py::class_<FeaturesLoaderConfig, std::shared_ptr<FeaturesLoaderConfig>>(m, "FeaturesLoaderConfig")
        .def(py::init<>())
        .def_readwrite("features_type", &FeaturesLoaderConfig::features_type)
        .def_readwrite("page_size", &FeaturesLoaderConfig::page_size)
        .def_readwrite("feature_dimension", &FeaturesLoaderConfig::feature_dimension)
        .def_readwrite("feature_size", &FeaturesLoaderConfig::feature_size);
    
    py::class_<FeaturesLoader, PyFeaturesLoader, std::shared_ptr<FeaturesLoader>>(m, "FeaturesLoader")
        .def("num_pages_for_nodes", &FeaturesLoader::num_pages_for_nodes, py::arg("node_ids"));
    
    py::class_<LinearFeaturesLoader, FeaturesLoader, std::shared_ptr<LinearFeaturesLoader>>(m, "LinearFeaturesLoader")
        .def(py::init([](shared_ptr<FeaturesLoaderConfig> config, shared_ptr<MariusGraph> graph) {
                return std::make_shared<LinearFeaturesLoader>(config, graph);
             }),
             py::arg("config"), py::arg("graph"));
    
    m.def("get_feature_loader", &get_feature_loader, py::arg("config"), py::arg("graph"));
}