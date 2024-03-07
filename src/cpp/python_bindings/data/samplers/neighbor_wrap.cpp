//
// Created by Jason Mohoney on 2/14/22.
//

#include "common/pybind_headers.h"
#include "data/samplers/neighbor.h"
#include "data/features_loader.h"

class PyNeighborSampler : NeighborSampler {
   public:
    using NeighborSampler::NeighborSampler;
    DENSEGraph getNeighbors(torch::Tensor node_ids, shared_ptr<MariusGraph> graph, int worker_id) override {
        PYBIND11_OVERRIDE_PURE_NAME(DENSEGraph, NeighborSampler, "getNeighbors", getNeighbors, node_ids, graph, worker_id);
    }

    int64_t getNeighborsPages(torch::Tensor node_ids, shared_ptr<MariusGraph> graph, int worker_id) override {
        PYBIND11_OVERRIDE_PURE_NAME(int64_t, NeighborSampler, "getNeighborsPages", getNeighborsPages, node_ids, graph, worker_id);
    }
};

void init_neighbor_samplers(py::module &m) {
    py::class_<NeighborSampler, PyNeighborSampler, std::shared_ptr<NeighborSampler>>(m, "NeighborSampler")
        .def_readwrite("storage", &NeighborSampler::storage_)
        .def("getNeighbors", &NeighborSampler::getNeighbors, py::arg("node_ids"), py::arg("graph") = nullptr, py::arg("worker_id") = 0)
        .def("getNeighborsPages", &NeighborSampler::getNeighborsPages, py::arg("node_ids"), py::arg("graph") = nullptr, py::arg("worker_id") = 0);

    py::class_<LayeredNeighborSampler, NeighborSampler, std::shared_ptr<LayeredNeighborSampler>>(m, "LayeredNeighborSampler")
        .def_readwrite("sampling_layers", &LayeredNeighborSampler::sampling_layers_)
        .def("getAvgScalingFactor", &LayeredNeighborSampler::getAvgScalingFactor)
        .def("getAvgPercentRemoved", &LayeredNeighborSampler::getAvgPercentRemoved)

        .def(py::init([](shared_ptr<GraphModelStorage> storage, std::vector<int> num_neighbors, bool use_hashmap_sets) {
                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;
                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(storage, sampling_layers);
             }),
             py::arg("storage"), py::arg("num_neighbors"), py::arg("use_hashmap_sets") = false)

        .def(py::init([](shared_ptr<MariusGraph> graph, std::vector<int> num_neighbors, bool use_hashmap_sets) {
                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;

                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(graph, sampling_layers);
             }),
             py::arg("graph"), py::arg("num_neighbors"), py::arg("use_hashmap_sets") = false)
        
        .def(py::init([](shared_ptr<MariusGraph> graph, std::vector<int> num_neighbors, torch::Tensor in_mem_nodes, shared_ptr<FeaturesLoaderConfig> features_config, 
                        bool use_incoming_nbrs, bool use_outgoing_nbrs, bool use_hashmap_sets) {

                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;
                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(graph, sampling_layers, in_mem_nodes, features_config, use_incoming_nbrs, use_outgoing_nbrs);
             }),
             py::arg("graph"), py::arg("num_neighbors"), py::arg("in_mem_nodes"), py::arg("features_config"), py::arg("use_incoming_nbrs") = false, 
             py::arg("use_incoming_nbrs") = true, py::arg("use_hashmap_sets") = false)

        .def(py::init([](std::vector<int> num_neighbors, bool use_hashmap_sets) {
                 std::vector<shared_ptr<NeighborSamplingConfig>> sampling_layers;

                 for (auto n : num_neighbors) {
                     shared_ptr<NeighborSamplingConfig> ptr = std::make_shared<NeighborSamplingConfig>();
                     if (n == -1) {
                         ptr->type = NeighborSamplingLayer::ALL;
                         ptr->options = std::make_shared<NeighborSamplingOptions>();
                     } else {
                         ptr->type = NeighborSamplingLayer::UNIFORM;
                         auto opts = std::make_shared<UniformSamplingOptions>();
                         opts->max_neighbors = n;
                         ptr->options = opts;
                     }
                     ptr->use_hashmap_sets = use_hashmap_sets;
                     sampling_layers.emplace_back(ptr);
                 }
                 return std::make_shared<LayeredNeighborSampler>(sampling_layers);
             }),
             py::arg("num_neighbors"), py::arg("use_hashmap_sets") = false);
}
