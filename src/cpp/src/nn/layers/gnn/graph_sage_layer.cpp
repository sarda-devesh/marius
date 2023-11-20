//
// Created by Jason Mohoney on 9/29/21.
//

#include "nn/layers/gnn/graph_sage_layer.h"

#include "nn/layers/gnn/layer_helpers.h"
#include "reporting/logger.h"

GraphSageLayer::GraphSageLayer(shared_ptr<LayerConfig> layer_config, torch::Device device) {
    config_ = layer_config;
    options_ = std::dynamic_pointer_cast<GraphSageLayerOptions>(config_->options);
    input_dim_ = config_->input_dim;
    output_dim_ = config_->output_dim;
    device_ = device;

    reset();
}

void GraphSageLayer::reset() {
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    // Note: need to multiply the fans by 1/2 to match DGL's initialization
    torch::Tensor edge_mat = initialize_tensor(config_->init, {input_dim_, output_dim_}, tensor_options).set_requires_grad(true);
    w1_ = register_parameter("w1", edge_mat);

    if (options_->aggregator == GraphSageAggregator::MEAN) {
        edge_mat = initialize_tensor(config_->init, {input_dim_, output_dim_}, tensor_options).set_requires_grad(true);
        w2_ = register_parameter("w2", edge_mat);
    }

    if (config_->bias) {
        init_bias();
    }
}

torch::Tensor GraphSageLayer::forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train) {
    inputs = inputs.to(torch::kFloat32);

    torch::Tensor total_num_neighbors;
    torch::Tensor a_i;
    //
    //    return in_neighbors_mapping_;
    //    return out_neighbors_mapping_;

    if (dense_graph.out_neighbors_mapping_.defined()) {
        Indices outgoing_neighbors = dense_graph.getNeighborIDs(false, false);
        Indices outgoing_neighbor_offsets = dense_graph.getNeighborOffsets(false);
        torch::Tensor outgoing_num = dense_graph.getNumNeighbors(false);

        torch::Tensor outgoing_embeddings = inputs.index_select(0, outgoing_neighbors);

        total_num_neighbors = outgoing_num;
        a_i = segmented_sum_with_offsets(outgoing_embeddings, outgoing_neighbor_offsets);
    }

    if (dense_graph.in_neighbors_mapping_.defined()) {
        Indices incoming_neighbors = dense_graph.getNeighborIDs(true, false);
        Indices incoming_neighbor_offsets = dense_graph.getNeighborOffsets(true);
        torch::Tensor incoming_num = dense_graph.getNumNeighbors(true);

        if (total_num_neighbors.defined()) {
            total_num_neighbors = total_num_neighbors + incoming_num;
        } else {
            total_num_neighbors = incoming_num;
        }

        torch::Tensor incoming_embeddings = inputs.index_select(0, incoming_neighbors);

        if (a_i.defined()) {
            a_i = a_i + segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        } else {
            a_i = segmented_sum_with_offsets(incoming_embeddings, incoming_neighbor_offsets);
        }
    }

//    Indices neighbors;
//    Indices neighbor_offsets;
////    Indices total_num_neighbors;
//
//    if (dense_graph.out_neighbors_mapping_.defined() && dense_graph.in_neighbors_mapping_.defined()) {
//        auto tup = dense_graph.getCombinedNeighborIDs();
//        neighbors = std::get<2>(tup);
//        neighbor_offsets = std::get<0>(tup);
//        total_num_neighbors = std::get<1>(tup);
//    } else if (dense_graph.in_neighbors_mapping_.defined()) {
//        neighbors = dense_graph.getNeighborIDs(true, false);
//        neighbor_offsets = dense_graph.getNeighborOffsets(true);
//        total_num_neighbors = dense_graph.getNumNeighbors(true);
//    } else if (dense_graph.out_neighbors_mapping_.defined()) {
//        neighbors = dense_graph.getNeighborIDs(false, false);
//        neighbor_offsets = dense_graph.getNeighborOffsets(false);
//        total_num_neighbors = dense_graph.getNumNeighbors(false);
//    } else {
//        throw MariusRuntimeException("No neighbors defined.");
//    }
//
//    torch::Tensor nbr_embeddings = inputs.index_select(0, neighbors);
//    a_i = segmented_sum_with_offsets(nbr_embeddings, neighbor_offsets);


    int64_t layer_offset = dense_graph.getLayerOffset();
    torch::Tensor self_embs = inputs.narrow(0, layer_offset, inputs.size(0) - layer_offset);

    torch::Tensor outputs;
    if (options_->aggregator == GraphSageAggregator::GCN) {
        a_i = a_i + self_embs;
        a_i = a_i / (total_num_neighbors + 1).unsqueeze(-1);
        outputs = torch::matmul(a_i, w1_);
    } else if (options_->aggregator == GraphSageAggregator::MEAN) {
        if (total_num_neighbors.defined()) {
            torch::Tensor denominator = torch::where(torch::not_equal(total_num_neighbors, 0), total_num_neighbors, 1).to(a_i.dtype()).unsqueeze(-1);
            a_i = a_i / denominator;
            outputs = torch::matmul(self_embs, w1_) + torch::matmul(a_i, w2_);
        } else {
            outputs = torch::matmul(self_embs, w1_);
        }
    } else {
        throw std::runtime_error("Unrecognized aggregator");
    }

    return outputs;
}