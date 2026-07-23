#include "model.hpp"
#include "arena.hpp"

//TODO: implement layer_backward and model_backward functions

Layer* make_layer_dense(Arena* arena, int in_size, int out_size) {
    Layer* layer = PUSH_STRUCT(arena, Layer);
    layer->type = LayerType::Dense;
    layer->in_size = in_size;
    layer->out_size = out_size;

    return layer;
}

void init_layer(Layer* layer, Arena* arena) {
    if (layer->type == LayerType::Dense) {
        layer->weights = MatrixCreate(arena, layer->in_size, layer->out_size);
        layer->biases = MatrixCreate(arena, 1, layer->out_size);
    }

    else if(layer->type == LayerType::ReLU) {
        layer->weights = nullptr;
        layer->biases = nullptr;
    }

    else if(layer->type == LayerType::Softmax) {
        layer->weights = nullptr;
        layer->biases = nullptr;
    }

    else {
        cout << "Unknown layer type, terminating...";
        exit(1);
    }
}

Layer* make_layer_relu(Arena* arena, int size) {
    Layer* layer = PUSH_STRUCT(arena, Layer);
    layer->type = LayerType::ReLU;
    layer->in_size = size;
    layer->out_size = size;
    
    return layer;
}

Matrix* layer_forward(Layer* layer, Matrix* input) {
    layer->input = input;

    Matrix* output = new Matrix(layer->out_size, layer->in_size);

    if (layer->type == LayerType::Dense) {
        // y = w x + b
        output = MatrixMult(input, layer->weights);
        output = MatrixSum(output, layer->biases);
        layer->output = output;
    }

    else if(layer->type == LayerType::ReLU) {
        output = ReLU(input);
        layer->output =  output;
    }

    else if(layer->type == LayerType::Softmax) {
        output = softmax(input);
        layer->output = output;
    }

    else {
        cout << "Unknown layer type, terminating...";
        exit(1);
    }

}

Matrix* layer_backward(Layer* layer, Matrix* grad_output);

Model* model_create(Arena* arena, int num_layers) {
    Model* model = PUSH_STRUCT(arena, Model);
    model->num_layers = num_layers;
    model->layers = PUSH_ARRAY(arena, Layer, num_layers);

    return model;
}

Matrix* model_forward(Model* model, Matrix* input) {
    Matrix* a = input;

    for (int i = 0; i < model->num_layers; i++) {
        a = layer_forward(&model->layers[i], a);
    }

    return a;
}

Matrix* model_backward(Model* model, Matrix* grad_loss, float learning_rate);
