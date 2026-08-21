#include "model.hpp"
#include "arena.hpp"

//TODO: implement layer_backward and model_backward functions
//TODO: implement claude fixes

Layer* model_add_layer(Model* model, LayerType type, int in_size, int out_size) {
    assert(model->curr_layer < model->num_layers && "model_add_layer: exceeded num_layers...");
    Layer* layer = &model->layers[model->curr_layer++];

    layer->type = type;
    layer->in_size = in_size;
    layer->out_size = out_size;

    return layer;
}

Layer* add_layer_dense(Arena* arena, Model* model, int in_size, int out_size) {
    Layer* layer = model_add_layer(model, LayerType::Dense, in_size, out_size);
    init_layer(layer, arena);

    return layer;
}

Layer* add_layer_relu(Arena* arena, Model* model, int size) {
    Layer* layer = model_add_layer(model, LayerType::ReLU, size, size);
    init_layer(layer, arena);

    return layer;
}

void init_layer(Layer* layer, Arena* arena) {
    if (layer->type == LayerType::Dense) {
        //xavier init weights and 0 init biases soon
        layer->weights = matrix_create(arena, layer->in_size, layer->out_size);
        layer->biases = matrix_create(arena, 1, layer->out_size);

        float scale = sqrtf(2.0f / layer->in_size);

        for(int i = 0; i < layer->weights->size; i++) {
            layer->weights->data[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
        }

        for(int i = 0; i < layer->biases->size; i++) {
            layer->biases->data[i] = 0.0f;
        }
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

Matrix* layer_forward(Arena* arena, Layer* layer, Matrix* input) {
    layer->input = input;
    if(layer->output) {
        layer->output = nullptr;
    }

    Matrix* output = nullptr;
    if (layer->type == LayerType::Dense) {
        // y = x @ w + b
        Matrix* z = matrix_mult(arena, input, layer->weights);
        assert(z && "matrix_mult failed...");
        output = matrix_add_bias(arena, z, layer->biases);
        assert(output && "matrix_sum failed...");
        layer->output = output;
    }

    else if(layer->type == LayerType::ReLU) {
        output = matrix_relu(arena, input);
        layer->output =  output;
    }

    else if(layer->type == LayerType::Softmax) {
        output = matrix_softmax(arena, input);
        layer->output = output;
    }

    else {
        cout << "Unknown layer type, terminating...";
        exit(1);
    }

    return output;
}

Matrix* layer_backward(Arena* arena, Layer* layer, Matrix* grad_output) {
    Matrix* grad_input = nullptr;

    if(layer->type == LayerType::Dense) {
        Matrix* input_t = matrix_transpose(arena, layer->input);
        layer->dweights = matrix_mult(arena, input_t, grad_output);
        assert(layer->dweights && "dweights: matrix_mult failed...");
    }
}

Model* model_create(Arena* arena, int num_layers) {
    Model* model = PUSH_STRUCT(arena, Model);
    model->num_layers = num_layers;
    model->layers = PUSH_ARRAY(arena, Layer, num_layers);

    return model;
}

Matrix* model_forward(Arena* arena, Model* model, Matrix* input) {
    Matrix* a = input;

    for (int i = 0; i < model->num_layers; i++) {
        a = layer_forward(arena, &model->layers[i], a);
    }

    return a;
}

Matrix* model_backward(Arena* arena, Model* model, Matrix* grad_loss, float learning_rate);
