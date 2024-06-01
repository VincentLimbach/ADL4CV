import torch

def flatten_model_weights(model):
    flat_weights = []
    for layer in model.layers:
        flat_weights.append(layer.weight.data.view(-1))
        flat_weights.append(layer.bias.data.view(-1))
    return torch.cat(flat_weights)

def unflatten_weights(flat_weights, model):
    batch_size = flat_weights.shape[0]
    weights, biases = [], []
    
    offset = 0
    for layer in model.layers:
        weight_shape = (batch_size, *layer.weight.shape)
        bias_shape = (batch_size, *layer.bias.shape)
        weight_numel = layer.weight.numel()
        bias_numel = layer.bias.numel()
        
        layer_weights = flat_weights[:, offset:offset + weight_numel].view(weight_shape)
        offset += weight_numel
        layer_biases = flat_weights[:, offset:offset + bias_numel].view(bias_shape)
        offset += bias_numel
        
        weights.append(layer_weights)
        biases.append(layer_biases)
    return weights, biases

