import torch

def flatten_model_weights(model):
    flat_weights = []
    for layer in model.layers:
        flat_weights.append(layer.weight.data.view(-1))
        flat_weights.append(layer.bias.data.view(-1))
    return torch.cat(flat_weights)

def unflatten_weights(flat_weights, model):
    external_parameters = {}
    offset = 0
    for i, layer in enumerate(model.layers):
        weight_shape = layer.weight.shape
        bias_shape = layer.bias.shape
        weight_numel = layer.weight.numel()
        bias_numel = layer.bias.numel()

        external_parameters[f"layers.{i}.weight"] = flat_weights[offset:offset + weight_numel].view(weight_shape)
        offset += weight_numel
        external_parameters[f"layers.{i}.bias"] = flat_weights[offset:offset + bias_numel].view(bias_shape)
        offset += bias_numel

    return external_parameters