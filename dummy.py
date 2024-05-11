from architectures import HyperNetwork
hypernet = HyperNetwork(1)
predicted_weights = hypernet(input)
dummy_input = torch.randn(1, input_size, requires_grad=True)
model_output = model(dummy_input, predicted_weights)
model_output.backward()
print(dummy_input.grad)  # Should not be None if gradients are flowing
