def train_smaller_models(image, model_save_name, helper=0):
    if helper==0:
        image = F.pad(image, (0, 28, 0, 0))
    elif helper==1:
        image = F.pad(image, (28, 0, 0, 0))
    height, width = image.shape[1], image.shape[2]

    coords = [[i, j] for i in range(height) for j in range(width)]
    intensities = [image[:, i, j].item() for i in range(height) for j in range(width)]
    coords = torch.tensor(coords, dtype=torch.float32)
    intensities = torch.tensor(intensities, dtype=torch.float32)

    model = sMLP(32,64, 1)
    #model = MLP()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs.squeeze(), intensities)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    
    torch.save(model.state_dict(), model_save_name)

    model.eval()
    with torch.no_grad():
        pred_intensities = model(coords)
        pred_image = pred_intensities.reshape(height, width).numpy()

def extract_and_concat_weights(model_paths, concat=False):
    concatenated_weights = []
    for path in model_paths:
        model = sMLP(32,64, 1)
        model.load_state_dict(torch.load(path))
        model_weights = []
        for param in model.parameters():
            model_weights.append(param.data.view(-1))
        concatenated_weights.append(torch.cat(model_weights))
    if not concat:
        return torch.stack((concatenated_weights))
    return torch.cat(concatenated_weights)