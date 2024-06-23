from ADL4CV.architectures import sMLP
import torch
import json
import mcubes
import trimesh

def apply_sdf(model, len_dims_x, len_dims_y, len_dims_z, coordinates, weights=None, biases=None):
    model.eval()
    if weights is not None and biases is not None:
        sdf = model(coordinates, weights, biases)
    else:
        sdf = model(coordinates)
    sdf = sdf.view(len_dims_x, len_dims_y, len_dims_z)
    return sdf.detach().numpy()

def generate_coordinates(step, scale_x, scale_y, scale_z):
    coordinates = [(-scale_x + x * step, -scale_y + y * step, -scale_z + z * step)
                   for x in range(int(2 * scale_x / step))
                   for y in range(int(2 * scale_y / step))
                   for z in range(int(2 * scale_z / step))]
    return torch.tensor(coordinates, dtype=torch.float32)

def render_and_store_models(step, scale_x, scale_y, scale_z, model_path=None, weights=None, biases=None):
    with open("config.json") as json_file:
        json_file = json.load(json_file)
        INR_model_config = json_file["INR_model_config_3D"]
    
    dimensions_x = int(2 * scale_x / step)
    dimensions_y = int(2 * scale_y / step)
    dimensions_z = int(2 * scale_z / step)
    
    coordinates = generate_coordinates(step, scale_x, scale_y, scale_z)
    
    model = sMLP(42, INR_model_config)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    sdf = apply_sdf(model, dimensions_x, dimensions_y, dimensions_z, coordinates, weights, biases)
    
    vertices, triangles = mcubes.marching_cubes(-sdf, 0)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    N = scale_x / step
    print(type(mesh.vertices))
    mesh.vertices = (mesh.vertices / N - 1) + 0.5 / N
    mesh.export(f"./sdf_pipeline.obj")
