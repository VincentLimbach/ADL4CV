from ADL4CV.architectures import sMLP
import torch
import json
import mcubes
import trimesh
import numpy as np

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
    mesh.vertices = (mesh.vertices / N - np.array([[1, 2, 1]]))
    mesh.export(f"./sdf_pipeline.obj")

def normalize(coords):
    coords -= np.mean(coords, axis=0, keepdims=True)
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
    coords -= 0.45
    return coords


def generate_merged_gt_3D(obj_idx_1, obj_idx_2, offsets, output_file_path):
    vertices = []
    faces = []
    x_1, y_1, z_1, x_2, y_2, z_2 = offsets

    with open(f"ADL4CV/data/gt_data/shrec_16_original/armadillo/train/T{obj_idx_1}.obj", 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = list(map(float, parts[1:4]))
                vertex[0:3] = np.array(vertex) #+ np.array([x_1, y_1, z_1])
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.split()
                face = []
                for part in parts[1:]:
                    face.append(int(part.split('/')[0]))
                faces.append(face)
        n_vertex_1 = len(vertices)
    
    vertices = (normalize(vertices) + np.array([[x_1, y_1, z_1]])).tolist()

    with open(f"ADL4CV/data/gt_data/shrec_16_original/armadillo/train/T{obj_idx_2}.obj", 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = list(map(float, parts[1:4]))
                vertex[0:3] = np.array(vertex) #+ np.array([x_2, y_2, z_2])
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.split()
                face = []
                for part in parts[1:]:
                    face.append(int(part.split('/')[0]) + n_vertex_1)
                faces.append(face)

    vertices[n_vertex_1:] = (normalize(vertices[n_vertex_1:]) + np.array([[x_2, y_2, z_2]])).tolist()

    #vertices = np.array(vertices) * 10/9

    with open(output_file_path, 'w') as output_file:
        for vertex in vertices:
            output_file.write("v {0} {1} {2}\n".format(vertex[0],vertex[1],vertex[2]))

        for face in faces:
            output_file.write("f {0} {1} {2}\n".format(face[0],face[1],face[2]))              
