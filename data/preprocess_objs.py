import os
import numpy as np

def parse_obj(file_path):
    vertices = []
    faces = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.split()
                face = []
                for part in parts[1:]:
                    face.append(int(part.split('/')[0]) - 1)
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def calculate_normals(vertices, faces):
    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        area = np.linalg.norm(face_normal) / 2.0  
        
        face_normal /= (2.0 * area)
        
        for idx in face:
            vertex_normals[idx] += face_normal
    return vertex_normals

def write_obj_with_normals(input_file_path, output_file_path, vertices, faces, vertex_normals):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        vertex_index = 0
        for line in input_file:
            if line.startswith('v '):
                output_file.write(line)
                normal = vertex_normals[vertex_index]
                output_file.write(f'vn {normal[0]} {normal[1]} {normal[2]}\n')
                vertex_index += 1
            elif line.startswith('f '):
                parts = line.split()
                face_indices = [int(part.split('/')[0]) for part in parts[1:]]
                new_face_parts = [f"{index}//{index}" for index in face_indices]
                new_face_line = f"f {' '.join(new_face_parts)}\n"
                output_file.write(new_face_line)
            else:
                output_file.write(line)

def process_obj_file(filepath):
    filename = os.path.splitext(filepath)[0]
    output_filepath = f"{filename}.xyz"
    
    with open(output_filepath, 'w') as output_file:
        with open(filepath, 'r') as obj_file:
            vertices = []
            normals = []
            
            for line in obj_file:
                if line.startswith("v "):
                    vertices.append(line.strip())
                elif line.startswith("vn "):
                    normals.append(line.strip())

            if len(vertices) != len(normals):
                print(f"Warning: Number of vertices and normals do not match in {filepath}")
                return
            
            for vertex, normal in zip(vertices, normals):
                vertex_data = vertex.split()[1:]
                normal_data = normal.split()[1:]
                output_line = " ".join(vertex_data + normal_data) + "\n"
                output_file.write(output_line)

def process_directory(start_directory):
    for root, _, files in os.walk(start_directory):
        for file in files:
            if file.endswith('_with_normals.obj'):
                continue
            if file.endswith('.obj'):
                filepath = os.path.join(root, file)
                output_file_path = os.path.splitext(filepath)[0] + '_with_normals.obj'
                vertices, faces = parse_obj(filepath)
                vertex_normals = calculate_normals(vertices, faces)
                write_obj_with_normals(filepath, output_file_path, vertices, faces, vertex_normals)
                process_obj_file(output_file_path)

start_directory = './data/shrec_16'
process_directory(start_directory)
