import json
import os
import shutil
from tqdm import tqdm

path_to_project_folder = "/Users/kuba/Desktop/assigns"
path_to_data_folder = os.path.join(path_to_project_folder, "data")
path_to_images_folder = os.path.join(path_to_data_folder, "images")
path_to_labels_folder = os.path.join(path_to_data_folder, "labels")

os.makedirs(path_to_images_folder, exist_ok=True)
os.makedirs(path_to_labels_folder, exist_ok=True)

img = "img_"
height, width = 640, 640

id_map = {
    "X1-Y1-Z2": 0, 'X1-Y2-Z1': 1, 'X1-Y2-Z2': 2,
    'X1-Y2-Z2-CHAMFER': 3, 'X1-Y2-Z2-TWINFILLET': 4,
    'X1-Y3-Z2': 5, 'X1-Y3-Z2-FILLET': 6,
    'X1-Y4-Z1': 7, 'X1-Y4-Z2': 8,
    'X2-Y2-Z2': 9, 'X2-Y2-Z2-FILLET': 10
}

cont_img, cont_lab = 0, 0

def convert_id_to_num(id):
    if id in id_map:
        return id_map[id]
    print(f"ERROR: no matching of id {id}")
    exit()

def normalize_coordinates(coords):

    return [coord / width if i % 2 == 0 else coord / height for i, coord in enumerate(coords)]

def create_txt_annotation(path_json, path_txt_folder):
    global cont_lab
    new_file_name = img + str(cont_lab) + ".txt"
    new_label_path = os.path.join(path_txt_folder, new_file_name)

    with open(path_json, 'r') as file:
        data = json.load(file)

    with open(new_label_path, 'w') as annotation_file:
        for key in data:
            y = data[key]['y']
            bbox_3d_pixel_space = data[key]['3d_bbox_pixel_space']
            
            coords = [point for sublist in bbox_3d_pixel_space[:4] for point in sublist]
            
            norm_coords = normalize_coordinates(coords)
            
            id_num = convert_id_to_num(y)
            annotation_line = f"{id_num} " + " ".join(map(str, norm_coords))
            
            annotation_file.write(annotation_line + "\n")
    
    cont_lab += 1

def save_img(img_file_path, path_to_images):
    global cont_img
    new_file_name = img + str(cont_img) + '.jpg'
    new_image_path = os.path.join(path_to_images, new_file_name)
    shutil.copyfile(img_file_path, new_image_path)
    cont_img += 1

pbar = tqdm(os.listdir(path_to_project_folder))

for folder_name in pbar:
    if "assign" in folder_name:
        folder_path = os.path.join(path_to_project_folder, folder_name)
        for scene_name in os.listdir(folder_path):
            if "scene" in scene_name:                           
                scene_path = os.path.join(folder_path, scene_name)
                json_path_list = []
                img_path_list = []
                
                for file_name in os.listdir(scene_path):
                    file_path = os.path.join(scene_path, file_name)
                    if file_name.endswith(".json"):
                        json_path_list.append(file_path)
                    if file_name.startswith("view=") and file_name.endswith(".jpeg") and all(x not in file_name for x in ["_vertices", "_depth", "_depth_plane", "_bbox"]):
                        img_path_list.append(file_path)
                
                json_path_list.sort()
                img_path_list.sort()
                
                if len(json_path_list) != len(img_path_list):
                    print("ERROR: DO NOT FIND CORRESPONDENT IMAGE FOR AN ANNOTATION")
                    exit()
                
                for jpath in json_path_list:
                    create_txt_annotation(jpath, path_to_labels_folder)
                for ipath in img_path_list:
                    save_img(ipath, path_to_images_folder)

print(f"Processed {cont_img} images and {cont_lab} annotations")


