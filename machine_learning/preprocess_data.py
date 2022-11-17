import os
import cv2
import numpy as np
from tqdm import tqdm

TRAIN_PATH = 'ast_dataset/asl_alphabet_train'
TEST_PATH = 'ast_dataset/asl_alphabet_test'

dir_paths = os.listdir('ast_dataset/asl_alphabet_train')
dir_paths.sort()

mean_r, mean_b , mean_g = 0, 0, 0
std_r, std_b, std_g = 0, 0, 0

running_r, running_b, running_g = 0, 0, 0
std_running_r, std_running_b, std_running_g = 0, 0, 0

count = 0 
for idx, dir_path in tqdm(enumerate(dir_paths), desc='Preprocessing train data', total=len(dir_paths)):
    images_in_dir = os.listdir(os.path.join(TRAIN_PATH, dir_path))
    os.makedirs(f"ast_dataset/preprocessed_images/{dir_path}", exist_ok=True)
    for index, file in tqdm(enumerate(os.listdir(f"ast_dataset/asl_alphabet_train/{dir_path}")), desc=f"Preprocessing {dir_path}", total=len(images_in_dir)):
        img = cv2.imread(f"ast_dataset/asl_alphabet_train/{dir_path}/{file}")
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        count += 1
        cv2.imwrite(f"ast_dataset/preprocessed_images/{dir_path}/{dir_path + str(index)}.jpg", img)
        
print("Done preprocessing train data")

mean_r = np.mean(running_r/(count))
mean_b = np.mean(running_b/(count))
mean_g = np.mean(running_g/(count))

print(f"Mean R: {mean_r}, Mean B: {mean_b}, Mean G: {mean_g}")
