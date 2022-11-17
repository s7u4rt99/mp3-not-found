import os
import cv2

files = os.listdir('./Videos')

PATH = "ast_dataset/asl_alphabet_train"

print(files)


for i in files:
    letter = i.split('.')[0]
    vidcap = cv2.VideoCapture('./Videos/' + i)
    success,image = vidcap.read()
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Length of {i} is {length}")
    count = 0
    while vidcap.isOpened():
        success,image = vidcap.read()
        path_to_write = './Frames/' + letter[:-1]
        os.makedirs(path_to_write, exist_ok=True)
        if success:
            cv2.imwrite(f"{path_to_write}/{letter}_{count}.jpg", image)
            count += 1
        else:
            break