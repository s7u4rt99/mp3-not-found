import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths


image_paths = list(paths.list_images("ast_dataset/preprocessed_images"))

data = pd.DataFrame()

labels = []

for idx, image_path in tqdm(enumerate(image_paths), desc="Creating dataframe"):
    label = image_path.split(os.path.sep)[-2]
    data.loc[idx, "image_path"] = image_path
    labels.append(label)
    
labels = np.array(labels)
lb = LabelBinarizer()
lb.fit(labels)
print(lb.classes_)
labels = lb.transform(labels)

for i in range(len(labels)):
    data.loc[i, 'label'] = int(np.argmax(labels[i]))
    
data = data.sample(frac=1).reset_index(drop=True)

print(data.head())

data.to_csv("ast_dataset/data_final.csv", index=False)


# pickle the binarized labels
print('Saving the binarized labels as pickled file')
joblib.dump(lb, 'lb.pkl')