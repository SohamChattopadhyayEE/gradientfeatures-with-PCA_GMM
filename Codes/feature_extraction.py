import os
import numpy as np
from features.gradient_features import choose_features

#path_directory = '/content/drive/MyDrive/Dataset_Educazone_Test/'
path_directory = 'E:/Job_Internships/Educazone/Dataset_Educazone_Test/'#input("Enter the directory of data: \n")
filename = path_directory#input("Enter the path where the data would be saved: \n")
directories = os.listdir(path_directory)

i = 0
label = []
features = []
for directory in directories : 
    file_path = path_directory + directory
    print(file_path)
    file_list = os.listdir(file_path)
    for id in file_list : 
        image = file_path+ '/' + id
        print('Image path : ', image)
        feature = choose_features(image, 'hog')
        features.append(feature)
        label.append(i)
    i += 1

filename = filename + '/features.npy'
np.save(filename, features)
arr = np.load(filename)