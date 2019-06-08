# This code save already computed C3D features into 32 (video features) segments.
# We assume that C3D features for a video are already computed. We use default settings for computing C3D features, 
# i.e., we compute C3D features for every 16 frames and obtain the features from fc6.

import numpy as np
import os
import array


def readBinary(path):
    f = open(path, 'rb')
    s = array.array("i") # int32
    s.fromfile(f, 5)
    m = s[0]*s[1]*s[2]*s[3]*s[4]
    data_aux = array.array("f")
    data_aux.fromfile(f, m)
    data = np.array(data_aux.tolist())
    return s,data



C3D_path='Anomaly-Detection/C3D_Features'
C3D_path_seg='Anomaly-Detection/C3D_Features_Avg_py'

if not os.path.exists(C3D_path_seg):
    os.makedirs(C3D_path_seg)

all_folders = os.listdir(C3D_path)
all_folders.sort()
subcript='_C.txt'


x = 1
for ifolder in all_folders:
    folder_path = os.path.join(C3D_path , ifolder) 
    #folder_path is path of a folder which contains C3D features (for every 16 frames) for a paricular video.
    all_files = os.listdir(folder_path)
    all_files.sort()

    feature_vect = np.zeros((len(all_files) , 4096))
    for ifile in range(len(all_files)):
        file_path  = os.path.join(folder_path , all_files[ifile])
        s , data = readBinary(file_path)
        feature_vect[ifile] = data

    
    if np.sum(feature_vect) == 0:
        print("all data are zeros")
        exit()

    if np.sum( np.sum(feature_vect , 1) == 0  ):
        print("some rows are zeros")
        exit()
    
    if np.isnan(feature_vect).any():
        print("some values are missing")
    
    if np.isinf(feature_vect).any():
        print("some values are inf")
    

    # Now all is okay, time to store the features for 32 segments

    segment_features = np.zeros((32,4096))
    positions= np.round(np.linspace(0,len(all_files)-1,33))

    for iposition in range(len(positions) - 1):
        cur = int(positions[iposition])
        nxt = int(positions[iposition + 1])-1
        
        if cur >= nxt:
            temp_vect = feature_vect[cur]
        else:
            temp_vect = np.mean(feature_vect[cur:nxt+1] , 0)
        
        nrm = np.linalg.norm(temp_vect)

        if nrm == 0:
            print("normalization is wrong")
            exit()
        
        temp_vect=temp_vect/nrm
        segment_features[iposition] = temp_vect
    
    # save features file
    np.savetxt(os.path.join(C3D_path_seg , ifolder + subcript) , segment_features , fmt="%f")
    




