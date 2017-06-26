"""
This file accesses images that need to be run using "run_model.lua" and calls them using os.system call.
When lua file is run, it stores encoded features and the indices in order of highest scores (of dense captions)
first in temp files inside ./encoded_feats.
The temp files are renamed to match the image name before calling run_model on the next image
"""


import os
import shutil
import sys
import numpy as np
import subprocess
import torchfile as t


def run_model(model_file_path = './run_model_pascal.lua', q_ref = 'Queries'): 
    if('pascal' in model_file_path):
        db = 'rPascal'
    else:
        db = 'rImageNet'
    images_folder = os.path.join('../../data/Databases/', db, q_ref)
    images = sorted(os.listdir(images_folder))
    # Path for storing intermediate torchfile output
    dest_folder_temp = './encoded_feats/'
    # Path for storing numpy converted output
    dest_folder = os.path.join('./encoded_feats/', db, 'full', q_ref)

    # 4--> rPascal | 5--> rImagenet
    if(db == 'rPascal'):
        clip  = 4
    elif(db == 'rImageNet'):
        clip = 5


    for i,image in enumerate(images):
        print(i)
        os.system(os.path.join('th '+ model_file_path + ' -input_image ' + images_folder, image))
        inds = t.load(os.path.join(dest_folder_temp, 'temp_inds.t7'))
        inds = inds['inds']
        feats = t.load(os.path.join(dest_folder_temp, 'temp_feats.t7'))
        feats = feats['feats']
        np.save(os.path.join(dest_folder, image[:len(image)-clip]+'_feats'), feats)
        np.save(os.path.join(dest_folder, image[:len(image)-clip]+'_inds'), inds)


def extract_features(db = 'rPascal', q_ref = "Queries"):
    """
    Inputs : inds.npy and feats.npy of extracted Densecap.
    Outputs: Picks out only those indices in inds and puts in final npy
    """
    files_folder = os.path.join('./encoded_feats/', db, 'full/', q_ref)
    files = sorted(os.listdir(files_folder))
    feats = files[::2]
    inds = files[1::2]

    dest_folder = os.path.join('../../data/training_files/model_outputs/', db, 'bord/', q_ref + '_meanpooled_normalized')
    for i in range(len(feats)):
        feat = np.load(os.path.join(files_folder, feats[i])).squeeze()
        ind = np.load(os.path.join(files_folder, inds[i])).squeeze()
        final_feat = np.asarray([feat[j-1] for j in ind]).squeeze()
        final_feat = np.mean(final_feat[:5,:], axis = 0)                # Meanpooling top 5 
        final_feat = final_feat/np.linalg.norm(final_feat)              # Normalizing meanpooled features
        np.save(os.path.join(dest_folder, feats[i]), final_feat)



if(__name__ == '__main__'):
    model_file_path = './run_model_pascal.lua'       # For Pascal images
#    model_file_path = './run_model_imagenet.lua'       # For ImageNet images

    if('pascal' in model_file_path):
        db = 'rPascal'
    else:
        db = 'rImageNet'

    print("Running model for Query images")
    run_model(model_file_path, q_ref = "Queries")
    print("Extracting features for Query images")
    extract_features(db, q_ref = 'Queries')                         # rPascal/rImageNet     Queries/References


    print("Running model for Reference images")
    run_model(model_file_path, q_ref = "References")
    print("Extracting features for Reference images")
    extract_features(db, q_ref = 'References')
