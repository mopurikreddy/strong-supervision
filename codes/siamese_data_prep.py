'''
Data preparation of Siamese network training for rPascal/rImagenet dataset
'''

import os, sys
import numpy as np
from scipy.io import loadmat, savemat
import random
import time
random.seed(1234)


# These global values will be set in main function when the Database for which this code is run is known.
# Densecap 5RD-mmd/meanpooled features filepath
rd_query_path = None
rd_ref_path = None
# Image Caption Show and tell Hidden State Features path
fic_query_path = None
fic_ref_path = None



def get_fused_vec(index, q_ref):
        '''
        Returns the complete fused vector in the order -> [RD, FIC] of the given index (represents file name when in sorted order).
        '''
        rd_ref_files = sorted(os.listdir(rd_ref_path))
        rd_query_files = sorted(os.listdir(rd_query_path))
        fic_query_files = sorted(os.listdir(fic_query_path))
        fic_ref_files = sorted(os.listdir(fic_ref_path))
        temp = []
        if(q_ref):                                                                      # Query Vectors
                vec = np.load(os.path.join(rd_query_path, rd_query_files[index]))
                vec = vec.squeeze()
                temp.extend(vec)                                                                # Appending BoRD vector
                temp.extend(np.load(os.path.join(fic_query_path, fic_query_files[index])))             # Appending FIC vector
        else:
                vec = np.load(rd_ref_path+rd_ref_files[index])                          # Reference Vectors
                vec = vec.squeeze()
                temp.extend(vec)                                                                # Appending BoRD vector
                temp.extend(np.load(fic_ref_path+fic_ref_files[index]))                         # FIC vector

        return np.asarray(temp,dtype = np.float32)


def prep_query_ref(db = 'rPascal', q_ref = 1):
    '''
    Prepares fused vectors for all query images or all reference images based on q_ref taf
    '''

    X_queries = []
    X_all = []

    if(q_ref == 1):
        iterations = 50         # For queries 
    else:                       # For all images
        if('pascal' in db.lower()):
            iterations = 1835
        else:
            iterations = 3353

    for i in range(iterations):	
	vec = get_fused_vec(i,q_ref)	
        if(q_ref == 1):
            X_queries.append(vec)
        else:
            X_all.append(vec)
    if(q_ref == 1):
        np.save(os.path.join("../data/training_files/training_inputs/", db, "Final_feats/X_" + db.lower() + "_queries"), X_queries)
    else:
        np.save(os.path.join("../data/training_files/training_inputs/", db, "Final_feats/X_" + db.lower() + "_all"), X_all)



def main(db = 'rPascal'):
    # Loading annotation scores (0,1,2,3) so that images with score 0 can be removed from positive examples of Siamese network input
    medannot_scores = loadmat(os.path.join('../data/Databases/', db, 'MedianAnnot.mat'))['medAnnot']
    medannot_scores = medannot_scores.squeeze()
    for i in range(50):
    	medannot_scores[i] = medannot_scores[i].squeeze()

    ref_path = os.path.join('../data/Databases/', db, 'Reference_Lists/')
    references = sorted(os.listdir(os.path.join('../data/Databases/', db, 'References/')))
   
    # Setting the global query and references path based on the parameter *db* 
    global rd_query_path, rd_ref_path, fic_query_path, fic_ref_path
    rd_query_path = os.path.join('../data/training_files/model_outputs/', db, 'bord/Queries_meanpooled_normalized/')
    rd_ref_path = os.path.join('../data/training_files/model_outputs/', db, 'bord/References_meanpooled_normalized/')

    # Image Caption Show and tell Hidden State Features path
    fic_query_path = os.path.join('../data/training_files/model_outputs/', db, 'fic/Queries_fc7_normalized/')
    fic_ref_path = os.path.join('../data/training_files/model_outputs/', db, 'fic/References_fc7_normalized/')

    rd_ref_files = sorted(os.listdir(rd_ref_path))
    rd_query_files = sorted(os.listdir(rd_query_path))
    fic_query_files = sorted(os.listdir(fic_query_path))
    fic_ref_files = sorted(os.listdir(fic_ref_path))

    ref_files = sorted(os.listdir(ref_path))		# Files end with .npy here

    X1 = []
    X2 = []
    Y = []


    prep_query_ref(db, q_ref = 1)       # Preparing fused vector for queries
    prep_query_ref(db, q_ref = 0)       # Preparing fused vector for references


    # Saving positive and negative training pairs CROSS-VALIDATION wise
    count = 0
    count_gt = 0
    query_numbers = list(range(40)) #+list(range(40,50))
    for i in query_numbers:     
	query_vec = get_fused_vec(i,1)
        refname = loadmat(os.path.join(ref_path, ref_files[i]))['refNames']
	refname = list(refname)	
        if('imagenet' in db.lower()):			
            for p in range(len(refname)):				# For rImageNet
		refname[p] = refname[p][0][0]			# For rImageNet
	    query_name = fic_query_files[i][:len(fic_query_files[i])-4]+'.jpg'	# For rImageNet
        else:
            query_name = fic_query_files[i][:len(fic_query_files[i])-4]+'.jpg'	# For rPascal

	# Removing files with 0 median annotation score from relevant images list
	inds_relevant = [p for p,q in enumerate(medannot_scores[i]) if q != 0]
	refname_relevant = [q for p,q in enumerate(refname) if p in inds_relevant]
	length_relevant = len(refname_relevant)
    
	for j in range(length_relevant):
		reference_loc = references.index(refname_relevant[j])
		rel_vec = get_fused_vec(reference_loc,0)
		X1.append(query_vec)
		X2.append(rel_vec)
		Y.append(medannot_scores[i][inds_relevant[j]])		# GT same as the Relevance score given (1/2/3)
		print(refname[j],medannot_scores[i][inds_relevant[j]])

	# Adding the non-relevant images
	# For skewed +ve and -ve distribution
	nonrel_set = list(set(refname)-(set(refname_relevant).union(set([query_name]))))
	length_irrelevant = len(nonrel_set)


	for nonrel_image in nonrel_set:
		reference_loc = references.index(nonrel_image)
		nonrel_vec = get_fused_vec(reference_loc, 0)
		X1.append(query_vec)
		X2.append(nonrel_vec)
		Y.append(0)					# Since 0 is for dissimilar images in Siamese network
	count += length_relevant+length_irrelevant

	# For non-skewed +ve and -ve distribution, i.e, adding hard -ves as remainders
	if(length_relevant > length_irrelevant):
		count_gt += 1
		remainder = length_relevant-length_irrelevant
		other_nonrel_set = list(set(references)-set(refname).union(set([query_name])))
		other_nonrel_images = random.sample(other_nonrel_set,remainder)
		for nonrel_image in other_nonrel_images:
			reference_loc = references.index(nonrel_image)
			nonrel_vec = get_fused_vec(reference_loc, 0)
			X1.append(query_vec)
			X2.append(nonrel_vec)
			Y.append(0)
		count += remainder


    X1 = np.asarray(X1,dtype='float32')
    X2 = np.asarray(X2,dtype='float32')
    Y = np.asarray(Y, dtype = 'uint8')
    print(X1.shape, X2.shape, Y.shape)
    
    train_type = 'Final_feats'          # Destination folder type for storing the training pairs. Change when creating new experiment
    np.save(os.path.join("../data/training_files/training_inputs/", db, train_type, "X_train1_notskewed_1"), X1)
    np.save(os.path.join("../data/training_files/training_inputs/", db, train_type, "X_train2_notskewed_1"), X2)
    np.save(os.path.join("../data/training_files/training_inputs/", db, train_type, "Y_train_notskewed_1"), Y)


