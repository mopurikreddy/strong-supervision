from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1234) # for reproducibility
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense,  Dropout, Activation, RepeatVector, Flatten
from keras.layers import Input, Embedding, LSTM, Dense, merge, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from scipy.io import loadmat, savemat
import theano
import keras
import theano.tensor as T
import h5py
from keras.utils.io_utils import HDF5Matrix
import os
from keras import backend as K
import pdb
import json
dest='../data/training_files/training_inputs/'
dataset_type = 'rPascal/'
model_type='Final_feats/'


def cosine_distance(vects):
    '''  cosine similarity between two vectors  '''
    yt,yp = vects
    ytn=yt / K.sqrt((yt*yt).sum(axis=-1,keepdims=True))		# Changed T.sqrt to K.sqrt. Problem loading model json file otherwise
    ypn=yp / K.sqrt((yp*yp).sum(axis=-1,keepdims=True))		# Changed T.sqrt to K.sqrt. Problem loading model json file otherwise
    dp=(ytn*ypn).sum(axis=-1,keepdims=True)
    return (1-dp)/2

def euclidean_distance(vects):
    x, y = vects
    y=y / K.sqrt((y*y).sum(axis=-1,keepdims=True))		# Changed T.sqrt to K.sqrt. Problem loading model json file otherwise
    x=x / K.sqrt((x*x).sum(axis=-1,keepdims=True))		# Changed T.sqrt to K.sqrt. Problem loading model json file otherwise
    p = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    return p

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    #return K.mean((1-y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))	# d^2 loss
    #return K.mean((1-y_true) * y_pred + y_true * K.maximum(margin - y_pred, 0))			#   d loss
    #return K.mean(y_true * K.square(y_pred) + K.maximum(0,(1-y_true)) * K.square(K.maximum(margin - y_pred, 0)))# non-binary relevance loss1
    #return K.mean(y_true * (y_pred) + K.maximum(0,(1-y_true)) * K.maximum(margin - y_pred, 0))			# non-binary relevance loss2
    #return K.mean(y_true * (y_pred) + K.maximum(0,(1-y_true)) * K.square(K.maximum(margin - y_pred, 0)))	# non-binary relevance loss3
    return K.mean(K.square(y_true) * (y_pred) + K.maximum(0,(1-y_true)) * K.maximum(margin - K.square(y_pred), 0))	# non-binary relevance loss4

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


###################

# 	Loading inputs for training
def load_inputs(base_path, db = "rPascal", train_type = "Final_feats", train_split = '1'):
    X_train1 = np.load(os.path.join(base_path, db, train_type, "X_train1_notskewed_"+train_split+".npy"))
    X_train2 = np.load(os.path.join(base_path, db, train_type, "X_train2_notskewed_"+train_split+".npy"))
    Y_train = np.load(os.path.join(base_path, db, train_type, "Y_train_notskewed_"+train_split+".npy"))

    #### Shuffling inputs
    np.random.seed(1234)
    np.random.shuffle(X_train1)
    np.random.seed(1234)
    np.random.shuffle(X_train2)
    np.random.seed(1234)
    np.random.shuffle(Y_train)

    return X_train1, X_train2, Y_train


dim0 = 1024
dim1 = 2048
dim2 = 2048
dim3 = 512
dim4 = 512


######## FC 1
l0 = Dense(dim0, input_dim = 1024, init = 'he_normal', activation = 'relu', name = 'h0')
l1 = Dense(dim1, input_dim = dim0, init = 'he_normal', name = 'h1')
l2 = Dense(dim2, input_dim = dim1, init = 'he_normal', name = 'h2')
l3 = Dense(dim3, input_dim = dim2, init = 'he_normal', activation = 'relu', name = 'h3')
l4 = Dense(dim4, input_dim = dim3, init = 'he_normal', activation = 'relu', name = 'h4')

############ Combining weights with individual features to get F1
X1 = Input(shape=(1024,))
X11 = l0(X1)
X12 = l1(X11)
X13 = l2(X12)
X14 = l3(X13)
X15 = l4(X14)


############ Combining weights with individual features to get F2
X2 = Input(shape=(1024,))
X21 = l0(X2)
X22 = l1(X21)
X23 = l2(X22)
X24 = l3(X23)
X25 = l4(X24)



####### Final euclidean distance between Final representations F1 and F2
los='euc'
forward = Lambda(euclidean_distance, output_shape=(1,))
distance = forward([X15, X25])

########### Model compile
def compile_train(X_train1, X_train2, Y_train, train_split = '1'):
    model = Model(input = [X1,X2], output = [distance])
    sgd=SGD(lr=0.001,momentum=0.9, decay=1e-06,clipvalue=0.10)		# clipnorm = 10
    model.compile(loss=contrastive_loss, optimizer=sgd,metrics=[])

    ########## Model training

    num_epochs=2000

    earlyStopping=EarlyStopping(monitor='val_loss', patience=10, verbose=2,mode='auto')
    model.fit(
	[X_train1, X_train2],
	Y_train,
	batch_size=600,
	nb_epoch=num_epochs,
	verbose=1,
	callbacks=[earlyStopping],
	validation_split = 0.1,
	shuffle=1,
    )
    name = 'siamese_mlp_'+los+str(dim1)+'_margin1_loss4_'+str(num_epochs)+'epochs_notskewed_121k55_henormal'+train_split
    with open(os.path.join(dest, dataset_type, model_type, name + '.json'), 'w') as f:
        f.write(model.to_json(indent=2))
    model.save_weights(os.path.join(dest, dataset_type, model_type, name + '.h5'),overwrite=True)
    print('model saved.')
    return name


# Getting intermediate representations from trained model

### Comment training and testing parts of code before running this. model.compile remains "un-commented"
def get_intermediate_rep(model_name, base_path = '../data/training_files/training_inputs', db = 'rPascal', model_type = 'Final_feats'):
    model.load_weights(os.path.join(base_path, db, model_type, model_name + '.h5'))

    rep_query = K.function([X1],[X12])
    rep_all = K.function([X2],[X22])

    X_query = np.load(os.path.join(base_path, db, model_type, "X_" + db.lower() + "_queries.npy"))
    X_all = np.load(os.path.join(base_path, db, model_type, "X_" + db.lower() + "_all.npy"))

    final_rep_query = rep_query([X_query])
    final_rep_all = rep_all([X_all])

    final_rep_query = final_rep_query[0]
    final_rep_all = final_rep_all[0]
    print('Got rep')
    if(not os.path.exists(base_path, db, model_type, 'Final_reps')):
        os.mkdir(os.path.join(base_path, db, model_type, 'Final_reps'))

    final_rep_query_file_path = os.path.join(base_path, db, model_type, 'Final_reps/', 'Final_rep_queries_' + los + '_mlp_' + db.lower() + str(dim2)+'_margin1_loss4_'+str(num_epochs)+'epochs_notskewed_121k55_henormal_2ndlayer_' + train_split)

    final_rep_all_file_path = os.path.join(base_path, db, model_type, 'Final_reps/', 'Final_rep_all_' + los + '_mlp_' + db.lower() + str(dim2)+'_margin1_loss4_' + str(num_epochs) + 'epochs_notskewed_121k5_henormal_2ndlayer_' + train_split)

    np.save(final_rep_query_file_path, final_rep_query)
    np.save(final_rep_all_file_path, final_rep_all)
    
    return final_rep_query_file_path, final_rep_all_file_path
