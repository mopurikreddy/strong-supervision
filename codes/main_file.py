import os
import siamese_data_prep
import siamese_regression_mkr_mlp as siamese




# Calling Densecap script to evaluate features
'''
 Script for getting, meanpooling and normalizing densecap features is in /data2/hrishikesh16/densecap/save_encoded_feats.py. Needs to be 
 run from that base path only. The normalized feats are stored in final_files base path under Queries_meanpooled_normalized / 
 References_meanpooled_normalized folders
'''

    # The database for which this script is run is decided inside that script by the variable *model_file_path*. Choosing the appropriate 
    # path (both paths provided inside that script) will run the feature extraction for the corresponding database.
main_dir = os.getcwd()
os.chdir('./densecap/')
os.system('python save_encoded_feats.py')
os.chdir(main_dir)

# Prepare pairs for siamese training
# 5 fold split etc etc
db = 'rPascal'
siamese_data_prep.main(db)

# Run code for siamese training
base_path = '../data/training_files/training_inputs'
train_type = "Final_feats"                      # Experiment type
train_split = '1'                               # Split number of cross-fold validation

# Loading inputs
X_train1, X_train2, Y_train = siamese.load_inputs(base_path, db, train_type, train_split)

# Compiling and training model
model_name = siamese.compile_train(X_train1, X_train2, Y_train, train_split)

# Getting intermediate representations after training
final_rep_query_file_path, final_rep_all_file_path = siamese.get_intermediate_rep(model_name, base_path, db, train_type)

# Run matlab file giving nDCG score
# Run the Matlab file eval_retrieval_copy2.m to check the nDCG score
# Make sure you're loading the correct Final_reps files in the Matlab file and also set the correct range1 and range2 for test split
