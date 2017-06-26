%% This is the file to compute nDCG and see the retrieval results
%  Requires the distances and the groundtruth files


% Depending on the cross validation split, range1 and range2 fields (that determine the test set) need to be changed along with F_q and F files
% that are loaded.
% Cross val split       range1          range2
%       1               41              50
%       2               1               10
%       3               11              20
%       4               21              30
%       5               31              40


%% add the paths
addpath(genpath('~/Documents/MATLAB/npy-matlab-master/'))
addpath(genpath('/data2/mkreddy/BoRD/final_files/codes/'))

%% loads the query and reference image names

db='rPascal';

q_imgs=dir(['../data/Databases/' db '/Queries/']);
q_imgs=q_imgs(3:end);
r_imgs=dir(['../data/Databases/' db '/References/']);
r_imgs=r_imgs(3:end);

for i=1:numel(q_imgs)
    q_imgs(i).name = strcat(q_imgs(i).name(1:end-3), 'txt');
end

for i=1:numel(r_imgs)
    r_imgs(i).name = strcat(r_imgs(i).name(1:end-3), 'txt');
end
disp(r_imgs(1).name)
%% loads the GT. Reference list and median annotations
db2 = 'rPascal';

ref_path=['../data/Databases/' db2 '/Reference_Lists/'];
load(['../data/Databases/' db2 '/MedianAnnot.mat'])
dataset_size = numel(r_imgs);
 
%% construct the matrix GT_sim with ground truth relevances 3, 2, 1, 0 (similarity) to all the queries

GT_sim=single(zeros(numel(q_imgs),dataset_size));
GT_sim(1:end,1:end) = -1;

for i=1:numel(q_imgs)
    load(strcat(ref_path,q_imgs(i).name(1:end-4),'_refs.mat'))
    for r=1:size(refNames)
        im=refNames(r,:);   
 	ind=find_ind(r_imgs,strcat(im(1:end-4),'.txt'));  % For Pascal dataset 
        GT_sim(i,ind)=medAnnot{i}(r);
    end
    disp(i);
end

%% load the distances (similarities) computed by us
F_q=readNPY(['../data/training_files/training_inputs/rPascal/Final_feats3_normalized_inputs/Final_reps/Final_rep_queries_euc_mlp_pascal1024_margin1_loss4_2000epochs_notskewed_121k55_henormal_TRIAL1.npy']);
F=readNPY(['../data/training_files/training_inputs/rPascal/Final_feats3_normalized_inputs/Final_reps/Final_rep_all_euc_mlp_pascal1024_margin1_loss4_2000epochs_notskewed_121k55_henormal_TRIAL1.npy']);
disp('done')

%% normalizing and computing the cosine distances
F_n=normr(F);
F_q_n=normr(F_q);

distances = pdist2(F_q_n,F_n,'euclidean');

%% Sorting the database images and computing ranked list
[~,gt_ranks]=sort(GT_sim,2,'descend');
[a,pred_ranks]=sort(distances,2,'ascend');
disp('loaded.')


%% computing the nDCG metric WITHOUT DISTRACTOR IMAGES

rlist=single(zeros(50,dataset_size));
rlist(1:end,1:end) = -1;
% contains the relevances for the retreived (ranked) images

for i=1:numel(q_imgs)
    rlist(i,1:end-1)=GT_sim(i,pred_ranks(i,2:end));
    % remove query itself from the retrieved images
end
K=[5 10 20 30 40 50 60 70 80 90 100];
metric_wo=single(zeros(1,numel(K)));

range1 = 41;
range2 = 50;
for i=1:numel(K)
    temp_metric=single(zeros(1,numel(q_imgs)));	
    for q=range1:range2 	
        newlist=rlist(q,rlist(q,:)~=-1);
        temp_metric(q)=nDCG(newlist',K(i),2);
    end
    [val,ind] = min(temp_metric);
metric_wo(i)=mean(temp_metric(range1:range2));
end
disp(metric_wo)
%figure,plot(K,metric_wo);

