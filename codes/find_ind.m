function ind=find_ind(r_imgs,ref)
%% this functio is used in the retrieval engine
for i=1:numel(r_imgs)
    if strcmp(r_imgs(i).name,ref)
        ind=i;
        break
    end
end
