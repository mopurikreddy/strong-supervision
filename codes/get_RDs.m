function RDs=get_RDs(txt_path,K)
%% Reads the region descriptors for displaying in figure
%  for retrieval engine
%

fid=fopen(txt_path,'r');
% RDs='DenseCap RDs:';
RDs={};
for i=1:K
%     RDs=strcat(RDs, strcat(fgets(fid),';'));
    RDs{i}=fgets(fid);
end
fclose(fid);
