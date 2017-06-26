%% Function to calculate nDCG

function val = nDCG(rlist,at_param, method)
    ordered = flipud(sort(rlist));
    
    if size(rlist,1)<at_param
       at_param = 0; 
    end
    
    if at_param~=0     
        rlist = rlist(1:at_param);
        ordered = ordered(1:at_param);
    end
    
    if method == 1
        % Method 1 to calculate DCG
        den = log2(2:size(rlist,1))';
        dcg = rlist(1) + sum(rlist(2:end)./den);
        
%         ordered = flipud(sort(rlist));
        idcg = ordered(1) + sum(ordered(2:end)./den);
        
        val = dcg/idcg;
%         val = dcg;    
    
    else if method ==2
            % Method 2 to calculate DCG
            den = log2(2:size(rlist,1)+1)';
            dcg = sum((2.^rlist-1)./den);
            
%             ordered = flipud(sort(rlist));
            idcg = sum((2.^ordered-1)./den);
            
            val = dcg/idcg;
%             val = dcg;
        else
            disp('ERROR! Please select method 1 or 2');
            val = NaN;
        end
    end
end