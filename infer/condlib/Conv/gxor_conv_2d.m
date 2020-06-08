function [ bin,diffbin] = gxor_conv_2d( bin,rel_inds_cell,path_cell )
%GXOR_CONV_2D Summary of this function goes here
%   Detailed explanation goes here
diffbin = gpuArray.false(size(bin));
for ch = 1: size(bin,3)
    rel_inds_this = rel_inds_cell{ch};
    path_this = path_cell{ch};
    
    if isempty(rel_inds_this)
        continue
    else
        diffbin(:,:,ch,:) = true;
    end
    for j = 1 : size(rel_inds_this,1)
        tempbin = bin(:,:,rel_inds_this(j,3),:);
        tempbin = xor(tempbin,~path_this(j));
        tempbin = circshift(tempbin,-rel_inds_this(j,1),1);
        tempbin = circshift(tempbin,-rel_inds_this(j,2),2);
        if rel_inds_this(j,1)>0
            tempbin(end-abs(rel_inds_this(j,1))+1:end,:,:,:) = false;
        else
            if rel_inds_this(j,1)~=0
                tempbin(1:abs(rel_inds_this(j,1)),:,:,:) = false;
            end
        end
        if rel_inds_this(j,2)>0
            tempbin(:,end-abs(rel_inds_this(j,2))+1:end,:,:) = false;
        else
            if rel_inds_this(j,2)~=0
                tempbin(:,1:abs(rel_inds_this(j,2)),:,:) = false;
            end
        end
        diffbin(:,:,ch,:) = and(diffbin(:,:,ch,:),tempbin);
    end
    
end
bin = xor(bin,diffbin);
end

