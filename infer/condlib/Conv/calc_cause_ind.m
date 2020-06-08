function [bin, rel_inds,paths] = calc_cause_ind(bin,anch_ind,method)
%% [bin, rel_inds,paths] = calc_cause_ind(bin,anch_ind,chan_ind)
% calculates the variables that should operate on the
% anchor variable in a 4d tensor, to minimize the entropy maximally.
% ========================================================================
% Inputs :
% ------------------------------------------------------------------------
% bin:
%
%
% bin is the 4d data tensor containing logical variables where the first 2 dimensions
% are considered stationary. 
% The 4th dimension is the sample dimension and the 3rd is the channel
% dimension
% ------------------------------------------------------------------------
% anch_ind :
% 
% 
% anch_ind is the spatial location to infer structure from and is the 2d
% coordinate.
% 
% 
% ========================================================================
% Outputs:
% bin remains unchanged
% 
% ------------------------------------------------------------------------
% rel_inds:
%
% rel_inds the relative location of the causal variables in the spatial
% location , and absolute locations of the channel dimension.
% 
% the size is n by 3.
% ------------------------------------------------------------------------
% paths is n by 1 where it shows the paths of conditional dependencies.


switch method.trainer.bitselect
    case 'mixing'
          [bin,rel_inds,paths]  =choosebitConv(bin,anch_ind,method,0);
          if ~isempty(rel_inds)
          rel_inds(:,1:2) = (rel_inds(:,1:2)- anch_ind(:,1:2));
          end
          
       
      
end
end


