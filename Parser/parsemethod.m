function [ method] = parsemethod(name)
%parse Method will get the name from description and outputs the setting
%a struct that has detailed setting for the program.
% to modify provide other options to switch case.
method.maxbatchsize = 5000;
method.epochnum = 2;
method.usegpu = true;
method.tobin = true;
method.sccodemethod= 'hebian';
method.trainer.trainorder = 'permute';
method.trainer.bitselect = 'random';
method.verbose.exist = true;
%% Verbose library
%% Method Library : modify below

switch name
    %% FREDKIN METHODS
    case 'fredkin-random-flatrep'
        % uses fredkin gates and sorting optimization
        % randomly selects the conditioning bit
        % flattens the representation
        method.sccodemethod = 'fredkin';
        method.trainer.optim = 'sort';
        method.trainer.bitselect = 'random';
        method.flatten = true;
    case 'fredkin-random-origrep'
        % uses fredkin gates and sorting optimization
        % randomly selects the conditioning bit
        % does not flatten the representation
        method.sccodemethod = 'fredkin';
        method.trainer.optim = 'sort';
        method.trainer.bitselect = 'random';
        method.flatten = false;
    case 'fredkin-mixing-origrep'
        % uses fredkin gates and sorting optimization
        % randomly selects the conditioning bit
        % does not flatten the representation
        method.sccodemethod = 'fredkin';
        method.trainer.optim = 'sort';
        method.trainer.bitselect = 'mixing';
        method.flatten = false;
    case 'fredkin-mixing-fullrep'
        % uses fredkin gates and sorting optimization
        % selects the maximum discrimination conditioning bit
        % flattens the representation
        method.sccodemethod = 'fredkin';
        method.trainer.optim = 'sort';
        method.trainer.bitselect = 'mixing';
        method.flatten = true;
        %% GXOR METHODS
    case 'gxor-mixing-fullrep'
        method.sccodemethod = 'gxor';
        method.trainer.optim = 'gxor-optim';
        method.trainer.bitselect ='mixing';
        method.flatten = true;
    case 'gxor-random-fullrep'
        method.sccodemethod = 'gxor';
        method.trainer.optim = 'gxor-optim';
        method.trainer.bitselect ='random';
        method.flatten = true;
        
    case 'gxor-mixing-origrep'
        method.sccodemethod = 'gxor';
        method.trainer.optim = 'gxor-optim';
        method.trainer.bitselect ='mixing';
        method.flatten = false;
    case 'gxor-random-origrep'
        method.sccodemethod = 'gxor';
        method.trainer.optim = 'gxor-optim';
        method.trainer.bitselect ='random';
        method.flatten = false;
        case 'gxor-conv2d'
        method.sccodemethod = 'gxor-conv2d';
        method.trainer.optim = 'gxor-optim';
        method.trainer.bitselect ='mixing';
        method.flatten = false;
        %% Hybrid Methods
    case 'hyb-lmcl-full'
        method.tobin = false;
        method.sccodemethod = 'lmcl-hyb';
        method.trainer.optim = 'gxor-optim';
        method.trainer.bitselect ='random';
        method.flatten = true;
        %% MCL METHODS
    case 'maxuse'
        method.trainer.bitselect = 'maxused';
    case 'default'
        method.usegpu = true;
        method.sccodemethod= 'hebian';
        method.trainer.trainorder = 'permute';
        method.trainer.bitselect = 'random';
    case 'nogpu'
        method.usegpu = false;
        method.sccodemethod= 'hebian';
        method.trainer.trainorder = 'permute';
        method.trainer.bitselect = 'random';
        %% Mean Value
    case 'kl-mean'
        method.tobin = false;
        method.sccodemethod = 'kl-mean';
    otherwise
        error('method undefined');
end

end

