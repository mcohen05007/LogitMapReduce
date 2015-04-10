%% Example Script for Concensus MCMC estimation of Conditional Logit Choice Model with MapReduce
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated for use or adaptation, please cite as:
% Cohen, M. A. (2015). MapReduce for MCMC Logit Estimation [Computer software]. 
% Retrieved from http://www.macohen.net/software or
% https://github.com/mcohen05007/MapReduceLogit
clear 
clc

%% 
% Seed Random number geenrator and use the new SIMD-oriented Fast Mersenne
% Twister only for use with MATLAB 2015a or newer
% rng(0,'simdTwister')
rng(100,'twister')

%% Make BigData
Data =  dgp();
writetable(Data,'BigData.csv')

%% Run MapReduce
DS = datastore('BigData.csv');
DS.RowsPerRead = 5000;
Nobs = size(Data,1);
M = floor(Nobs/DS.RowsPerRead)+(mod(Nobs,DS.RowsPerRead)>0); % Nubmber of Data Shards
preview(DS)
mapper = @(d,ignore,intermKVStore) mcmcmapper(M,d,ignore,intermKVStore);
result = mapreduce(DS,mapper,@mcmcreducer);
OUT = readall(result);

%% Evluate Results
out.thetadraw = OUT.Value{1};
disp('Mean')
disp(mean(out.thetadraw))
subplot(1,3,1), hist(out.thetadraw(:,1),30)
subplot(1,3,2), hist(out.thetadraw(:,2),30)
subplot(1,3,3), hist(out.thetadraw(:,3),30)

