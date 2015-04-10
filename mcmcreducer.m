function mcmcreducer(~,intermValIter,outKVStore)
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated for use or adaptation, please cite as:
% Cohen, M. A. (2015). MapReduce for MCMC Logit Estimation [Computer software]. 
% Retrieved from http://www.macohen.net/software or
% https://github.com/mcohen05007/MapReduceLogit
%% Combine Sub-samples
sumWmThetam = 0;
sumWm = 0;
while hasnext(intermValIter)
    thetam = getnext(intermValIter);
    Wm = eye(size(thetam,2))/cov(thetam);
    sumWmThetam = sumWmThetam + thetam*Wm;
    sumWm = sumWm + Wm;
end
theta = sumWmThetam*(eye(size(thetam,2))/sumWm);
add(outKVStore,'key',theta);
end