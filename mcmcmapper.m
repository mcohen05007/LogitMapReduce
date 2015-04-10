function mcmcmapper(M,d,~,intermKVStore)
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated for use or adaptation, please cite as:
% Cohen, M. A. (2015). MapReduce for MCMC Logit Estimation [Computer software]. 
% Retrieved from http://www.macohen.net/software or
% https://github.com/mcohen05007/MapReduceLogit
%%
y = table2array(d(:,10:end));
Xf = table2array(d(:,1:9))';
[N,J] = size(y);
K = size(Xf,1)/J;
X = reshape(Xf,K,J*N)';
Data = struct('X',X,'y',y);
Prior = struct('loglike',@llmnl,'logpriorden',@lndmvn,'thetabar',zeros(K,1),'A',eye(K)*0.01);
Mcmc = struct('R',2e4,'keep',1,'s',2.93/sqrt(size(X,2)),'M',M);
thetam = mcmcMR(Data,Prior,Mcmc);
add(intermKVStore, 'key', thetam)
end

function out = mcmcMR(Data,Prior,Mcmc)
%% Unpack Estimation Arguments
%Data
    y = Data.y;
    X = Data.X;
    k = size(X,2);
    
%Prior
    loglike = Prior.loglike;
    logpriorden = Prior.logpriorden;
    thetabar = Prior.thetabar; 
    A = Prior.A;
    rootpi = chol(A);

%MCMC params
    R = Mcmc.R;
    keep = Mcmc.keep;  
    s = Mcmc.s;
    M = Mcmc.M;
    
%%
x_0 = zeros(k,1);
options = optimset('Display','off','MaxIter',10000);
[mle,~,~,~,~, H] = fminunc(@(x) -loglike(x,y,X),x_0,options);
root = eye(k)/chol(H);

%% Pre-Allocate Store Draws
thetadraw  = zeros(R,k);
loglikedraw = zeros(R,1);

oll = loglike(mle,y,X);
otheta = mle;
naccept = 0;

tic
disp('MCMC Iteration (Estimated time to end)')
for rep = 1:R
    %% New draw
    [otheta,oll,naccept] = metropMR(loglike,y,X,logpriorden,otheta,oll,s,root,thetabar,rootpi,naccept,M); 
    %% Store draws
    if (mod(rep,keep) == 0) 
        mkeep = rep/keep;
        thetadraw(mkeep,:) = otheta;
        loglikedraw(mkeep) = oll; 
    end
end
disp(['Total Time Elapsed: ', num2str(round(toc/60)),' ','Minutes']) 
out = thetadraw;
end

function [otheta,oll,naccept] = metropMR(ll,y,X,lprior,otheta,oll,s,root,thetabar,rootpi,naccept,M)
thetac = otheta + s * root'*randn(numel(otheta),1);
cll = ll(thetac,y,X);
ldiff = cll + lprior(thetac,thetabar,rootpi)/M - oll - lprior(otheta,thetabar,rootpi)/M;
if rand(1)<=exp(ldiff)
    otheta = thetac;
    oll = cll;
    naccept = naccept+1;

end
end

function ll = llmnl(beta, y, X)
n = size(y,1);
j = size(X,1)/n;
eXbeta = exp(reshape(X * beta, j, n));
ll = sum(log(eXbeta(y'==1)'./sum(eXbeta)));
end

function p = lndmvn(x, mu, rooti) 
z = rooti' * (x - mu);
p = -(length(x)/2) * log(2 * pi) - 0.5 * (z(:)' * z(:)) + sum(log(diag(rooti)));
end
