%-------------------------------------------------------------------------------------------------------------------------------------
% Contributed by Jinglin Xu
% Ref:
% [1] Multi-View K-Means Clustering on Big Data. (IJCAI,2013).
% [2] Discriminatively Embedded K-Means for Multi-view Clustering. (CVPR,2016)
% [3] Robust and Sparse Fuzzy K-Means Clustering. (IJCAI2016)
% [4] A new simplex sparse learning model to measure data similarity for clustering (AAAI2015)
% [5] https://github.com/ZJULearning/MatlabFunc/tree/master/Clustering
% [6] COMPACT: A Comparative Package for Clustering Assessment. (ISPA2005)
%--------------------------------------------------------------------------------------------------------------------------------------
clear all; clc;
% 
% % --------------- data ----------------- %
dataname = 'COIL100';
percent = 'X_30';
load(strcat('..\',dataname,'\',percent,'.mat'));
X = eval(percent,'.mat');

for t = 1:length(X)
        Xtmp = X{t}';
        Xtmp(isnan(X{t}(1,:))',:)  =0;
        Xmean = repmat(mean(Xtmp),size(X{t},2),1);
        Xmean(find(isnan(X{t})'==0))=0;
        Xtmp = Xtmp + Xmean;
        Xo{t} = Xtmp';
end
%--------- parameters ---------%
qStart = 1;    qNum = 50;      qStride = 0.1;  % q
rStart = 0;     rNum = 5;         rStride = 0.1;    % gamma
times = 5;     maxIter = 50;     c = max(Y);    numview = length(X);
%------------------------------------------
Alpha = ones(numview,1)/numview; 
%------------------------------------------
AC = zeros(times,1);               NMI = zeros(times,1);
meanAC = zeros(rNum,1);    meanNMI = zeros(rNum,1);
stdAC = zeros(rNum,1);         stdNMI = zeros(rNum,1);
jaccard = zeros(times,1);         purity = zeros(times,1);
meanJac = zeros(rNum,1);    meanPur = zeros(rNum,1);
stdJac = zeros(rNum,1);         stdPur = zeros(rNum,1);
tmp = [1:5:qNum];
for qq = 1:length(tmp)
    qtmp = tmp(qq);
    q = qStart + qStride*qtmp;
    for r = 1:rNum  
        gamma = rStart + rStride*r;      
        for k = 1:times
            
            %---------initialization---------%
            % X: m_dimensions*n_samples, U0:c*n, V0{p}:m*c, outU:c*n, outV{p}:m*c
            [U0,V0,label_init] = initialU(Xo,c,numview);    
            [outU,outV,outAlpha,outObj,outNumIter] = multi_Fuzzy_KMeans(Xo,U0,V0,Alpha,gamma,q,numview,maxIter);
        
            %--------- results ----------% 
            [~,label_out] = max(outU);   
            AC(k)  = accuracy(Y,label_out');
            NMI(k) = nmi(Y,label_out');
            [jaccard(k), purity(k)] = myClustMeasure(label_out',Y); 
            fprintf('-------------k = %d-----------r = %d---------\n',k,r)
        end
        meanAC(r,qq) = mean(AC)           
        meanNMI(r,qq) = mean(NMI);
        meanJac(r,qq) = mean(jaccard);     meanPur(r,qq) = mean(purity);
        stdAC(r,qq) = std(AC);                  stdNMI(r,qq) = std(NMI);
        stdJac(r,qq) = std(jaccard);            stdPur(r,qq) = std(purity);
    end
end
save(strcat('..\',dataname,'\',percent,'_RsMVASM.mat'))

