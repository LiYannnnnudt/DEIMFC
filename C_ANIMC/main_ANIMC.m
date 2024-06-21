
%NOTE THAT our paper "ANIMC: A Soft Framework for Auto-weighted Noisy and Incomplete Multi-view Clustering"
% performs the experiments in MATLAB R2019b and our codes run on a Windows 10 machine with 3:30 GHz E3-1225 CPU, 64 GB main memory.
% clear all;
% clc;
clear all; clc;
 
% --------------- data ----------------- %
dataname = 'ORL4';
percent = 'X_10';
load(strcat('..\',dataname,'\',percent,'.mat'));
XX = eval(percent,'.mat');

tic
% load multi-view dataset
[X, T , ind , label , viewNum , clusters] =  loaddataset_modif(XX,Y);

for i = 1:viewNum
    X{i} = X{i}';
end

time1 = toc;
for time =1:5
    tic
    options.afa =0.1;
%     for j = 1:1
        options.beta =100;
        options.sigema=options.afa;
        
        disp([options.afa,options.beta,options.sigema]);
        [U,V,A,obj,omega,ACC(time),NMI(time),jac(time),Purity(time)] = ANIMC(X,T,label,clusters,viewNum,options);
%     end
% end
time2(time) = toc;
end
meantime = time1+mean(time2);
meanACC = mean(ACC);meanNMI = mean(NMI);meanJAC = mean(jac);meanPUR = mean(Purity);
stdACC=std(ACC);stdNMI = std(NMI);stdJAC = std(jac);stdPUR = std(Purity);
ANOVA = [ACC',NMI',jac';meanACC,meanNMI,meanJAC;stdACC,stdNMI,stdJAC];
save(strcat('..\',dataname,'\',percent,'_RsANIMC_time.mat')) 