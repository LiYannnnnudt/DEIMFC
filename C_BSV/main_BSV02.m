%% --------------best single view--------------------------

clear all; clc;
 
% --------------- data ----------------- %
dataname = 'sonar';
percent = 'X_00';
load(strcat('.\',dataname,'\',percent,'.mat'));
X = eval(percent,'.mat');
tic
%--------- process ---------%
for time = 1:5
    for t = 1:length(X)
        Xtmp = X{t}';
        Xtmp(isnan(X{t}(1,:))',:)  =0;
        Xmean = repmat(mean(Xtmp),size(X{t},2),1);
        Xmean(find(isnan(X{t})'==0))=0;
        Xtmp = Xtmp + Xmean;
        label_out = kmeans(Xtmp,max(Y));
        AC(time,t)  = accuracy(Y,label_out);
        NMI(time,t) = nmi(Y,label_out);
        [jaccard(time,t), purity(time,t)] = myClustMeasure(label_out,Y);   
    end
end
meanAC = mean(AC);stdAC = std(AC);
meanNMI = mean(NMI);stdAC = std(NMI);
meanjaccard= mean(jaccard);stdAC = std(jaccard);
meanpurity= mean(purity);stdAC = std(purity);

save(strcat('.\',dataname,'\',percent,'_RsBSV.mat')) 


