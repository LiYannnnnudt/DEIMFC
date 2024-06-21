%% --------------best single view--------------------------
% X:d*N
clear all; clc;
 
% --------------- data ----------------- %
dataname = 'ORL4';
percent = 'X_70';
load(strcat('..\',dataname,'\',percent,'.mat'));
X = eval(percent,'.mat');

%--------- process ---------%
for time = 1:5
    tic
    for t = 1:length(X)
        Xtmp = X{t}';
        Xtmp(isnan(X{t}(1,:))',:)  =0;
        Xmean = repmat(mean(Xtmp),size(X{t},2),1);
        Xmean(find(isnan(X{t})'==0))=0;
        Xtmp = Xtmp + Xmean;
        label_out = kmeans(Xtmp,max(Y));
        [AC(time,t),NMI(time,t),jaccard(time,t),purity(time,t)] = MeasureClustering(Y,label_out);
%         AC(time,t)  = accuracy(Y,label_out);
%         NMI(time,t) = nmi(Y,label_out);
%         [jaccard(time,t), purity(time,t)] = myClustMeasure(label_out,Y);   
    end
    timesfold(time) = toc;
end
meantime = mean(timesfold);
meanAC = mean(AC);stdAC = std(AC);
meanNMI = mean(NMI);stdNMI = std(NMI);
meanjaccard= mean(jaccard);stdjaccard = std(jaccard);
meanpurity= mean(purity);stdpurity = std(purity);
jj = 3;
ANOVA = [AC(:,jj) NMI(:,jj) jaccard(:,jj)];
ANOVA = [ANOVA;  mean(ANOVA); std(ANOVA)  ]
save(strcat('..\',dataname,'\',percent,'_RsBSV_ANOVA.mat'))


