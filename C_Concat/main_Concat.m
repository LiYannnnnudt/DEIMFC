%% --------------best single view--------------------------

clear all; clc;

% --------------- data ----------------- %
dataname = 'ORL4';
percent = 'X_00';
load(strcat('..\',dataname,'\',percent,'.mat'));
X = eval(percent,'.mat');
c = length(unique(Y));
%--------- process ---------%
for time = 1:5
    tic
    X_concat = [];
    for t = 1:length(X)
        Xtmp = X{t}';
        Xtmp(isnan(X{t}(1,:))',:)  =0;
        Xmean = repmat(mean(Xtmp),size(X{t},2),1);
        Xmean(find(isnan(X{t})'==0))=0;
        Xtmp = Xtmp + Xmean; 
        X_concat = [X_concat Xtmp];
    end
        label_out = kmeans(X_concat,c);
        [AC(time),NMI(time),jaccard(time),purity(time)] = MeasureClustering(Y,label_out);

%         AC(time)  = accuracy(Y,label_out);
%         NMI(time) = nmi(Y,label_out);
%         [jaccard(time), purity(time)] = myClustMeasure(label_out,Y); 
  timesfold(time) = toc;
end
meantime = mean(timesfold);
meanAC = mean(AC);stdAC = std(AC);
meanNMI = mean(NMI);stdNMI = std(NMI);
meanjaccard= mean(jaccard);stdjaccard = std(jaccard);
meanpurity= mean(purity);stdpurity = std(purity);


anova = [AC', NMI', jaccard';meanAC,meanNMI,meanjaccard;stdAC,stdNMI, stdjaccard ]
save(strcat('..\',dataname,'\',percent,'_RsConcat.mat')) 