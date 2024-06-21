
% If you use the code, please cite the following papers:
% [1] Hu M, Chen S. Doubly aligned incomplete multi-view clustering[C]//Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2018: 2262-2268.
% [2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on Incomplete Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS, 2022.
% Thanks Menglei Hu for providing the source code of DAIMC!
clear;clc

dataname ='ORL4';
percent = 'X_00';

load(strcat('..\',dataname,'\',percent,'.mat'));
X = eval(percent,'.mat');

tic
rand('seed',6821)
num_view = length(X);
numClust = length(unique(Y));
numInst  = length(Y); 


options.afa = 0.0001;
options.beta = 10; 
% options.afa = 0.0001;
% options.beta = 100; 
if size(X{1},2)~=length(Y)
    for iv = 1:num_view
        X{iv} = X{iv}';
    end
end
for iv = 1:length(X)
    X1 = X{iv};
    X1 = NormalizeFea(X1,0);
    ind_0 = isnan(X1(1,:));
    %ind_0 = find(ind_folds(:,iv) == 0);
    X1(:,ind_0) = 0 ;
    XTMP{iv} = X1; 
    W{iv} = diag(1-ind_0);                       
end

clear X X1 W1 ind_0
X = XTMP;
clear XTMP
time1 = toc;
for time =1:5
    tic
[U0,V0,B0] = newinit(X,W,numClust,num_view);
[U,V,B,F,P,N] = DAIMC(X,W,U0,V0,B0,Y,numClust,num_view,options);

% indic = litekmeans(V, numClust, 'Replicates', 20);
indic = kmeans(V, numClust);
[ACC(time) ,NMI(time) ,jm(time) ,Purity(time)] = MeasureClustering(Y, indic);  
time2(time) = toc;
% 
% result_CLU = ClusteringMeasure(Y, indic);   
% [jm(time),Purity(time)] =myClustMeasure(Y, indic);
% NMI(time)  = nmi(Y, indic);
% 
% ACC(time)  = result_CLU(1);
%[ARi(time),~,~,~] = RandIndex(Y, indic);
end
Result  = [mean(ACC),mean(NMI),mean(jm),mean(Purity)];
Result(2,:)  = [std(ACC),std(NMI),std(jm),std(Purity)];
meantime = time1+mean(time2);
disp(['acc=',num2str(Result(1)),'----nmi=',num2str(Result(2)),'----JARM=',num2str(Result(3)),'----pur=',num2str(Result(4))]);
ANOVA = [ACC', NMI', jm'; Result(:,1:3)];
save(strcat('..\',dataname,'\',percent,'_RsDAIMC_time.mat')) 

