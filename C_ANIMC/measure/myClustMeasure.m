function [ACC,NMI,jm,purity] =myClustMeasure(calculatedClust, realClust,emptyRealMapping)
% Initialization
efficiency = 0; purity = 0; PE = 0; jm = 0;

if nargin <3 || emptyRealMapping ==0
    pNum=length(calculatedClust);
    % both vectors should be in length*1 format
    [x,y] = size(calculatedClust);
    if y>x
        calculatedClust = calculatedClust';
    end
    [x,y] = size(realClust);
    if y >x
        realClust=realClust';
    end
    if 2 ~= length (find(size(calculatedClust) == size (realClust)))
        errordlg('Dimensions of Calculated and real mapping are mismatch','Wrong dimensions','modal');  
        return
    end
    % S=the clutering result in pairs - S(i,j)=1 if data point i and j are asigned to the same cluster
    clustPairs =  establishPairsMatrix (calculatedClust);
    realClustPairs=establishPairsMatrix (realClust);
    n11Mat= clustPairs&realClustPairs;        %pairs that appear in both methods
    n11=length(find(n11Mat));                       % number of pairs that appear in both methods
    n10 = length(find(realClustPairs)) - n11; % number of pairs that appear in 'real' classification, but not in the algorithm
    n01 =length(find(clustPairs)) - n11;         % number of pairs that appear in algorithm, but not in the 'real' classification
    if (n10+n01+n11)>0
        jm=n11/(n10+n01+n11);
    end
    if (n10+n11)>0
        %efficiency=sum(sum(efficiencyMat))/(numOfRealClust*numOfCalcClust);
        efficiency = n11/(n10+n11);
    end
    if (n01+n11)>0
        %purity=sum(sum(purityMat))/(numOfRealClust*numOfCalcClust);
        purity = n11/(n01+n11);
    end
    
    PE=sqrt(efficiency^2+purity^2);%norm
    
    ACC = accuracy(realClust, calculatedClust);
    
    NMI = nmi(realClust, calculatedClust);
    
    
    
    
    
end

function [pairsMatrix]= establishPairsMatrix (array)
len = length(array);
pairsMatrix = sparse(len,len);
for ind = 1:max(array)
    [indices]= find(array==ind);
    lenInd =   length(indices);
    if mod(ind,10) ==0
        disp('.');
    end
    for jnd = 1:lenInd-1
        pairsMatrix(indices(jnd),indices(jnd+1:lenInd))=1;
    end
end


function score = accuracy(true_labels, cluster_labels)
%ACCURACY Compute clustering accuracy using the true and cluster labels and
%   return the value in 'score'.
%
%   Input  : true_labels    : N-by-1 vector containing true labels
%            cluster_labels : N-by-1 vector containing cluster labels
%
%   Output : score          : clustering accuracy

% Compute the confusion matrix 'cmat', where
%   col index is for true label (CAT),
%   row index is for cluster label (CLS).
% S = SPCONVERT(D) converts the matrix D containing row-column-value
%   triples [i,j,v] as rows into a sparse matrix S such that
%      for k=1:size(D,1),
%         S(D(k,1),D(k,2)) = D(k,3).
%      end

n = length(true_labels);
cat = spconvert([(1:n)' true_labels ones(n,1)]);
cls = spconvert([(1:n)' cluster_labels ones(n,1)]);
cls = cls';
cmat = full(cls * cat);

%
% Calculate accuracy
%
[match, cost] = hungarian(-cmat);
score = (-cost/n);

function score = nmi(true_labels, cluster_labels)
%NMI Compute normalized mutual information (NMI) using the true and cluster
%   labels and return the value in 'score'.
%
%   Input    : true_labels    : N-by-1 vector containing true labels
%              cluster_labels : N-by-1 vector containing cluster labels
%
%   Output   : score          : NMI value
%
%   Reference: Shi Zhong, 2003.
%              http://www.cse.fau.edu/~zhong/software/textclust.zip

% Compute the confusion matrix 'cmat', where
%   col index is for true label (CAT),
%   row index is for cluster label (CLS).

n = length(true_labels);
cat = spconvert([(1:n)' true_labels ones(n,1)]);
cls = spconvert([(1:n)' cluster_labels ones(n,1)]);
cls = cls';
cmat = full(cls * cat);

n_i = sum(cmat, 1); % Total number of data for each true label (CAT), n_i
n_j = sum(cmat, 2); % Total number of data for each cluster label (CLS), n_j

% Calculate n*n_ij / n_i*n_j
[row, col] = size(cmat);
product = repmat(n_i, [row, 1]) .* repmat(n_j, [1, col]);
index = find(product > 0);
n = sum(cmat(:));
product(index) = (n*cmat(index)) ./ product(index);
% Sum up n_ij*log()
index = find(product > 0);
product(index) = log(product(index));
product = cmat .* product;
score = sum(product(:));
% Divide by sqrt( sum(n_i*log(n_i/n)) * sum(n_j*log(n_j/n)) )
index = find(n_i > 0);
n_i(index) = n_i(index) .* log(n_i(index)/n);
index = find(n_j > 0);
n_j(index) = n_j(index) .* log(n_j(index)/n);
denominator = sqrt(sum(n_i) * sum(n_j));

% Check if the denominator is zero
if denominator == 0
  score = 0;
else
  score = score / denominator;
end

