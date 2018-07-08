clear;
close all;
load('Dataset');
ii    = 3;
Data  = DataMatrices{ii};
Data  = zscore(Data);

nFeatures = size(Data,2);
nSamples  = size(Data,1);

K = 2;
m = 2;

MaxIteration = 1000;
iter         = 1;

Centers    = randn(K,nFeatures);
OldCenters = Centers; 
ChangeFlag = 1;

figure; hold on; colormap('jet');

while iter<MaxIteration && ChangeFlag
    
    Distances = pdist2(Data,Centers);
    U = Distances.^(-2/(m-1));
    U = bsxfun(@times,U,1./sum(U,2));
    
    for j = 1:K
        Centers(j,:) = (U(:,j).^m)'*Data/sum(U(:,j).^m);
    end
    
    [~,Clusters]= max(U,[],2);
    ChangeFlag = norm(Centers(:)-OldCenters(:))>0.001;
    OldCenters = Centers;
    
    cla;
    scatter(Data(:,1),Data(:,2),10,Clusters);
    scatter(Centers(:,1),Centers(:,2),50,(1:K)','fill');
    pause(0.5);
end
