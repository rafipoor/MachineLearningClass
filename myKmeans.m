clear;
close all;
load('Dataset');
ii    = 3;
Data  = DataMatrices{ii};
Data  = zscore(Data);
nFeatures = size(Data,2);
nSamples  = size(Data,1);

K = 10;
Centers = randn(K,nFeatures);
MaxIteration = 10000;
iter =1;
OldCenters   = Centers; 
Flag = 1;
while iter<MaxIteration && Flag
    
    Distances = pdist2(Data,Centers,'cosin');
    [~,Labels]= min(Distances,[],2);
    
    for j=1:K
        ThisCluster = Data(Labels==j,:);
        Centers(j,:) = mean(ThisCluster);
    end
    cla;
    hold on;
    colormap('jet');
    scatter(Data(:,1),Data(:,2),10,Labels);
    scatter(Centers(:,1),Centers(:,2),50,1:K,'fill');
    pause(0.5);
    
    Flag = norm(Centers(:)-OldCenters(:))>0;
    OldCenters = Centers;
end
