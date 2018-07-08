close; clc; clear;
load Dataset.mat
Data = DataMatrices{3};
load CircularData.mat
X    = zscore(Data);
nSamples = size(X,1);
nClusters= nSamples;

AllClusters    = cell(nSamples,1);
AllClusters{1} = num2cell(1:nSamples,1);

DataDistances    = pdist2(X,X);
DataDistances(eye(nSamples)==1) = inf;
ClusterDistances = DataDistances;
for Level = 2:nSamples
    ThisClustering = AllClusters{Level-1};
    nClusters      = numel(ThisClustering);
    [i,j] = find(ClusterDistances == min(ClusterDistances(:)),1);
    
    ThisClustering{j} = [ThisClustering{j};ThisClustering{i}]; %merge
    ThisClustering(i) = [];
    
    ClusterDistances(i,:) = [];
    ClusterDistances(:,i) = [];
    for k = 1:nClusters-1
        Dik = DataDistances(ThisClustering{j},ThisClustering{k});
        ClusterDistances(j,k) = mean(Dik(:));
        ClusterDistances(k,j) = ClusterDistances(j,k);
    end
    ClusterDistances(j,j) = inf;
    AllClusters{Level}    = ThisClustering;
end

K = 2;
Clustering = AllClusters{end-K+1};
Labels     = zeros(nSamples,1);
for k=1:numel(Clustering)
    Labels(Clustering{k})= k;
end
scatter(Data(:,1),Data(:,2),10,Labels,'filled');
axis('equal');
print(gcf,'HAG.png','-dpng','-r300');