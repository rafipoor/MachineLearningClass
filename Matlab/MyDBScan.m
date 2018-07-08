close; clc; clear;
%load CircularData.mat
load Dataset.mat
Data = DataMatrices{3};
X    = zscore(Data);


Epsilon  = 0.4;
MinPoints= 10;

nSamples = size(X,1);
Labels   = zeros(nSamples,1);
Distances= pdist2(X,X);
Visited  = false(nSamples,1);
IsOutlier= false(nSamples,1);
ThisClust= 0;

for i=1:nSamples
    if ~Visited(i)
        Visited(i) = true;
        Neighbors  = find(Distances(i,:)< Epsilon);
        if numel(Neighbors)<MinPoints
            IsOutlier(i) = true;
        else
            %New Cluster
            ThisClust = ThisClust+1;
            Labels(i) = ThisClust;
            
            while ~isempty(Neighbors)
                j = Neighbors(1);
                Neighbors(1)=[];
                if ~Visited(j)
                    Visited(j) = true;
                    Labels(j)  = ThisClust;
                    Neighbors2 = find(Distances(j,:)<= Epsilon);
                    if numel(Neighbors2)>= MinPoints
                        Neighbors = union(Neighbors,Neighbors2);
                    end
                end
            end
        end
    end
end

hold on;
colormap('jet')
scatter(Data(:,1),Data(:,2),10,Labels,'filled');
scatter(Data(IsOutlier==1,1),Data(IsOutlier==1,2),30,'r');
axis('equal');
print(gcf,'DBSCAN.png','-dpng','-r300');
