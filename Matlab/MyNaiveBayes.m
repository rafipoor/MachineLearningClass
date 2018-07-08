clear;
close all;
load Dataset;
Data = DataMatrices{1};
TrueLabel   = ClassLabels{1};

nFeatures= size(Data,2);
nSamples = size(Data,1);
nClasses = length(unique(TrueLabel));

%% parameter estimations:
Means = zeros(nClasses,nFeatures);
STDs  = zeros(nClasses,nFeatures);
Prior = zeros(nClasses,1);
for i=1:nClasses
    ThisClassData  = Data(TrueLabel == i,:);
    nClassSamples  = size(ThisClassData,1);
    Means(i,:)     = mean(ThisClassData);
    STDs(i,:) = std(ThisClassData);
    Prior(i) = nClassSamples/nSamples;
end

%% classification:
AssignedLabels = zeros(nSamples,1);
for i=1:nSamples
    P = Data(i,:);
    Probabilities = zeros(nClasses,1);
    for j=1:nClasses
        Likelihood=1;
        for k=1:nFeatures
            Likelihood = Likelihood * normpdf(P(k),Means(j,k),STDs(j,k));
        end
        Probabilities(j) = Likelihood * Prior(j);
    end
    [~,AssignedLabels(i)] = max(Probabilities); 
end

ConfusionMat = zeros(nClasses,nClasses);
for i=1:nClasses
    for j=1:nClasses
        ConfusionMat(i,j) = sum(TrueLabel == i & AssignedLabels == j); 
    end
end
disp(ConfusionMat);