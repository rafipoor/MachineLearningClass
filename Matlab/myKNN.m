clear;
close;

load('Dataset.mat');
ii   = 6;
Data = DataMatrices{ii};
TrueLabel = ClassLabels{ii};

nSamples  = size(Data,1);
nFeatures = size(Data,2);
nClasses  = max(TrueLabel);

nTr   = ceil(0.8*nSamples);
nTs   = nSamples - nTr;
TrIdx = randsample(nSamples,nTr);
TsIdx = setxor(1:nSamples,TrIdx);

TrainSet   = Data(TrIdx,:);
TrainLabel = TrueLabel(TrIdx,:);
TestSet    = Data(TsIdx,:);
TestLabel  = TrueLabel(TsIdx,:);

MaxK = 20;
Performance = zeros(MaxK,1);
for K=1:MaxK
    d = zeros(nTr,1);
    AssignedLabels = zeros(nTs,1);
    for i=1:nTs
        P = TestSet(i,:);
        for j =1:nTr
            d(j) = sum((P-TrainSet(j,:)).^2);
        end
        [~,Idx] = sort(d);
        C       = TrainLabel(Idx(1:(2*K-1)));
        AssignedLabels(i) = mode(C);
    end
    
    ConfusionMat = zeros(nClasses,nClasses);
    for i=1:nClasses
        for j=1:nClasses
            ConfusionMat(i,j) = sum(TestLabel == i & AssignedLabels == j);
        end
    end
    Performance(K) = trace(ConfusionMat)/nTs;
end
[~,K] = max(Performance);
BestK = 2*K-1;
plot(1:2:(2*MaxK-1),Performance)
ylim([0.5,1]);
disp(ConfusionMat);



[x,y] = meshgrid(min(Data(:,1)):0.1:max(Data(:,1)),min(Data(:,2)):0.1:max(Data(:,2)));
nSamples = numel(x);
TstData  = [x(:),y(:)];
for i=1:nSamples
    d = pdist2(TrainSet,TstData(i,:));
    [~,idx]= sort(d);
    C = TrainLabel(idx(1:BestK));
    AssignedLabels(i) = mode(C);
end
figure;
hold on
colormap('jet');
scatter(TstData(:,1),TstData(:,2),100,2+AssignedLabels,'fill','s');
scatter(Data(:,1),Data(:,2),5,TrueLabel,'fill');
axis('tight','square','off');
title('KNN');
print(gcf,'KNNSample.png','-dpng','-r300');