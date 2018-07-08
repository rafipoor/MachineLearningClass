clear;
close all;
load('Dataset');

nTrain  = 500;
nHidden = 10;
Alpha   = 0.01;
MaxEpoch= 500;
nOutput = 1;

ii=8;
Data       = DataMatrices{ii};
TrueLabel  = ClassLabels{ii};
Y    = TrueLabel;
Y(TrueLabel==2) = 0;

nSamples = size(Data,1);
nFeatures= size(Data,2);
Data= [ones(nSamples,1) Data];

nTest = nSamples- nTrain;
TrIdx = randsample(nSamples,nTrain);
TrainSet = Data(TrIdx,:);
YTrain   = Y(TrIdx);
TestSet  = Data(setxor(1:nSamples,TrIdx),:);
YTest    = Y(setxor(1:nSamples,TrIdx));

W = rand(nHidden,nFeatures+1);
V = rand(nOutput,nHidden+1);

ETest = zeros(MaxEpoch,1);
ETrain= zeros(MaxEpoch,1);

for epoch=1:MaxEpoch
    idx = randperm(nTrain);
    for j=1:nTrain
        
        X  = TrainSet(idx(j),:);
        Y  = YTrain(idx(j));
        
        X1 = W*X';
        X2 = sigmf(X1,[1 0]);
        X3 = V*[1;X2];
        O  = sigmf(X3,[1 0]);
        
        dEdO  = 2*(O - Y);
        dOdX3 = (1-O)*O;
        dX3dV = [1;X2]';
        dEdV  = dEdO*dOdX3*dX3dV;
        V     = V - Alpha * dEdV;
        
        dX3dX2 = V;
        dX2dX1 = X2 .* (1-X2);
        dX1dW = zeros(size(W));
        for i1=1:nHidden
            for i2=1:(nFeatures+1)
            dX1dW(i1,i2) = X(i2);
            end
        end
        dEdW = dEdO*dOdX3*dX3dX2(2:nHidden+1)*dX2dX1*dX1dW;
        W = W -Alpha*dEdW;
    end
    O = sigmf([ones(nTrain,1) sigmf(TrainSet*W',[1 0])]*V',[1 0]);
    ETrain(epoch) = sum((O-YTrain).^2);
    
    O = sigmf([ones(nTest,1) sigmf(TestSet*W',[1 0])]*V',[1 0]);
    ETest(epoch) = sum((O-YTest).^2); 
end
hold on
plot(1:MaxEpoch,ETrain,'r');
plot(1:MaxEpoch,ETest,'b');
legend('Train Error','Test Error');

O = sigmf([ones(nSamples,1) sigmf(Data*W',[1 0])]*V',[1 0]);
AssignedLabel = (O <0.5) + 1;
Performance = mean(AssignedLabel == TrueLabel);


[x,y] = meshgrid(min(Data(:,2)):0.1:max(Data(:,2)),min(Data(:,3)):0.1:max(Data(:,3)));
nSamples = numel(x);
TstData = [ones(nSamples,1),x(:),y(:)];

O = sigmf([ones(nSamples,1) sigmf(TstData*W',[1 0])]*V',[1 0]);
AssignedLabel = (O <0.5) + 1;

AssignedLabel(AssignedLabel==0) = 2;

figure;
hold on
colormap('jet');
scatter(TstData(:,2),TstData(:,3),100,2+AssignedLabel,'fill','s');
scatter(Data(:,2),Data(:,3),5,TrueLabel,'fill');
axis('tight','square','off');
title('MLP');
print(gcf,'MLPSample.png','-dpng','-r300');