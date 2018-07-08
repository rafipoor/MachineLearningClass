clear;
close all;
load('Dataset');
ii=3;
Data      = DataMatrices{ii};
TrueLabel = ClassLabels{ii};

nSamples = size(Data,1);

Data = bsxfun(@minus,Data,mean(Data));
X    = [ones(nSamples,1) Data];
nFeatures = size(X,2);
Lambda = 0.1;
Y = TrueLabel;
Y(TrueLabel==2) = -1;

W = (X'*X + Lambda*eye(nFeatures))^-1 * X'*Y;
hold on
colormap('jet');
scatter(Data(:,1),Data(:,2),5,TrueLabel,'fill');
x = [min(Data(:,1)) max(Data(:,1))];
y = -W(2)/W(3)*x -W(1)/W(3);
plot(x,y,'k');
legend('Data','Boundry','Location','Best');
print(gcf,'LinearSample.png','-dpng','-r300');