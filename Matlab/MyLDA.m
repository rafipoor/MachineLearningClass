clear;
close;

load('Dataset.mat');
ii   = 3;
Data      = DataMatrices{ii};
TrueLabel = ClassLabels{ii};

Data      = bsxfun(@minus,Data,mean(Data));
nSamples  = size(Data,1);
nFeatures = size(Data,2);
nClasses  = max(TrueLabel);

Prior = zeros(nClasses,1);
Means = zeros(nClasses,nFeatures);
SW    = zeros(nClasses,nFeatures,nFeatures);
SB    = zeros(nClasses,nFeatures,nFeatures);
for i=1:nClasses
    ThisClassData = Data(TrueLabel == i,:);
    Prior(i)      = size(ThisClassData,1);
    Means(i,:)    = mean(ThisClassData);
    SW(i,:,:)     = cov(ThisClassData);
    SB(i,:,:)     = Prior(i)*Means(i,:)*Means(i,:)';
end

SW = squeeze(sum(SW,1));
SB = squeeze(sum(SB,1));
J  = SW^-1*SB;
[V,~] = eig(J);
W     = V(:,1);

Projection = Data*W;

m1 = mean(Projection(TrueLabel==1));
m2 = mean(Projection(TrueLabel==2));
s1 = std(Projection(TrueLabel==1));
s2 = std(Projection(TrueLabel==2));
syms x;
eq  = exp(-((x-m1)^2)/(2*s1^2))/s1 - exp((-(x-m2)^2)/(2*s2^2))/s2 == 0; 
Thr = solve(eq,x);
Thr = round(Thr);
x = [min(Data(:,1)) max(Data(:,1))];
y1 = W(2)/W(1).*x + min(Data(:,2));
y2 = -W(1)/W(2).*x + Thr(1)/W(2);

subplot(1,2,1);
hold on
colormap('jet');
scatter(Data(:,1),Data(:,2),5,TrueLabel,'fill');
plot(x,y1,'r');
plot(x,y2,'k');
axis('equal');
legend('Data','Projective Line','Boundry','Location','Best');

subplot(1,2,2);
hold on
h1 = histogram(Projection(TrueLabel==1),40);
h2 = histogram(Projection(TrueLabel==2),40);

h1.Normalization = 'probability';
h1.BinWidth = 0.25;
h2.Normalization = 'probability';
h2.BinWidth = 0.25;
print(gcf,'LDASample.png','-dpng','-r300');