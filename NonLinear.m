clear;
close all;
nSamples = 300;
r1 = rand(nSamples,1); 
r2 = rand(nSamples,1)+ 1;
t1 = rand(nSamples,1)*2*pi;
t2 = rand(nSamples,1)*2*pi;
x1 = [r1.*cos(t1) r1.*sin(t1)];
x2 = [r2.*cos(t2) r2.*sin(t2)];
Data = [x1;x2];
CL   = [ones(nSamples,1);2*ones(nSamples,1)];

nSamples = size(Data,1);
Data = bsxfun(@minus,Data,mean(Data));

F1 = Data(:,1);
F2 = Data(:,2);

X  = [ones(nSamples,1) F1 F2 F1.*F2 F1.^2 F2.^2];
nFeatures = size(X,2);
Lambda = 0.1;
Y = CL;
Y(CL==2) = -1;

W = (X'*X+Lambda*eye(nFeatures))^-1 * X'*Y;

nClasses  =2;
[x,y] = meshgrid(min(F1):0.01:max(F1),min(F2):0.01:max(F2));
TstData = [x(:),y(:)];
nSamples = numel(x);
AssignedLabels = zeros(nSamples,1);
for i=1:nSamples
    P = TstData(i,:);
    X = [1 P(1) P(2) P(1)*P(2) P(1)^2 P(2)^2];
    g = X*W;
    AssignedLabels(i) = g>0;
end
AssignedLabels(AssignedLabels==0) = 2;
cla;
hold on
colormap('hot')
scatter(TstData(:,1),TstData(:,2),100,AssignedLabels,'fill','s');
scatter(Data(:,1),Data(:,2),5,4+CL,'fill');
axis('tight','equal','off');

print(gcf,'NonlinearSample.png','-dpng','-r300');
