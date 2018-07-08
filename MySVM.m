clear;
close all;
load('Dataset');
ii=4;
Data = DataMatrices{ii};
CL   = ClassLabels{ii};

Data(CL==2,1) = Data(CL==2,1) + 2;
Data(CL==2,2) = Data(CL==2,2) + 2;

nSamples = size(Data,1);

Data = bsxfun(@minus,Data,mean(Data));
X    = [ones(nSamples,1) Data];
Y    = CL;
Y(CL==2) = -1;

H = [0 0 0; 0 1 0; 0 0 1];
f = [0 0 0];
A = bsxfun(@times,X,Y);
b = ones(nSamples,1);
W = quadprog(H,f,-A,-b);


disp(W');
nClasses  =2;
hold on
colormap('jet');
scatter(Data(:,1),Data(:,2),5,CL,'fill');

t = min(Data(:,1)):0.1:max(Data(:,1));
y = (-W(2)*t  - W(1))/W(3);
yu = (-W(2)*t - W(1)+1)/W(3);
yl = (-W(2)*t - W(1)-1)/W(3);

plot(t,y,'k');
plot(t,yu,'r');
plot(t,yl,'r');

axis('equal');
legend('DataPoints','Boundary','Margins')
print(gcf,'SVMSample.png','-dpng','-r300');