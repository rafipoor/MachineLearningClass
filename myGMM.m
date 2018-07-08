clear;
close all;
load('Dataset');
ii    = 3;
Data  = DataMatrices{ii};

nSamples = size(Data,1);
nFeatures= size(Data,2);
X        = zscore(Data);


MaxIter = 30;
K       = 2;
Lambda  = 0.00001;

C       = rand(nSamples,K);
Alpha   = rand(K,nFeatures);
Mu      = randn(K,nFeatures);
Sigma   = repmat(eye(nFeatures,nFeatures),1,1,K);
Alpha   = Alpha/sum(Alpha);
LL      = zeros(MaxIter,1);

figure;
hold on; colormap('jet')
for Iter=1:MaxIter
    % expectation:
    for j=1:K
        C(:,j) = Alpha(j)*mvnpdf(X,Mu(j,:),squeeze(Sigma(:,:,j)));
    end
    TotalProb = sum(C,2);
    C = bsxfun(@times,C,1./TotalProb);
    
    %maximization:
    Pk    = sum(C)';
    Alpha = Pk/nSamples;
    Mu    = bsxfun(@times,C'*X,1./Pk);
    for j= 1:K
        Xtmp        = bsxfun(@minus,X,Mu(j,:));
        Xtmp        = bsxfun(@times,Xtmp,C(:,j).^(0.5));
        Sigma(:,:,j)= Xtmp'*Xtmp/Pk(j) + eye(nFeatures)*Lambda;
    end
    [~,Clusters]= max(C,[],2);
    LL(Iter)= sum(log(TotalProb));
    cla;
    scatter(X(:,1),X(:,2),5,Clusters);
    scatter(Mu(:,1),Mu(:,2),100,(1:K)','filled','s');
    title(Iter);
    pause(0.2);
end
figure;
plot(LL);
ylabel('Log Likelihood function');
xlabel('Iteration');