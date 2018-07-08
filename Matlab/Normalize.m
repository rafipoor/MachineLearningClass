function y = Normalize(X)
MeanX = mean(X);

for i=1:size(X,1)
    X(i,:) = X(i,:)-MeanX;
end

for i=1:size(X,2)
    X(:,i) = X(:,i)/std(X(:,i));
end
y = X;
end