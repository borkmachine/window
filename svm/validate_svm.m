% Load the training data.
load('data.mat');

% feat should be a MxN design matrix with M instances 
% of N dimensional feature vectors.
feat = double(feat);

% labels should be a Mx1 column vector of the corresponding labels.
labels = double(labels');

% K number of folds for K-fold cross validation
K = 5;

[m,n] = size(feat);

% Scale features to have zero-mean and unit variance.
Z = zscore(feat);

tic
validation_acc = train(labels, sparse(Z), ['-s ', num2str(K)]);
toc