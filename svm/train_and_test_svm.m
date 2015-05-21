function [ plabel, acc, dvalues ] = train_and_test_svm(X_train, y_train, X_test, y_test, C)
%TRAIN_AND_TEST_SVM Train a SVM on the given data.
%  X_train - MxN design matrix with M instances of N dim feature vectors.
%  y_train - Mx1 column vector of the corresponding labels.
%  X_test  - Design matrix for testing.
%  y_test  - Testing labels.
%  C       - String containing LIBLINEAR parameters.
%
%  @return
%  plabel  - predicted label for each test instance
%  acc     - test accuracy
%  dvalues - decision values

if nargin < 5
    C = '';
end

tic
model = train(y_train, sparse(X_train), C);
[plabel, acc, dvalues] = predict(y_test, sparse(X_test), model);
toc

end

