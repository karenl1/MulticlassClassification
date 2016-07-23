function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% compute the regularized cost function
% hypothesis and y are m x 1 matrices
hypothesis = sigmoid(X*theta);
costTerm = -1/m .* sum(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
% add the regularization term, but don't regularize theta(1) 
regularizedTheta = theta(2:end);
regularizationTerm = lambda/(2.*m) .* sum(regularizedTheta .^ 2);
J = costTerm + regularizationTerm;

% compute the gradient of the regularized cost function
% first, compute the gradient without the regularization term
% X' is n x m and (hypothesis - y) is m x 1, so X' * (hypothesis - y) is n x 1
gradientTerm = 1/m .* (X' * (hypothesis-y));
% don't add regularized term for theta(1)
newTheta = theta;
newTheta(1) = 0;
regularizationTerm = lambda/m .* newTheta;
grad = gradientTerm + regularizationTerm;

% =============================================================

grad = grad(:);

end
