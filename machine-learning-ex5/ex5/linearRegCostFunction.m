function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

sum_err = sum((X * theta - y) .^2 );
sum_reg = sum(theta(2:end) .^ 2);
J = sum_err / (2 * m) + (lambda * sum_reg) / (2 * m);

tmp_theta = theta;
tmp_theta(1) = 0;

%for i=1:size(theta)
%  grad(i) = sum(X(:,i)' *(X * theta - y)) / m + (tmp_theta(i) .* lambda) ./ m;
%end

grad = X' * (X * theta - y) / m + (tmp_theta * lambda) / m;

% =========================================================================

grad = grad(:);

end
