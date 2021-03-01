function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


predictedVals = sigmoid(X*theta);
costVector = -y.*log(predictedVals) - (-y + 1).*log(-predictedVals + 1);
regVector = theta;
regVector(1) = 0;
regValue = (sum(regVector.^2) * lambda)/(2*m);
J = sum(costVector)/m + regValue;

preGrad = transpose(X) * (predictedVals - y);
regValueGrad = (regVector .* lambda) ./ m;
grad = (preGrad / m) + regValueGrad;



% =============================================================

end
