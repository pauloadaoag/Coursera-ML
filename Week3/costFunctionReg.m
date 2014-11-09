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



h = 1 ./ ( 1 .+ (exp(-1.*(transpose(theta) * transpose(X))
)));
h = transpose(h);
J = -y .* log(h) - (1 - y).*(log(1-h));
J = sum(J)/m;
J = J + ((lambda/ (2*m))*(sum (theta(2:end).*theta(2:end))));

n = size(theta);
t = ((h) - y) .*X(:,1);


grad(1) = (1/m)*sum(t);

for i=2:n
  t = ((h) - y) .*X(:,i);
  grad(i) = (1/m)*sum(t);
  grad(i) = grad(i) + ((lambda/m)*theta(i));
end
Xt = transpose(X);
b = h - y;
theta_mod = [0; theta(2:end)];
% theta_mod = (0 theta(2:end));
grad2 = (1/m) * (Xt * b) + ((lambda/m)*theta_mod);

grad - grad2


% % =============================================================

end
