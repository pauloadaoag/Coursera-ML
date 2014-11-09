function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% size(nn_params)
% size(input_layer_size)
% size(hidden_layer_size)
% size(num_labels)
% num_labels
% size(X)
% size(y)
% size(lambda)
% size(Theta1)
% size(Theta2)


Y = [];
for i=1:m
  yt = zeros(1, num_labels);
  yt(y(i)) = 1;
  Y = [Y; yt];
end

a1 = X;
a1_ = [ones(m,1) X];
z2 = a1_ * transpose(Theta1);
a2 = sigmoid(z2);
m2 = size(a2);
a2_ = [ones(m2, 1) a2];
z3 = a2_ * transpose(Theta2);
a3 = sigmoid(z3);
hm = a3;
% a3 = 5000 X 10 matrix


for i=1:m
  yt = Y(i,:);
  hmi = hm(i,:);
  for k=1:num_labels
    int = log(hmi(k))*(-1 * yt(k));
    int = int - ((1 - yt(k))  * log(1 - hmi(k) ) );
    J = J + int;
  end
end
J = J/m;
% size(Theta1)
Theta1_ = Theta1(:,2:end).*Theta1(:,2:end);
Theta1_ = sum(Theta1_(:));
Theta2_ = Theta2(:,2:end).*Theta2(:,2:end);
Theta2_ = sum(Theta2_(:));
J = J + ((lambda/ (2*m))*(Theta1_  + Theta2_));

 % size(z2)
% size a1 -> 5000x401
% size z2 -> 5000x25
% size a2 -> 5000x26
% size z3 -> 5000x10
% size a3 -> 5000x10
% Theta2 -> 10 x 26
% Theta1 -> 25 x 401
for t=1:m
  yt = Y(t,:);
  a3t = a3(t,:);
  d3 = a3t - yt; % d3 is 1 x 10
  
  d2 = (transpose(Theta2)*(transpose(d3)));
  % d2 = d2(2:end);
  d2 = d2 .* transpose(sigmoidGradient([1 z2(t,:)])); % d2 is 25 x 1
  d2 = d2(2:end);
  % d1 = (transpose(Theta1) * (d2));
  % d1 = d1(2:end);
  % d1 = d1 .* transpose(sigmoidGradient(z1(t,:)));

  
  Theta1_grad = Theta1_grad + ((d2) * (a1_(t,:)));

  
  Theta2_grad = Theta2_grad + (transpose(d3) * (a2_(t,:)));
  % Theta2_grad = Theta2_grad + (transpose(d3) * transpose(a3(t,:)));

end

mt = size(Theta1,1);
Theta1_reg = [zeros(mt, 1) Theta1(:,2:end)];
Theta1_reg = (lambda/m)*Theta1_reg;
mt = size(Theta2,1);
Theta2_reg = [zeros(mt, 1) Theta2(:,2:end)];
Theta2_reg = (lambda/m)*Theta2_reg;

Theta2_grad = Theta2_grad ./m; %(m.*(Theta2_reg));
Theta1_grad = Theta1_grad ./m; %(m.*(Theta1_reg));
Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
