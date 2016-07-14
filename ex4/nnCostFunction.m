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
X = [ones(m, 1) X];                                          % add x0 term 5000x401 matrix
ry = eye(num_labels)(y, :);                                  % re-code y to be vector output 5000x10 matrix
ry = ry';                                                    % transpose to 10x5000 matrix

a2 = sigmoid(Theta1 * X');                                   % hidden layer: 25x401 * 401x5000 = 25x5000 matrix
a2 = a2';                                                    % transpose to 5000x25 matrix
a2 = [ones(m, 1) a2];                                        % add a2_0 term 5000x26 matrix
a3 = sigmoid(Theta2 * a2');                                  % output layer: 10x26 * 26x5000 = 10x5000 matrix

cost = (ry .* log(a3)) + ((1 - ry) .* log(1 - a3));          % 10x5000 matrix
J = -sum(sum(cost, 1)) / m;                                  % cost function without regularized term

T1 = (Theta1(:, 2:end)).^2;                                  % get Theta1 without bias unit
T2 = (Theta2(:, 2:end)).^2;                                  % get Theta2 without bias unit

r = (lambda / (2* m)) * (sum(sum(T1, 2)) + sum(sum(T2, 2))); % calculate regularized term

J = J + r;                                                   % regularized cost function

Delta1 = zeros(size(Theta1));                                % 25x401 matrix
Delta2 = zeros(size(Theta2));                                % 10x26 matrix

for i = 1:m
b_a1 = X(i, :);                                              % 1x401 matrix
b_z2 = Theta1 * b_a1';                                       % 25x1
b_a2 = sigmoid(b_z2);                                        % 25x1 matrix
b_a2 = b_a2';                                                % 1x25 matrix
b_a2 = [1 b_a2];                                             % 1x26 matrix
b_a3 = sigmoid(Theta2 * b_a2');                              % 10x26 * 26x1 = 10x1 matrix

d3 = b_a3 .- ry(:, i);                                       % delta3 10x1 matrix
d2 = ((Theta2' * d3)(2:end, :)) .* sigmoidGradient(b_z2);    % 25x10 * 10x1 = 25x1 matrix  

Delta2 = Delta2 .+ d3 * b_a2;                                % 10x1 * 1x26 = 10x26 matrix 
Delta1 = Delta1 .+ d2 * b_a1;                                % 25x1 * 1x401 = 25x401 matrix
end;

Theta1_grad = (1 / m) .* Delta1;
Theta2_grad = (1 / m) .* Delta2;

rNN1 = (lambda / m) .* [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];     % 25x401 matrix; add 0 to first column since we dont penalize bias unit
rNN2 = (lambda / m) .* [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];     % 10x21 matrix; add 0 to first column since we dont penalize bias unit

Theta1_grad = Theta1_grad + rNN1;
Theta2_grad = Theta2_grad + rNN2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
