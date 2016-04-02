function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    theta = theta - ((X' * ((X * theta) - y)) * (alpha / m)) % (2x1) - (2x97) x (97x1) = 2x1 matrix
    
    % Find the minimum J cost function by finding the minimum distance between sum of y and sum of prediction obtain from xj 

    %for j = 1:size(X, 2)
    %    for i = 1:m
    %        theta(j) = theta(j) - (sum((((X(i, :) * theta) - y(i)) * X(i, :))) * (alpha / m)) %theta = 2x1 X(i, :) = 1x2
    %    end     
    %end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
