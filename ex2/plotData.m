function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
y0 = y == 0; %find the index for y == 0 --> not admitted
y1 = y == 1; %find the index for y == 1 --> admitted
plot(X(y1, 1), X(y1, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7, 
X(y0, 1), X(y0, 2), 'o', "markerfacecolor", 'y', 'MarkerSize', 7);

% code provided by instructor
% pos = find(y == 1); neg = find(y ==0); Find indices of Positive and Negative 
%                                        example.

% Plot Examples
% plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidith', 2, 'MarkerSize', 7);
% plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);





% =========================================================================



hold off;

end
