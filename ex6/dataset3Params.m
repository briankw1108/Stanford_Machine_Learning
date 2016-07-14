function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
value = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
min_pred_err = 1;     % From error calculation --> mean(double(predictions ~= yval))
                      % 1 is the maximum error if predictions are all not 
                      % equal to yval since it takes mean of the boolean output ~=.
                      % And average of all 1(not equal) is 1.

for C_i = value
        for sigma_i = value
                model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
                predictions = svmPredict(model, Xval);
                pred_err = mean(double(predictions ~= yval));
                if pred_err < min_pred_err
                        min_pred_err = pred_err;
                        C = C_i;
                        sigma = sigma_i;
                end 
        end        
end


% =========================================================================

end
