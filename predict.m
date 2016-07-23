function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% match each training example to a predicted value
% Theta1 is 25 x 401
% Theta2 is 10 x 26

% add a column of ones (bias unit) to X (X is m x 401)
X = [ones(m, 1), X];

% for each example, get the hidden layer (m x 25)
hiddenLayer = X * Theta1';
hiddenLayer = sigmoid(hiddenLayer);

% add the bias unit to the hidden layer (m x 26)
hiddenLayer = [ones(m, 1), hiddenLayer];

% get the output layer for each training example (m x 10)
outputLayer = hiddenLayer * Theta2';
outputLayer = sigmoid(outputLayer);

% get the max from each row to determine what the predicted value is
[max, maxIndex] = max(outputLayer, [], 2);
p = maxIndex;

% =========================================================================


end
