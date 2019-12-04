function Y = MyMLPprediction(W, b, X, Act_fun)
% Multilayer perceptron prediction
% Input:
%   W: Weight matrix between each layer of the neural network
%   b: bias vector for each layer of the neural network
%   X: d x n data matrix
%   Act_fun: Kind of activation function to use
% Ouput:
%   Y: p x n predicted response matrix
% Written by Ben Beiter
T = length(W);
Y = X;

for t = 1:T-1
    Y = ActivationFunction(W{t},Y,b{t},Act_fun);
end
Y = ActivationFunction(W{T},Y,b{T},0);