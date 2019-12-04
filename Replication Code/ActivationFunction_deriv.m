function sigma_deriv = ActivationFunction_deriv(W,Z,b,fun_choice)
%Computes activation function for multi-layer perceptron algorithm
% Input:
%   W: n1 x n2 weight matrix - n1 = # of nodes in previous layer, n2 = # of nodes in next layer  
%   Z: n1 x 1 node value matrix - previous layer
%   b: n2 x 1 bias vector
%   fun_choice: choice of which activation functoin to use
% Ouput:
%   Znext: n2 x 1 node value matrix - next layer
%Written by Ben Beiter
switch fun_choice
    case 0 %none
        sigma_deriv = 1+0*W*Z;
    case 1 %tanh
        sigma_deriv = sech(W*Z+b).^2;
    case 2 %exp
        sig = ActivationFunction(W,Z,b,2);
        sigma_deriv = sig.*(ones(size(sig))-sig);
end
    
end