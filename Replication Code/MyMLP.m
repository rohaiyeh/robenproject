function [W, b, Loss] = MyMLP(X,Yd,k,lambda,Act_fun,eta,epoch_num)
% Train a multilayer perceptron neural network
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   k: T x 1 vector to specify number of hidden nodes in each layer
%   lambda: regularization parameter
%   Act_fun: Which Activation Function to use
%   eta: learning rate
%   epoch_num: number of epochs to train MLP
% Ouput:
%   W: T x 1 cell of weight matrices
%   b: T x 1 cell of bias vectors
%   L: 1 x epoch_num vector of total loss in each epoch
% Written by Ben Beiter

%% Initialize variables
Loss = zeros(1,epoch_num); %initialize vector of loss measurements for each iteration

k = [size(X,1);k(:);size(Yd,1)]; %size of each layer
T = numel(k)-1;  %number of connections btwn layers 
W = cell(T,1);   %Empty Weight matrices
b = cell(T,1);   %empty bias vectors
for t = 1:T
    W{t} = randn(k(t+1),k(t)); %initialize weight matrices  ****I switched the dimensions
    b{t} = randn(k(t+1),1);    %initialize bias vector
end
del = cell(T,1); %Loss at each layer
H = cell(T,1);   %layer values for this data point

%% Forward then Backward Propogation
        %First layer is h1, 2nd is h2, etc., last is y
for iter = 2:epoch_num
%   Forward Propogation
    H{1} = ActivationFunction(W{1},X,b{1},Act_fun); %values of first hidden layer
    for t = 2:T-1
        H{t} = ActivationFunction(W{t},H{t-1},b{t},Act_fun); %values of hidden layers
    end 
    H{T} = ActivationFunction(W{T},H{T-1},b{T},0); %values of output vector

%   Loss function
    J = H{T}-Yd;     
    Wn = cellfun(@(x) dot(x(:),x(:)),W);% |W|^2
    Loss(iter) = dot(J(:),J(:))+lambda*sum(Wn);

%   Backward Propogation
    del{T} = J/size(X,2);              % delta
    for t = T-1:-1:2
        %df = 1-H{t+1}.^2;    % h'(a)
        del{t} = (W{t+1}'*del{t+1}).*ActivationFunction_deriv(W{t},H{t-1},b{t},Act_fun);    % delta
    end
    del{1} = (W{2}'*del{2}).*ActivationFunction_deriv(W{1},X,b{1},Act_fun);    % delta
    
%   Gradient Descent
    dE_dWt = del{1}*X'+lambda*W{1};
    db = sum(del{1},2);
    W{1} = W{1}-eta*dE_dWt;
    b{1} = b{1}-eta*db;
    
    for t=2:T
        dE_dWt = del{t}*H{t-1}';%+lambda*W{t};
        db = sum(del{t},2);
        W{t} = W{t}-eta*dE_dWt;
        b{t} = b{t}-eta*db;
    end
end
Loss = Loss(1,2:iter);
