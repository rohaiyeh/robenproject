clear
clc
close all
format compact

%% Define System to Identify
SysID = 3;
switch SysID
    case 1 %First Order Linear System
        A = [-3];
        B = [1];
        C = [3];
        D = 2;
        x_deriv = @(x,u) A*x + B*u;
        y_out = @(x,u) C*x + D*u;
    case 2 %Second Order Non-Linear System
        A = [-3 5; -9 -1];
        B = [0; 1];
        C = [3 4];
        D = 2;
        x_deriv = @(x,u) A*x + B*u;
        y_out = @(x,u) C*x + D*u;
    case 3 %First order non-linear system
        A = [-3];
        B = [1];
        x_deriv = @(x,u) A*x.^(1.5) + log(x+10) + B*u*sin(x);
        C = [3];
        D = 2;
        y_out = @(x,u) C*sin(x) + D*x*u;
end

%Define Noise to Add
NoiseID = 2;
switch NoiseID
    case 1
        Noise = @(v) zeros(size(v));
    case 2
        Noise = @(v) 0.005*randn(size(v));
end
%% Simulate system
%Number of simulations to train on and test on
num_train_sim = 9;
num_test_sim = 2;
num_sim = num_train_sim + num_test_sim;

%Time settings for simulation
dt = 0.001;
t_end = 3;
tvec = 0:dt:t_end;
for i = 1:num_sim
    %Initiate variables for each Sim
    x = zeros(size(A,1),length(tvec));
    y = zeros(size(C,1),length(tvec));
    u = zeros(size(B,2),length(tvec));
    x(:,1) = zeros(size(A,1),1);
    %Define an input law for each simulation
    switch i
        case 1
            u_in = @(time) 1;
        case 2
            u_in = @(time) 5;
        case 3
            u_in = @(time) 10;
        case 4
            u_in = @(time) 4;
        case 5
            u_in = @(time) 2;
        case 6
            u_in = @(time) 1 + (time>1) + (time>2);
        case 7
            u_in = @(time) sin(time);
        case 8
            u_in = @(time) 5*exp(-time);
        case 9
            u_in = @(time) exp(1 - time);
        case 10
            u_in = @(time) 3;
        case 11
            u_in = @(time) 1 + (time>1) - 2*(time>2);
    end
    %Simulate System with Euler's Method
    for t = 1:length(tvec)-1
        u(:,t) = u_in(tvec(t)) + Noise(u(:,t));
        dx = x_deriv(x(:,t),u(:,t));
        x(:,t+1) = x(:,t) + dx*dt;
        y(:,t) = y_out(x(:,t),u(:,t));
    end
    u(:,t+1) = u_in(tvec(t+1));
    y(:,t+1) = y_out(x(:,t+1),u(:,t+1));
    Sims(i).x = x;
    Sims(i).y = y;
    Sims(i).u = u;
end

%% Generate Training + Testing Data
% Cut the simulation data down n datapoints to use in the MLP
n = 500; %data points per sim
ext = mod(length(tvec),n);
tpn = (length(tvec)-ext)/n;
for i = 1:num_sim
    Sims(i).x_cut = [Sims(i).x(:,1:tpn:end) Sims(i).x(:,end)];
    Sims(i).y_cut = [Sims(i).y(:,1:tpn:end) Sims(i).y(:,end)];
    Sims(i).u_cut = [Sims(i).u(:,1:tpn:end) Sims(i).u(:,end)];
end
t_cut = [tvec(:,1:tpn:end) tvec(end)];

% Prepare Training Data
xdata_train = []; udata_train = []; ydata_train = [];
for i = 1:num_train_sim
    xdata_train = [xdata_train Sims(i).x_cut];
    ydata_train = [ydata_train Sims(i).y_cut];
    udata_train = [udata_train Sims(i).u_cut];
end

%% Set up and train MLP
k = [5,5,5,5,5];        %number of nodes in each hidden layer
lambda = 2e-2;        %regularization parameter
eta = 1e-3;           %learning rate
epoch_num = 20000;     %number of iterations through data for training
Act_fun = 2;          %Activation function choice: 1-tanh 2-exp^-1

% Train MLP to predict y given x and u at every timestep
feat_vec = [xdata_train;udata_train];
[W1, b1, Loss1] = MyMLP(feat_vec,ydata_train,k,lambda,Act_fun,eta,epoch_num);

%% Plot Loss during Training
figure(20)
semilogy(Loss1);
xlabel('Loss')
ylabel('Training Epochs')
title('Loss1')

%% Test MLP on All Simulations
for j = 1:num_sim
    %test MLP
    Sims(j).X1 = [Sims(j).x_cut;Sims(j).u_cut];
    Sims(j).Y1 = MyMLPprediction(W1, b1, Sims(j).X1, Act_fun);
    
% Plot Simulations
    if j <= num_train_sim
        Ttl = 'Training';   i = j;
    else
        Ttl = 'Testing';   i = j-num_train_sim;
    end
    figure(j)
    title(sprintf([Ttl ' Sim %.1i'],i))
    hold on
    plot(t_cut,Sims(j).y_cut,t_cut,Sims(j).Y1,'LineWidth',2)
    plot([t_cut(1) t_cut(end)],[0 0]+12,'k--')
    xlabel('time')
    ylabel('Output y')
    legend('Actual Output','Predicted Output')
end

