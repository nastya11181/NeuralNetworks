

%  the learning rate
eta  = 0.05;

% the max number of epochs
tmax = 100;

% the number of examples we consider for training
P = 100;


% read in the data
data = load("data3.mat");

% extract the data samples (xi) and the labels (tau)
xi = data.xi;
tau = data.tau;

% N is the dimensionality of the input
% len is number of data samples
[N, len] = size(xi);

% initialize weights
w1 = rand(1, N);
w2 = rand(1, N);

% make sure that the norm is 1
w1 = w1/norm(w1);
w2 = w2/norm(w2);


for t=1:tmax*P
    % select training sample at random
    x = randi(P);
    x = xi(:, x);
    t = tau(:, x);
    
    % calculate sigma
    s = sigma(x, w1, w2);
    
    % calculate quadratic deviation (contribution)
    e = ((s-t)^2)/2;
    
    % calculate delta
    delta = (s-t)*...
    
    % calculate the gradients
    
    % update the weights
    w1 = w1-eta*g1*e;
    w2 = w2-eta*g2*e;
    
    
end

y = sigma(x, w1, w2)



% function for calculating sigma
function [y]= sigma(x, w1, w2)
y = tanh(w1*x)+tanh(w2*x);
end




    






