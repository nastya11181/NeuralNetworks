

%  the learning rate
eta  = 0.05;

% the max number of epochs
tmax = 200;

% the number of examples we consider for training
P = 3000;

% the testing data
Q = 2000;

% P+Q <= nlen = 5000

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
ñcdw2 = w2/norm(w2);


% create an array to keep track of the results
errors = zeros(tmax, 3);

for ep=1:tmax
    
    % create a random permutation of P training samples
    perm = randperm(P);
    E = 0;
    
    % consider 1 sample at once
    for p=1:P
    
        % select training sample at random
        ind = perm(p);
        x = xi(:, ind);
        t = tau(ind);
    
        % calculate sigma
        s = sigma(x, w1, w2);
    
        % calculate the gradients
        g1 = (s-t)*(1-(tanh(w1*x))^2)*x;
        g2 = (s-t)*(1-(tanh(w2*x))^2)*x;    
    
        % update the weights
        w1 = w1-(eta*g1)';
        w2 = w2-(eta*g2)';
        
        % calculate the error
        E = E+(s-t)^2;
    end
    
    % calculate the training error
    E = 1/P*1/2*E;
    
    % save the results
    errors(ep, 1) = ep;
    errors(ep, 2) = E;
    
    
    % calculate the test (generalization) error
    Etest = 0;
    for q=P+1:P+Q
        x = xi(:, q);
        t = tau(q);
        
        s = sigma(x, w1, w2);
        
        Etest = Etest+(s-t)^2;
        
    end
    
    Etest = 1/Q*1/2*Etest;
    % save the results for plotting
    errors(ep, 3) = Etest;
    
    
end

% display 
errors

% plot the cost function
figure
plot(errors(:, 1), errors(:,2))
xlabel("Time in epochs, t")
ylabel("Error, E")
title("Cost function and the generalization (test) error function")
hold on
plot(errors(:,1), errors(:, 3))
legend("cost function", "test error function")


% function for calculating sigma
function [y]= sigma(x, w1, w2)
y = tanh(w1*x)+tanh(w2*x);
end




    






