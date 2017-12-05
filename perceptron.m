
% number of dimensions
N = 20;
% number of samples
P = 1.5*N;
% max number of epochs
n_max = 100;

% generate the data
data = randn([P N]);

% generate the labels
labels=round(rand(P,1));
labels(labels<1)=-1;

% initialize the weights
w = zeros(1, N+1);

% add one dimension for the bias
data(:, N+1)=-1;


for epoch=1:n_max
    for sample=1:P
        % compute the actual output
        actual_output = sum(w.*data(sample, :))*labels(sample);
        % update the weights
        if actual_output<=0
            w = w + 1/N*(data(sample, :)*labels(sample));
        end
    end
    X = sprintf('Epoch %d', epoch);
    disp(X);
    misclassified = sum(sign(sum(w.*data,2))~=labels);
    X = sprintf('%d misclassified samples', misclassified);
    disp(X);
    if misclassified==0
        break;
    end
end
X = sprintf('Converged after %d iterations', epoch);
disp(X);
%disp(w);
            
            
        
       
            
        
        
        
    




