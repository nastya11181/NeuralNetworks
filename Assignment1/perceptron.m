% used matlab version: 2017b
% the script is ready to run

% change the N value underneath to obtain different results

% number of dimensions
N = 20;
% max number of epochs
n_max = 100;
% number of times we re-generate the data end re-run the perceptron
nd = 50;


% alphas range
alphas = 0.75:0.25:3;




% array to store the value of alpha and the corresponding number of successes 
n_of_successes = zeros(2,length(alphas));


% iterate over diferent alpha values
for setting=1:length(alphas)
    
    alpha = alphas(setting);

    % number of samples
    P = ceil(alpha*N);

    %number of times the perceptron converged
    success = 0;

    % re-generate the data nd times
    for run=1:nd
        % generate the data
        data = randn([P N]);

        % generate the labels
        labels=round(rand(P,1));
        labels(labels<1)=-1;

        % initialize the weights
        w = zeros(1, N);

        for epoch=1:n_max
            for sample=1:P
                % compute the actual output
                actual_output = w*data(sample, :)'*labels(sample);
                % update the weights
                if actual_output<=0
                    w = w + 1/N*(data(sample, :)*labels(sample));
                end
            end
            % calculate the number of miscalssified samples at the end of each epoch
            misclassified = sum(sign(sum(w.*data,2))~=labels);
            % if all the samples have been classified correctly, the algorithm has converged
            if misclassified==0
                break;
            end
        end
        % count the number of successful runs
        if misclassified==0 
            success=success+1;
        end
    end

    % save the results
    n_of_successes(1, setting) = alpha;
    n_of_successes(2, setting)=success;
    
end

disp(n_of_successes);

alpha_values = n_of_successes(1, :);
success_ratio = n_of_successes(2, :)./nd; %divide by the number of runs to obtain a value between 0 and 1

% plot the figure
figure
plot(alpha_values, success_ratio)
xlabel('Alpha')
ylabel('Success ratio')
X = sprintf('The dependence between the alpha value and the success ratio for N = %d', N);
title(X)




           
        
       
            
        
        
        
    




