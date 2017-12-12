
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



for setting=1:length(alphas)
    
    alpha = alphas(setting);

    % number of samples
    P = alpha*N;

    %number of times the perceptron converged
    success = 0;


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
            X = sprintf('Epoch %d', epoch);
            %disp(X);
            misclassified = sum(sign(sum(w.*data,2))~=labels);
            X = sprintf('%d misclassified samples', misclassified);
            %disp(X);
            if misclassified==0
                break;
            end
        end
        if misclassified==0
            X = sprintf('Converged after %d iterations', epoch);  
            success=success+1;
        else
            X = sprintf('Has not converged after %d iterations', epoch);
        end
        %disp(X);
    end
    %disp(alpha);
    %disp(success);
    
    n_of_successes(1, setting) = alpha;
    n_of_successes(2, setting)=success;
    
end

disp(n_of_successes);

alpha_values = n_of_successes(1, :);
success_ratio = n_of_successes(2, :)./nd;

figure
plot(alpha_values, success_ratio)
xlabel('Alpha')
ylabel('Success ratio')
title('The dependence between the alpha value and the success ratio for N = 20')


           
        
       
            
        
        
        
    




