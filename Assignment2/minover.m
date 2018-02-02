% used matlab version: 2017b
% the script is ready to run



% number of dimensions
N = 20;
% max number of epochs
n_max = 100;
% number of times we re-generate the data end re-run the perceptron
nd = 100;


% alphas range
alphas = 0.1:0.1:10;

% array to store the value of alpha and the average generalization error
% over nd generated datasets
gen_er = zeros(length(alphas),2);


% iterate over different alphas
for i=1:length(alphas)
    a = alphas(i);
    % number of samples
    P = ceil(a*N);
    
    % calculate the average error for the given setting
    av_error = 0;
    
    % re-generate the data nd times
    for j=1:nd

        % the minover algorithm starts here

        % generate the dataset and the labels:        
        % generate random weights (or weights equal to 1), length = N
        %w_t = rand(1, N);
        w_t=ones(1, N);

        % generate the dataset itself
        data = randn([P N]);

        % define the labels with a teacher perceptron
        labels = sign(w_t*data');

        % initialize the learned weights with zeros
        w = zeros(1, N);

        % variable for checking the convergence
        change = 0;

        % strore the lowest stability achieved
        % we start with the value that is guaranteed to be higher than all the subsequent ones
        lowest = 100;
        
        % iterate until the algorithm converges
        % or the maximum number of epochs is achieved
        for ep=1:100

            % determine the stability
            % we do not have to divide by w (normalize) to find the minimum value
            % because the weight vector is the same inside one epoch
            k = (w*data').*labels;
            [sorted, indices] = sort(k);
    
            % determine the lowest stability (to check the convergence)
            lowest_new = sorted(1);
    
            % determine the sample with minimal stability
            % and its label
            min = data(indices(1), :);
            min_label = labels(indices(1));
    
            % Hebbian update step
            w = w+min*min_label/N;  
    
            % check if the lowest stability has changed
            % and update it if needed
            % if the lowest stability stays the same for P successive steps,
            % then we stop the training
            if lowest_new<lowest
                lowest = lowest_new;
                change = 0;
            else
                change = change+1;
                if change>=P
                    break
                end
            end
        end
        % calculate the generalization error
        error = 1/pi * acos((w*w_t')/(norm(w)*norm(w_t)));
        
        % add the error in this iteration to the average error
        av_error = av_error+error;
    end
    % compute the generalization error for the given alpha
    av_error = av_error/nd;
    % save the error value and the alpha in the array
    gen_er(i,1) = a;
    gen_er(i,2) = av_error;     
end

% print out the error array
gen_er


% plot the generalization error
figure
plot(gen_er(:,1), gen_er(:, 2))
xlabel("Alpha value (alpha=P/N)")
ylabel("Average generalization error")
title("Learning curve for the minover algorithm")

