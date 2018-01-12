% number of dimensions
N = 20;
% max number of epochs
n_max = 100;
% number of times we re-generate the data end re-run the perceptron
nd = 100;


% alphas range
alphas = 0.1:0.1:5;

% array to store the value of alpha and the average generalization error 
gen_er = zeros(length(alphas),2);


% iterate over different alphas
for i=1:length(alphas)
    a = alphas(i);
    % number of samples
    P = ceil(a*N);
    
    % calculate the average error for the given setting
    errors = 0;
    

    for j=1:nd

        % run the minover algorithm

        % generate the dataset and the labels
        
        % generate random weights (or weights equal to 1), length = N
        %w_t = rand(1, N);
        w_t=ones(1, N);

        % generate the dataset
        data = randn([P N]);

        % define the labels with a teacher perceptron
        labels = sign(w_t*data');

        % initialize weights
        w = zeros(1, N);

        % for checking the convergence
        change = 0;

        % the lowest stability
        lowest = 100;


        for ep=1:100

            % determine the stability
            % we do not have to divide by w to find the minimum
            k = (w*data').*labels;
            [sorted, indices] = sort(k);
    
            % determine the lowest stability
            lowest_new = sorted(1);
    
            % determine the sample with minimal stability
            % and its label
            min = data(indices(1), :);
            min_label = labels(indices(1));
    
            % Hebbian update step
            w = w+min*min_label/N;  
    
            % check if the lowest stability has changed
            % and update it if needed
            if lowest_new<lowest
                lowest = lowest_new;
                change = 0;
            % if the lowest stability stays the same for P successive steps,
            % then we stop the training
            else
                change = change+1;
                if change>=P
                    break
                end
            end
        end

        error = 1/pi * acos((w*w_t')/(norm(w)*norm(w_t)));

        errors = errors+error;
    end
    % compute the generalization error for the given alpha
    errors = errors/nd;
    gen_er(i,1) = a;
    gen_er(i,2) = errors;     
end

plot(gen_er(:,1), gen_er(:, 2))

