% This file executes loading data and training of custom neural network
% Designed for 2-cell islet where g_ca is held constant. Only g_katp and
% g_c values vary -- for determining biological relevance.
%
% Neural Network Information:
%   Inputs: 
%     cell 1 g_katp
%     cell 2 g_katp
%     cell 1,2 g_c (coupling value)
%
%   Output:
%     binary (0 or 1) indicating existence of switch cell -- 1 = has switch
%
% This network will have 3 layers

function [num_tp,num_tn,num_fp,num_fn] = predictswitchisletv2(modeldata_filename,learning_rate,iterations)

% LAYER SIZES
input_layer_size = 3; %(coupling matrix + 2(# of cells)) because of gca and gkatp params for each cell
output_layer_size = 1;
hidden_layer_size = 20;
n = 5000; % number of modeldata sets
testn = 2000; % number of data sets used for testing

% GIVEN DATA DISTRIBUTION VALUES
gkatp_mean_value = 130; % mean of standard dev. from which data was drawn
gstar = 133;
fig_num = 1;


% CREATE TRAINING/TEST DATA
[modeldata] = load_data(modeldata_filename);
training_inputs = modeldata(1:input_layer_size,1:n-testn);
training_targets = modeldata(input_layer_size+1,1:n-testn);
test_inputs = modeldata(1:input_layer_size,n-testn:n);
test_targets = modeldata(input_layer_size+1,n-testn:n);

input = [training_inputs test_inputs];
training_data = vertcat(training_inputs,training_targets);

% Randomize training data
cols = size(training_data,2);
P = randperm(cols);
training_data = training_data(:,P);


% PLOT EXPECTED OUTCOMES
% Split data into expected switch/nonswitch islet predictions
% This determines what shape the data is plotted with
train_switch_input = [];
train_nonswitch_input = [];
for col = 1 : n
    if (modeldata(input_layer_size+1,col) == 1)
        train_switch_input = [train_switch_input input(:,col)];
    else
        train_nonswitch_input = [train_nonswitch_input input(:,col)];
    end
end

run(fig_num) = figure(); fig_num = fig_num + 1;
title_name = strcat('Eta: ',num2str(learning_rate),' Iterations: ', num2str(iterations),' Expected Outcomes Training Data');
plot_expected_outcome(title_name,train_switch_input,train_nonswitch_input,gstar,gkatp_mean_value);



% INITALIZE RANDOM SYNAPSE WEIGHTS AND BIASES
synapse0 = 2*randn(input_layer_size,hidden_layer_size);
synapse1 = 2*randn(hidden_layer_size,output_layer_size);
bias1 = zeros(input_layer_size,hidden_layer_size);
bias2 = zeros(output_layer_size,1);

% Initialize error value before the loop.
errorval = 2;
errorval_counter = [];
% Initialize iteration before loop and update throughout loop.
iteration = 0;
iteration_counter = [];
% Initialize "best" value holders
best_errorval = 1;
best_synapse0 = synapse0;
best_synapse1 = synapse1;
predicted_training_outcomes = [];
% Initialize things for calculating decision boundary
x0 = 0;
y0 = 0;
full_decision = zeros(input_layer_size,n-testn);
for col = 1 : 1000
    full_decision(1,col) = x0;
    full_decision(2,col) = y0;
    x0 = 0.55*rand()+0.65; % Give value in range (0.65,1.2) for plot
    y0 = 0.55*rand()+0.65; % Give value in range (0.65,1.2) for plot
end


% TRAIN THE NETWORK
% use a for or while loop to iterate through the training process
for iter=[1:iterations]
    % Update iteration counter value
    iteration = iteration + 1;
    
    for sample=[1:n-testn]
        % Feed data through the network
        layer0=training_data(1:input_layer_size,sample);
        layer1=activate(layer0',synapse0,bias1); % (training set size X hidden layer size) vector
        layer2=activate(layer1,synapse1,bias2); % 183x1 (hidden layer size X output size) vector
        
        % Backward Pass
        layer2_error = layer2 - training_data(input_layer_size+1,sample)';
        layer2_delta = layer2.*(1-layer2).*(layer2_error);

        layer1_error = layer2_delta.*synapse1.';
        layer1_delta = layer1.*(1-layer1).*(synapse1'.*layer2_delta);

        % Calculate root-mean-squared-error based on difference between output
        % and targets
        % errorval = sqrt(mean(layer2_error.^2)); % If more than one
        % training point at a time
        errorval = mean(abs(layer2_error)); % Simplified since there is only one training point being passed through
        
        % Gradient Step
        synapse0 = synapse0 - learning_rate.*(layer0.*layer1_delta);
        synapse1 = synapse1 - learning_rate.*(layer1'*layer2_delta);
        bias1 = bias1 - learning_rate*layer1_delta;
        bias2 = bias2 - learning_rate*layer2_delta;

        % If errorval is better than the best_errorval, update best_errorval
        % and best synapse values.
        if errorval < best_errorval
            best_errorval = errorval;
            best_synapse0 = synapse0;
            best_synapse1 = synapse1;
        end

        
    end
    % Display error value every 1000 iterations
    if mod(iteration,1000) == 0
        fprintf("Iteration Number: %f, Error value: %f\n\n", iteration, errorval);
    end
    % Save values every 10 iterations
    if mod(iteration,1) == 0
        iteration_counter = [iteration_counter iteration];
        errorval_counter = [errorval_counter errorval];
    end
end


% PLOT ERROR OVER TRAINING ITERATIONS
run(fig_num) = figure();fig_num = fig_num + 1;
plot(iteration_counter,errorval_counter,'b*-');
title_name = strcat('Eta: ',num2str(learning_rate),' Iterations: ', num2str(iterations),' Error over Training Iterations');
title(title_name);
hold off;

% Display best error value
fprintf("Best Training Error Value: %f\n", best_errorval);


% PLOT T/F POS/NEG FOR TRAINING DATA
predicted_training_outputs = [];
for sample=[1:n-testn]
    trainlayer1 = activate(training_data(1:input_layer_size,sample)',best_synapse0,bias1);
    trainlayer2 = activate(trainlayer1,best_synapse1,bias2);
    predicted_training_outputs = [predicted_training_outputs trainlayer2];
end
predicted_training_outputs = predicted_training_outputs(1,:)';

[tp_train,fp_train,tn_train,fn_train,num_tp_train,num_fp_train,num_tn_train,num_fn_train] = classification(1,n-testn,0.5,training_data(1:input_layer_size,:),predicted_training_outputs,training_data(input_layer_size+1,:)');
run(fig_num) = figure();fig_num = fig_num + 1;

title_name = strcat('Eta: ',num2str(learning_rate),' Iterations: ', num2str(iterations),' Training Data: T/F Pos/Neg');
plot_tfposneg(title_name,tp_train,fp_train,tn_train,fn_train,gstar,gkatp_mean_value);


% TEST TRAINED MODEL WITH TEST DATA
% Manipulate testing data by multiplying by the two sets of weights and
% applying the sigmoid activation function.
predicted_test_outputs = [];
for sample=[1:testn+1]
    testlayer1 = activate(test_inputs(:,sample)',best_synapse0,bias1);
    testlayer2 = activate(testlayer1,best_synapse1,bias2);
    predicted_test_outputs = [predicted_test_outputs testlayer2];
end
predicted_test_outputs = predicted_test_outputs(1,:)';

% Calculate test error.
disp(size(predicted_test_outputs));
disp(size(test_targets'));
trained_testerr = sqrt(mean((predicted_test_outputs - test_targets').^2));%immse(predicted_test_outputs, test_targets');
fprintf("Trained: Mean Squared Error with Testing data: %f\n", trained_testerr);


% PLOT OUTCOME OF TESTING DATA (EVENTUALLY WITH DECISION BOUNDARY)
% Classify Data
[test_switch_input,test_nonswitch_input] = split_data(1,testn,0.5,test_inputs,predicted_test_outputs);
title_name = strcat('Eta: ',num2str(learning_rate),' Iterations: ', num2str(iterations),' Testing Data Outcome: Error: ',num2str(trained_testerr));
run(fig_num) = figure();fig_num = fig_num + 1;
plot_expected_outcome(title_name,test_switch_input,test_nonswitch_input,gstar,gkatp_mean_value);


% PLOT TRUE/FALSE POSITIVE/NEGATIVE OF TESTING DATA
[tp,fp,tn,fn,num_tp,num_fp,num_tn,num_fn] = classification(1,testn,0.5,test_inputs,predicted_test_outputs,test_targets');
run(fig_num) = figure();fig_num = fig_num + 1
title_name = strcat('Eta: ',num2str(learning_rate),' Iterations: ', num2str(iterations),' Testing Data: T/F Pos/Neg, Error: ', num2str(trained_testerr));
plot_tfposneg(title_name,tp,fp,tn,fn,gstar,gkatp_mean_value);

figure_filename = strcat('Figures ',datestr(now));

savefig(run,figure_filename);
close(run);

disp(num_tp);disp(num_tn);disp(num_fp);disp(num_fn);

end