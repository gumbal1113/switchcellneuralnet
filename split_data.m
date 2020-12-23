function [switch_input,nonswitch_input] = split_data(first_col,last_col,decision_val,inputs,outputs)

switch_input = [];
nonswitch_input = [];

for col = first_col : last_col
    if (outputs(col,:) >= decision_val)
        switch_input = [switch_input inputs(:,col)];
    else
        nonswitch_input = [nonswitch_input inputs(:,col)];
    end
end