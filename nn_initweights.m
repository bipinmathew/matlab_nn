function [w] = nn_initweights(nn_structure)
num_weights = 0;
L = length(nn_structure);
for i=2:L
	num_weights+=(1+nn_structure(i-1))*nn_structure(i);
end

%w = rand(num_weights,1);
w = ones(num_weights,1);

end
