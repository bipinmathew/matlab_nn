function [X] = read_mnist_test(filename)
  X = dlmread(filename,",",1,0);
end
