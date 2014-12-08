function [X,y] = read_mnist(filename)
  train = dlmread(filename,",",1,0);
  X = train(:,2:end);
  y = train(:,1);
end
