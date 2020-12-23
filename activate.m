% Method Activate() evaluates a sigmoid function
%
% Parameters:
%   x: input vector
%   W: weights vector
%   b: bias term (shifts)
%
% Return:
%   y: output vector
%
function y = activate(x,W,b)
y = (1)./(1+exp(-1*(x*W+b)));