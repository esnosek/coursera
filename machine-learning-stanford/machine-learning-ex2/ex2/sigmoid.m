function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% for i=1:size(z,1)
%   for j=1:size(z,2)
%     g(i,j) = sigmoid_tmp(z(i,j));
%   end;
% end;

  g = arrayfun(@(x) sigmoid_tmp(x), z);
  
  function val = sigmoid_tmp(tmp)
     val = 1 / (1+exp(-tmp));
  end
% =============================================================

end
