v% Written by William Sawyer, CSCS
% Matlab code calculating the theta translations at iteration s
% for PCA clustering

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [Theta] = theta_s(Gamma, X)

%
%   \gamma \in \Re^{Nt \times K}
%   X \in \Re^{nj \times Nt}
%   \theta \in \Re^{nj \times K}  
%
%   \forall i   \;\; 
%     \theta_i = \frac{\sum_{t=1}^{Nt} \gamma_i (t) x(t)}
%                     {sum_{t=1}^{Nt} \gamma_i (t)}
%

K=size(Gamma,2);
Theta = zeros(size(X,1),K);

sum_gamma=sum(Gamma,1);

for i=1:K
    if (sum_gamma(i) > 0)
        Theta(:,i) = X*Gamma(:,i)/sum_gamma(i);
    end
end

%
% Distributed memory version:  mpi_allreduce(Theta)
%

end