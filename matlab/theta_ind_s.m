% Written by William Sawyer, CSCS
% Matlab code calculating the theta translations at iteration s
% for PCA clustering

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [Theta] = theta_ind_s(GammaInd, X, K)

%
%   \gamma \in \Re^{Nt \times K}
%   X \in \Re^{nj \times Nt}
%   \theta \in \Re^{nj \times K}  
%
%   \forall i   \;\; 
%     \theta_i = \frac{\sum_{t=1}^{Nt} \gamma_i (t) x(t)}
%                     {sum_{t=1}^{Nt} \gamma_i (t)}
%
%   In this particular case, all the Gamma probability is aggregated 
%   into one i

Theta = zeros(size(X,1),K);
sum_gamma = zeros(1,K);

for i=1:K
    sum_gamma(i)=sum(GammaInd==i);
end

for i=1:K
    if (sum_gamma(i) > 0)
        Theta(:,i) = sum(X(:,find(GammaInd==i)),2)/sum_gamma(i);
    end
end

%
% Distributed memory version:  mpi_allreduce(Theta)
%

end
