% Wvritten by Danny Bivckson, CMU
% Matlab code for running Lanczos algorithm (finding eigenvalues of a 
% PSD matrix)

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [val] = L_value_ind(GammaInd, TT, X, Theta)

%
%   \gamma \in \Re^{Nt \times K}
%   TT \in \Re^{nj \times K}
%   \theta \in \Re^{nj \times K}
%
%   Determine the value of the functional:
%
%   \begin{equation}
%      L(\gamma,\cal{T},\theta) = 
%      \frac{\sum_{t=1}^{Nt} \frac{\sum_{i=1}^{K} \gamma_i (t) 
%      \| (x(t) - \theta_i) - \cal{T}_i \cal{T}_i^T(x(t) - \theta_i)^T  \|
%   \end{equation}
%
%   We defined this as an eigenvalue problem:
% 
%   \begin{equation}
%     \forall i   \;\; 
%     \sum_{t=1}^{Nt} \gamma_i^{(s)} (t) (x(t) - \theta_i^{(s)}) 
%                     (x(t) - \theta_i^{(s)})^T \cal{T}_i^{(s)}
%       = \Lambda_i \cal{T}_i^{(s)} \sum_{t=1}^{Nt} \gamma_i^{(s-1)} (t)
%   \end{equation}
% 
%   The final summation $\sum_{t=1}^{Nt} \gamma_i^{(s-1)} (t)$
%   can be dropped since only the eigenvectors $\cal{T}_i^{(s)}$
%   are sought
%
%   The eigenvectors are calculated using the Lanczos algorithm
%   In this case, only the m=1 calculation is performed.  The first 
%   Ritz vector is random, the second should correspond to the 
%   largest eigenvalue, and is the one we want.

K = size(Theta,2);
nj = size(Theta,1);
Nt = size(X,2);
Xtr = zeros(nj,Nt);
Xfinal = zeros(nj,Nt);
colnorm = zeros(Nt,1);

val = 0;
for i = 1:K  % Over the meta-stable subspaces (independent)
    %
    Nonzeros=find(GammaInd==i);
    Xtr = bsxfun(@minus,X,Theta(:,i));   % Theta is new origin
    Xfinal = Xtr - TT(:,i)*TT(:,i)'*Xtr;
    colnorm = sqrt(sum(Xfinal.^2,1))';
    val = val+sum(colnorm(Nonzeros));
end

end
