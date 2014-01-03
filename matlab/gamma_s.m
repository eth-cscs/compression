% Written by William Sawyer (CSCS)
% Matlab code for determining the new gamma probabilities

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [Gamma] = gamma_s(Xtr,TT)

%
%   \gamma \in \Re^{Nt \times K}
%   X \in \Re^{nj \times Nt}
%   \theta \in \Re^{nj \times K}
%   \cal{T} \in \Re^{nj \times K}  % singleton eigenvectors
%
%  It is advantageous to pass in the already translated data set:
%  xtr(t) = x(t) - \theta_i^{(s)},
% 
%   We seek the new Gamma which minimizes the functional
%
%
%   \begin{equation}
%      \gamma_i^{(s)}(t) = \left\{ \begin{array}{l} 1, 
%         \mbox{if} i=argmin_{\bar{i} \| (x(t) - \theta_i^{(s)}) - 
%                       \cal{T}_{\bar{i}} \cal{T}_{\bar{i}}^T 
%                       (x(t) - \theta_i^{(s)}) \|
%   \end{equation}
%
%   Sinc
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

K = size(Gamma,2);
nj = size(X,1);
Nt = size(X,2);
Gamma = zeros(Nt,K);
colnorm = zeros(Nt,K);

for i = 1:K  % Over the meta-stable subspaces (independent)
    Func=Xtr - TT(:,i)*TT(:,i)'*Xtr;
    colnorm(:,i)=sqrt(sum(Func.^2,1))';      % Column norms $\in \Re^{Nt}$
end
[val ind]=min(colnorm,[],2);

for t=1:Nt
  Gamma(t,ind(t)) = 1;
end

end
