% Written by William Sawyer (CSCS)
% Matlab code for determining the new gamma probabilities

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [Gamma] = gamma_zero(Nt,K)

%
%  Determine a random $\gamma \in \Re^{Nt \times K}$ with
%
%  \begin{eqnarray}
%     \gamma_i (t) \ge 0 &&  \;\; \forall i,t
%     \sum_{i=1}^K gamma_i(t) = 1 && \forall t
%   \end{equation}
%

Gamma = zeros(Nt,K);
ind  = randi(K,Nt,1);
for t=1:Nt
  Gamma(t,ind(t)) = 1;
end

end
