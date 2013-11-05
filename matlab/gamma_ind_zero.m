% Written by William Sawyer (CSCS)
% Matlab code for determining the new gamma probabilities

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [GammaInd] = gamma_ind_zero(Nt,K)

%
%  Determine a random $\gamma \in \Re^{Nt \times K}$ with
%
%  \begin{eqnarray}
%     \gamma_i (t) \ge 0 &&  \;\; \forall i,t
%     \sum_{i=1}^K gamma_i(t) = 1 && \forall t
%   \end{equation}
%
%  Since we consider only distributions where all probability is
%  contained in one index, the gamma_ind version utilizes only an 
%  index vector
%

GammaInd  = randi(K,Nt,1);

end
