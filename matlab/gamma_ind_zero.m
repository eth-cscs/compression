% Written by William Sawyer (CSCS)
% Matlab code for determining the new gamma probabilities

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [GammaInd] = gamma_ind_zero(nl,K)

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

% RANDOM VARIANT:  GammaInd  = randi(K,nl,1);

for m=1:48
for l=m:48:nl
   GammaInd(l) = mod((l-1),K) + 1;   % One-based indices
end
end

end
