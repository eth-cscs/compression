% Written by William Sawyer (CSCS)
% Matlab code for determining the new gamma probabilities

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [GammaInd] = gamma_ind_s(X,Theta,TT)

%
%   \gamma \in \Re^{Nt \times K}
%   X \in \Re^{nj \times Nt}
%   \theta \in \Re^{nj \times K}
%   \cal{T} \in \Re^{nj \times K}  % singleton eigenvectors
%
%  It is advantageous to pass in the already translated data set:
%  xtr(t) = x(t) - \theta_i^{(s)},
% 
%  We seek the new Gamma which minimizes the functional
%
%
%   \begin{equation}
%      \gamma_i^{(s)}(t) = \left\{ \begin{array}{l} 1, 
%         \mbox{if} i=argmin_{\bar{i} \| (x(t) - \theta_i^{(s)}) - 
%                       \cal{T}_{\bar{i}} \cal{T}_{\bar{i}}^T 
%                       (x(t) - \theta_i^{(s)}) \|
%   \end{equation}
%
%  Since all the probability is always packed into one index, GammaInd
%  is only an index vector.
%

K  = size(TT,2);
GammaInd = zeros(size(X,2),1);
colnorm = zeros(size(X,2),K);

for i = 1:K  % Over the meta-stable subspaces (independent)
    Xtr = bsxfun(@minus,X,Theta(:,i));   % Theta is new origin
    Func=Xtr - TT(:,i)*TT(:,i)'*Xtr;
    colnorm(:,i)=sqrt(sum(Func.^2,1))';      % Column norms $\in \Re^{Nt}$
end
[val GammaInd]=min(colnorm,[],2);    	% Allreduce overall all PEs??
					%  Illia says no -- local computation

end
