% Wvritten by Danny Bivckson, CMU
% Matlab code for running Lanczos algorithm (finding eigenvalues of a 
% PSD matrix)

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [TT] = lanczos_pca_mone(Gamma, X, Theta,myseed)

%
%   \gamma \in \Re^{Nt \times K}
%   X \in \Re^{nj \times Nt}
%   \theta \in \Re^{nj \times K}
%
%   The matrix whose eigenvectors are sought is:
%
%   \begin{equation}
%      A_i = \frac{\sum_{t=1}^{Nt} \gamma_i^{(s)} (t) 
%                  (x(t) - \theta_i^{(s)}) (x(t) - \theta_i^{(s)})^T}
%               {\sum_{t=1}^{Nt} \gamma_i^{(s)} (t)}
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

K = size(Gamma,2);
nj = size(X,1);
Nt = size(X,2);
R  = zeros(2,2);       % Tridiagonal matrix
s = RandStream('mcg16807','Seed',myseed);  Important that all PEs reset
RandStream.setDefaultStream(s);
Vzero = rand(nj,1);    % Initial vector (random)  All PEs should have same
Vzero = Vzero/norm(Vzero,2);
XtrTV = zeros(Nt,1); 


V = zeros(nj,1);

TT = zeros(nj,K);

for i = 1:K  % Over the meta-stable subspaces (independent)
    %  Basically:  w = gamma*Xtr*Xtr'*Vzero;  %
    Xtr = bsxfun(@minus,X,Theta(:,i));   % Theta is new origin
    XtrTV = Xtr'*Vzero;    % Order important: do not build matrix
    w = bsxfun(@times,Xtr,Gamma(:,i)')*XtrTV;
    % mpi_allreduce(w);
    R(1,1) = w'*Vzero;
    w = w - R(1,1)*Vzero;
    R(2,1) = norm(w,2);
    R(1,2) = R(2,1);
    V = w/R(1,2);
    XtrTV = Xtr'*V;
    w = bsxfun(@times,Xtr,Gamma(:,i)')*XtrTV - R(1,2)*Vzero;
    R(2,2) = w'*V;
    % Vnew should represent the second eigenvector
    if ( norm(R) >= eps ) 
        [U,lambda] = eig(R);   % Eigenvectors/values of R
        TT(:,i) = [Vzero V]*U(:,2);   % Ritz vector of A to largest EV
    else  % generate a random orthonormal vector
        qorth = rand(nj,1);
        qorth = qorth -  (qorth'*qorth)*qorth;
        TT(:,i) = qorth / norm(qorth);
    end
end
    
% disp(['approximating eigenvalues are: ', num2str(eig(R)')]);
end
