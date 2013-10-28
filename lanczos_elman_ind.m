function [EV,j,flag] = lanczos_elman_ind(Xtr,myseed,k,tol,maxits,orth)

% Written be Howard Elman, October 2012
% Alterations by Sawyer for PCA clustering algorithm
% No copyright
%
%Estimate k eigenvalues of a symmetric matrix A or of A^{-1} using the 
%Lanczos algorithm.  Stop when the scaled residual for each of the 
%k eigenvalues (scaled by the eigenvalue) is less than tol.
%
%Input
%
%   Xtr      Translated observations with nonzero probabilities
%   myseed   To ensure that all nodes create the same Vzero
%   k        desired number of eigenvalues
%   tol      tolerance
%   maxits   upper bound on number of steps taken
%   orth     1 to do complete reorthogonalization, otherwise no
%            reorthogonalization

%Output

%   e, the first 5 eigenvalues 
%   j, the number of iterations used
%   flag=0 if tolerance is met, 1 otherwise.

% Initialize

n = size(Xtr,1);
v = zeros(n,2);
T = sparse(k,k);
normr = zeros(k,1);
v(:,1) = randn(n,1); 
v(:,1) = v(:,1)/norm(v(:,1));

W = zeros(n,k);

W(:,1) = v(:,1);

XtrTV = Xtr'*v(:,1);    % Order important: do not build matrix
Av = Xtr*XtrTV;

stop = 0;

j=1;

% Lanczos loop

while ~stop && j<maxits, 

    if j==1, 
       v(:,j+1) = Av;
    else
       v(:,j+1) = Av - gmma*v(:,j-1); 
    end

    delta = v(:,j+1)'*v(:,j);
    v(:,j+1) = v(:,j+1) - delta*v(:,j);
    gmma = norm(v(:,j+1));
    v(:,j+1) = v(:,j+1)/gmma;

% Reorthogonalize
    if orth,
       for i=1:j,
          v(:,j+1) = v(:,j+1) - (v(:,j+1)'*v(:,i))*v(:,i);
       end
    end

    W(:,j+1) = v(:,j+1);
    XtrTV = Xtr'*v(:,j+1);    
    Av = Xtr*XtrTV;
    T(j,j) = delta;
    T(j,j+1) = gmma;
    T(j+1,j) = gmma;

    [UT,E] = eig(full(T(1:j,1:j)));
    V = W(:,1:j)*UT;   %V = orth(V);

% Identify indices of desired eigenvalues
    if j>=k,
        indices = (j:-1:j-k+1);
        r = Xtr*Xtr'*V(:,indices)- V(:,indices)*E(indices,indices); 
        for i=1:k, normr(i) = norm(r(:,i)); end,  %normr
% Stop when relative residual norms of k largest eigenvalues are less than tol 
        stop = max(abs(normr(1:k)./diag(E(indices,indices))))<tol;

    end
    j = j+1;
end
EV = V(:,indices);   % Return the eigenvectors
flag = ~stop;
% LargestEigenvalues_Iterations_flag = [diag(E(indices,indices))' j flag]

end
