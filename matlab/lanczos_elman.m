function [e,j,flag] = lanczos_elman(A,k,tol,maxits,format,orth)

% Written be Howard Elman, October 2012
% No copyright

%function [e,W,T] = lanczos(A,k,maxits,format)

%Estimate k eigenvalues of a symmetric matrix A or of A^{-1} using the 

%Lanczos algorithm.  Stop when the scaled residual for each of the 

%k eigenvalues (scaled by the eigenvalue) is less than tol.

%format = 1 for A, -1 for A^{-1}

%Input

%   A        matrix

%   k        desired number of eigenvalues

%   tol      tolerance

%   maxits   upper bound on number of steps taken

%   format   1 to use A, otherwise use A^{-1}

%   orth     1 to do complete reorthogonalization, otherwise no

%            reorthogonalization

%Output

%   e, the first 5 eigenvalues 

%   j, the number of iterations used

%   flag=0 if tolerance is met, 1 otherwise.

% Howard Elman

% October 2012

% Initialize

n = size(A,1);

v = zeros(n,2);

T = sparse(k,k);

normr = zeros(k,1);

rng(22154);

v(:,1) = randn(n,1); 

v(:,1) = v(:,1)/norm(v(:,1));

W = zeros(n,k);

W(:,1) = v(:,1);

if format==1, 

   Av = A*v(:,1);

else

   [L,U]=lu(A);

   Av = U\(L\v(:,1));

end

stop = 0;

j=1;

% Lanczos loop

while ~stop && j<maxits, 

    %%%v(:,j+1) = Av - delta*v(:,j);

    %%%if j>1, v(:,j+1) = v(:,j+1) - gmma*v(:,j-1); end, %norm(v(:,j+1)),

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

    if format==1,

       Av = A*v(:,j+1);

    else

       Av = U\(L\v(:,j+1));

    end

    T(j,j) = delta;

    T(j,j+1) = gmma;

    T(j+1,j) = gmma;

    [UT,E] = eig(full(T(1:j,1:j)));

    V = W(:,1:j)*UT;   %V = orth(V);

% Identify indices of desired eigenvalues

    if j>=k,

        indices = (j:-1:j-k+1);

% Use inverse of computed eigenvalues if format ~= 1

        if format~=1,

           E = diag(1./diag(E));  %diag(E)

        end

        r = A*V(:,indices)- V(:,indices)*E(indices,indices); 

        for i=1:k, normr(i) = norm(r(:,i)); end,  %normr

% Stop when relative residual norms of k largest eigenvalues are less than tol 

        stop = max(abs(normr(1:k)./diag(E(indices,indices))))<tol;

    end

    j = j+1;

end

e = diag(E);

if format~=1, e = flipud(e); end
if format~=1, e = e(1:5); else e=e(end-4:end); end
flag = ~stop;

end
