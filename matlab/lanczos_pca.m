% Written by Danny Bickson, CMU
% Matlab code for running Lanczos algorithm (finding eigenvalues of a 
% PSD matrix)

% Code available from: http://www.cs.cmu.edu/~bickson/gabp/
function [] = lanczos_pca_m(gamma, x, theta, m)

[K,nj,Nt] = size(gamma);
R = zeros(2,2);
V = rand(nj,1);                !
V = V/norm(V,2);

    w = gamma*x*x'*V;
    R(1,1) = w'*V;
    w = w - alpha(j)*V;
    R(2,1) = norm(w,2);
    R(1,2) = R(2,1);
    Vnew = w/R(1,2);
    w = A*Vnew - R(1,2)*V;
    R(2,2) = w'(Vnew);
    
T=sparse(m+1,m+1);
for i=2:m+1
    T(i-1,i-1)=alpha(i);
    T(i-1,i)=beta(i+1);
    T(i,i-1)=beta(i+1);
end 
T(m+1,m+1)=alpha(m+2);
V = V(:,2:end-1);
disp(['approximation quality is: ', num2str(norm(V*T*V'-A))]);
disp(['approximating eigenvalues are: ', num2str(eig(T)')]);
end
