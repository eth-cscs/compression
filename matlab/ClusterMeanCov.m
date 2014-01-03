function [res,mu,T,L]=ClusterMeanCov(x,gamma); 
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
dim=size(x,2);
[K,T]=size(gamma);
mu=zeros(K,dim);TT=zeros(K,dim);res=zeros(K,T);
for k=1:K
    Nk=0;
    for t=1:T
        mu(k,:)=mu(k,:)+gamma(k,t)*x(t,:);
        Nk=Nk+gamma(k,t);
    end
    mu(k,:)=mu(k,:)./Nk;
    cov_m=zeros(dim);
    for t=1:T
        r=(x(t,:)-mu(k,:));
        cov_m=cov_m+gamma(k,t)*r'*r;
    end
    opts.tol=1e-14;
    [V,D]=eigs(cov_m./Nk,1,'la',opts);
    TT(k,:)=V(:,1)';
    for t=1:T
        r=(x(t,:)-mu(k,:))';
        res(k,t)=norm(r-TT(k,:)'*TT(k,:)*r,2)^2;
    end    
end
L=sum(sum(gamma.*res))
