function output=cluster_EOF(x,opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Set (random) intial cluster affiliations gamma(k,t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T=size(x,1);gamma=zeros(opt.K,T);
for t=1:T
   k=floor(opt.K*rand(1))+1; 
   gamma(k,t)=1; 
end
dL=100000;
n_iter=1
while and(dL>opt.tol,n_iter<opt.MaxIter)
   %%% Identifikation of optimal theta (mu and T) for fixed gamma
   [res,mu,T,L(n_iter)]=ClusterMeanCov(x,gamma); 
   %%% Identifikation of optimal gamma for a fixed theta (mu and T)
   for t=1:T
      gamma(:,t)=0; 
      [~,k]=min(res(:,t));
      gamma(k,t)=1;
   end
   if sum(sum(gamma.*res))>L(n_iter)
      disp('Funktional is increasing by gamma-optimization: error!');
      keyboard       
   end
   if n_iter>1
       dL=L(n_iter-1)-L(n_iter);
   end
   if and(dL<0,abs(dL)<1e-12)
       dL=abs(dL);
   elseif and(dL<0,abs(dL)>1e-12) 
      disp('Funktional is increasing between iterations: error!');
      keyboard
   end
   n_iter=n_iter+1
end
output.L=L;
output.gamma=gamma;
