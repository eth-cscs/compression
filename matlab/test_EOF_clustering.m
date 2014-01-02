clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Create a test data set as an output of Gaussian mixture
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu1=[0 0];mu2=[0 0];
S1=[20 0;0 1];S2=[1 0;0 20];
n=100;
gamma=[ones(1,n) 2*ones(1,n) ones(1,n) 2*ones(1,n)];
T=length(gamma);
for t=1:T
x(t,:)=(gamma(t)==1)*mvnrnd(mu1,S1)+(gamma(t)==2)*mvnrnd(mu2,S2);
end
figure;plot(x(:,1),x(:,2),'.');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Set clustering parameters: tolerance, number of repetitions
%%% (annealings), maximal number of subspace iterations
%%% and number of clusters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt.K=2
opt.N_anneal=10
opt.tol=1e-6;
opt.MaxIter=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Repeat the clustring opt.N_anneal times with random initial gamma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:opt.N_anneal
output{i}=cluster_EOF(x,opt);
L_min(i)=min(output{i}.L);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Identify the best clustering instance (according to the
%%% minimum value of the clustering functional L) and plot results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,jj]=min(L_min);
figure;plot(output{jj}.L,':<');xlabel('Number of Subspace Iteration i');ylabel('Value of the clustering functional L^i')
col='rgbmck';
figure;hold on;
for k=1:opt.K
   [~,ii]=find(output{jj}.gamma(k,:)==1);
   plot(x(ii,1),x(ii,2),[col(k) 'x']);
end
