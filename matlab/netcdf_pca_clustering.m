function [compr_factor,compr_factor_stat,tol] = netcdf_pca_clustering( filename, varname, quality )

% [data, compressed, reconstructed] = netcdf_pca( filename, varname, quality )
%
% Principal component analysis on fields generated from climate model
%
% input
%        filename       name of the netcdf file
%        varname        name of the variable of interest
%        quality        cutoff value for sigma_p/sigma_1, perhaps 10^-4
%
% output
%        data           the full field as read from the netcdf file
%        compress       the PCA compressed form of the data
%        reconst        the reconstructed field
%        factor         compression factor n / p
%
% description
%
%
data=0;
compress=0;
reconst=0;
factor=0;


% filename = '/project/csstaff/outputs/echam/echam6/echam_output/t31_196001.01_echam.nc';
ncid = netcdf.open(filename,'NC_NOWRITE');
 
%% Explore the Contents
[numdims,nvars,natts] = netcdf.inq(ncid);
 
%% Get Global attributes Information
for ii = 0:natts-1
    fieldname = netcdf.inqAttName(ncid, netcdf.getConstant('NC_GLOBAL'), ii);
    fileinfo.(fieldname) = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'), fieldname );
end
 
% allocate structure
dimension = repmat(struct('name', '', 'length', 0), numdims, 1);
 
for ii = 1:numdims
    [dimension(ii).name, dimension(ii).length] = netcdf.inqDim(ncid,ii-1);
 
    % padding name for table layout
    padlength   = min(0, length(dimension(ii).name));
    name_padded = [dimension(ii).name repmat(' ', padlength+1)];
 
    fprintf('%s\t\t%d\n', name_padded, dimension(ii).length);
end
 
%% Get the Data
for ii = 1:nvars
    [name, ~, ~, natts] = netcdf.inqVar(ncid,ii-1);
    % Get Variable Attributes
    tmpstruct = struct();
    for jj = 1:natts
        fieldname = netcdf.inqAttName(ncid, ii-1, jj-1);
        tmpstruct.(fieldname) = netcdf.getAtt(ncid, ii-1, fieldname );
    end
 name
    % Get raw data
    if( strcmp(name,varname) )
        %name
        data = netcdf.getVar(ncid,ii-1);
   
    % Replace Missing Numbers (if necessary
        if (isfield(tmpstruct, 'missing_value') )
           data( data == tmpstruct.missing_value ) = NaN;
	    end
   
    % Scale data (if necessary)
	    if( isfield(tmpstruct, 'scale_factor') )
	        data = double(data) * tmpstruct.scale_factor;
	    end
   
    % Apply offset (if necessary)
	    if( isfield(tmpstruct, 'add_offset') )
	        data = data + tmpstruct.add_offset;
	    end
   
    % Transpose data from column major to row major
	    if( isnumeric(data) && ndims(data) > 2 )
	        data = permute(data, [2 1 3:ndims(data)]);
	    elseif ( isnumeric(data) && ndims(data) == 2 )
	        data = data';
	    end
% store attribute and data with appropriate name
	    varinfoname = [name '_info']
	    assignin('caller', varinfoname, tmpstruct);
	    assignin('caller', name, data);
    end
end
 

%% Close File
netcdf.close(ncid);

% 'data' contains the variable of interest
% now perform the PCA using the singular value decomposition

datasize=size(data)

dim1=datasize(1);
dim2=datasize(2);
dim3=0;
if (size(datasize,2) == 3 )
    n=datasize(3);
elseif (size(datasize,2) == 4 )
    dim3=datasize(3);
    n=datasize(4);
else
    data_dims_not_supported = size(datasize)
end

if (dim3 == 0 )
    m=dim1*dim2;
else
    m=dim1*dim2*dim3;
end
X=reshape(data,m,n);    %  see help squeeze to remove singleton dims


clear U S V UU SS VV rel_err compr_factor rel com compr_factor_stat idx
     
 X=X./max(max(abs(X)));
 
%n=size(X,2);
% offsetX=mean(X,2)*ones(1,n);
Xd=double(X);
save -ascii X Xd
%X=X-offsetX;
[T,nnn]=size(X);
[UU,SS,VV]=svd(X,'econ');
iK=[10];
for num_clus=1:length(iK)
    K=iK(num_clus);
    [idx{num_clus},C,L]=kmeans(X,K,'EmptyAction','singleton');sum(L)
    for k=1:K
        [U{k,num_clus},S{k,num_clus},V{k,num_clus}]=svd(X(find(idx{num_clus}==k),:),'econ');
    end
end
tol=[1e-2 1e-3 5e-4 1e-4 5e-5 1e-6];
for ind_tol=1:length(tol)
    ind=0;sS=sum(diag(SS));
    rel=100;
    while and(ind<nnn,rel>tol(ind_tol))
        ind=ind+1;
        rel=mean(mean(abs(UU(:,1:ind)*SS(1:ind,1:ind)*VV(:,1:ind)'-X)));
        %rel=sum(diag(SS(ind+1:nnn,ind+1:nnn)))/prod(size(X));
    end
    com=ind*T+ind*(1+nnn);
    clear UUU VVV SSS
    UUU=UU(:,1:ind);SSS=diag(SS(1:ind,1:ind));VVV=VV(:,1:ind);
    save X_compressed UUU VVV SSS
    list_compr=dir('X_compressed.mat');
    list=dir('X');
    
    compr_factor_stat(ind_tol)=list.bytes/list_compr.bytes;%prod(size(X))/com;
    %ind
    for num_clus=1:length(iK)
        K=iK(num_clus)
        comp=0;
        clear UUU SSS VVV
       for k=1:K
            ind=0;%sS=sum(diag(S(:,:,k,num_clus)));
            rel_err(k)=100;
            while and(ind<nnn,rel_err(k)>tol(ind_tol))
                ind=ind+1;
                rel_err(k)=mean(mean(abs(U{k,num_clus}(:,1:ind)*S{k,num_clus}(1:ind,1:ind)*V{k,num_clus}(:,1:ind)'-X(find(idx{num_clus}==k),:))));
                %rel_err(k)=sum(diag(S(ind+1:nnn,ind+1:nnn,k)))/prod(size(X));
            end
            UUU{k}(:,:)=(U{k,num_clus}(:,1:ind));
            VVV{k}(:,:)=(V{k,num_clus}(:,1:ind));
            SSS{k}=(diag(S{k,num_clus}(1:ind,1:ind)));
            comp=comp+ind*(sum(idx{num_clus}==k))+ind*(1+nnn);
            %ind
        end
        [iiid,mmmd]=find(abs(diff(idx{num_clus}))>0);
        %iiid=idx{num_clus};
        %rm X_compressed         
        save X_compressed UUU VVV SSS iiid mmmd
        list_compr=dir('X_compressed.mat');
        list=dir('X');
        comp=comp+sum(diff(idx{num_clus})>0)+1;
        compr_factor(ind_tol,num_clus)=list.bytes/list_compr.bytes;%prod(size(X))/comp;
    end
end
save X_comp_Lossless X
list_compr_lossless=dir('X_comp_Lossless.mat');
figure;loglog((tol),(compr_factor),'o-','LineWidth',2,'MarkerSize',10);
hold on;loglog(tol,(compr_factor_stat)','x:','LineWidth',2,'MarkerSize',10);
loglog(tol,list.bytes/list_compr_lossless.bytes*ones(size(tol)),'k--','LineWidth',2)
xlabel('Mean Relative Compression Error','FontSize',16);
ylabel('Compression Factor','FontSize',16)
title(['Lossy-Lossless Compression of ' varname ' Data'],'FontSize',16)
legend('Non-homogenous EOF + Lossless','Standard EOF +Lossless','Purely Lossless Compression');
set(gca,'FontSize',14,'LineWidth',2)

compr_factor_stat

end
