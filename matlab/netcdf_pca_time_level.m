function [] = netcdf_pca_devel( filename, varname, quality )

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

m=5;   % Size of compression subspace
K=10;
max_iter = 1000;

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

if (size(datasize,2) == 3 )
    n=datasize(3);
elseif (size(datasize,2) == 4 )
    n=datasize(3)*datasize(4);
else
    data_dims_not_supported = size(datasize)
end

X=reshape(data,datasize(1)*datasize(2),n);    %  see help squeeze to remove singleton dims

clear rel_err compr_factor rel com compr_factor_stat idx
 
%%%Xd=double(X);
%%%save -ascii X Xd
X = X';   % use the transpose to better correspond to old code.
[Ntl,nl]=size(X)

X(:,1:48:240)

EV = zeros(Ntl,m,K);

%tol=[1e-2 1e-3 5e-4 1e-4 5e-5 1e-6];
tol=[1e-2];
Lold = realmax('single');
for ind_tol=1:length(tol)
    iter = 0;
    stop = 0;
    GammaInd = gamma_ind_zero(nl,K);
    while ~stop & (iter < max_iter)
        Theta = theta_ind_s(GammaInd, X, K);
        for i=1:K
            Nonzeros = find(GammaInd==i);
            Xtr = bsxfun(@minus,X(:,Nonzeros),Theta(:,i));
            [TT(:,i),j,flag] = lanczos_elman_ind(Xtr,i,1,1e-5,20,0);
        end
        LafterTT = L_value_ind(GammaInd, TT, X, Theta)
        GammaInd = gamma_ind_s(X,Theta,TT);
        Lnew = L_value_ind(GammaInd, TT, X, Theta)
        stop = (Lold - Lnew < tol);
        iter = iter + 1;
        Lold = Lnew;
    end
    %
    % Perform the actual compression
    %
    for i=1:K
        Nonzeros = find(GammaInd==i);
        Xtr = bsxfun(@minus,X(:,Nonzeros),Theta(:,i)); % Theta is new origin
        [EV(:,:,i),j,flag] = lanczos_elman_ind(Xtr,i,m,ind_tol,100,0);
    end
    Lfinal = L_value_mlarge_ind(GammaInd, EV, X, Theta)
end

%save X_comp_Lossless X
%list_compr_lossless=dir('X_comp_Lossless.mat');
%figure;loglog((tol),(compr_factor),'o-','LineWidth',2,'MarkerSize',10);
%hold on;loglog(tol,(compr_factor_stat)','x:','LineWidth',2,'MarkerSize',10);
%loglog(tol,list.bytes/list_compr_lossless.bytes*ones(size(tol)),'k--','LineWidth',2)
%xlabel('Mean Relative Compression Error','FontSize',16);
%ylabel('Compression Factor','FontSize',16)
%title(['Lossy-Lossless Compression of ' varname ' Data'],'FontSize',16)
%legend('Non-homogenous EOF + Lossless','Standard EOF +Lossless','Purely Lossless Compression');
%set(gca,'FontSize',14,'LineWidth',2)

% compr_factor_stat

end