function [data, compress, reconst, factor] = netcdf_pca( filename, varname, quality )

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
 
    % Get raw data
    if( strcmp(name,varname) )
        name
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

offsetX=mean(X,2)*ones(1,n);
Xprime=(X-offsetX);
[U,S,V]=svd(Xprime,'econ');

sigma=diag(S)   % Singular values are sorted largest to smallest

p=1;   % At least one singular vector
while ( p < n ) & ( sigma(p)>sigma(1)*quality )
    p=p+1;
end

factor = n/p

compress=U(:,1:p)'*Xprime;   % Compressed form as per Moeslund PCA Intro
tmp=U(:,1:p)*compress+offsetX; % Reconstructed field

max_abs_reconstruction_error=max(max(tmp-X))
min_abs_reconstruction_error=min(min(tmp-X))

if (dim3 == 0 )
    reconst=reshape(tmp,dim1,dim2,n);
else
    reconst=reshape(tmp,dim1,dim2,dim3,n);
end

clear ndims nvars natts ii jj tmpstruct idx ncid filename
clear fieldname name vartype dimids varinfoname date_offset
clear offsetX Xprime tmp U S V

end
