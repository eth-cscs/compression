#!/usr/bin/env python3

# This Python script computes RMSZ values as described in the paper
# "A Methology for Evaluating the Impact of Data Compression on Climate
# Simulation Data" by Baker et al. The file paths are read from the command
# line input arguments but the names of the variables are hardcoded for
# convenience (for the CESM model, excluding variables for which the
# compression failed). The parsing of the file name should be adapted to the
# file name formatting used for the reconstructed data files.

import numpy as np
import netCDF4 as nc
import argparse, sys

# index for first NetCDF dimension
time_index = 1

# tolerance for std, otherwise excluded in calculations
threshold = 1e-12

# variables for which the score will be calculated
variables_3d = "CCN3 dst_a1 dst_a3 CLDICE IWC CLDLIQ DCQ T CMFDQ pom_a1 wat_a2 wat_a3 wat_a1 SLV DMS CLDFSNOW DTV H2O2 ANRAIN dgnd_a03 FREQL ICIMR Z3 WSUB FREQS AQSNOW NUMICE Q VQ UU WTKE num_a2 num_a3 num_a1 ANSNOW FREQI TOT_CLD_VISTAU ABSORB dgnd_a02 VD01 TOT_ICLD_VISTAU dgnd_a01 soa_a2 soa_a1 AREL AREI OMEGA RELHUM bc_a1 FICE AWNC DTCOND V CONCLD AWNI CMFDQR NUMLIQ ICWMR H2SO4 LCLOUD ncl_a3 ncl_a2 ncl_a1 ICLDIWP SL ICLDTWP so4_a1 so4_a2 so4_a3 QRL QRS OMEGAT EXTINCT CLOUD FREQR SO2 VT VU VV U CMFDT AQRAIN QT dgnw_a02 dgnw_a03 SOAG dgnw_a01 QC".split()
variables_3di = "SLFLX VFLX KVM KVH TKE CMFMCDZM QTFLX UFLX CMFMC".split()
variables_2d = "FLNTC FREQSH TMQ FREQZM LWCF SSAVIS TGCLDIWP U10 SNOWHLND TREFMNAV TAUTMSX TROP_Z DSTSFMBL SNOWHICE AODVIS TSMN PRECCDZM TROP_P FSNSC TAUTMSY RHREFHT PCONVT AODMODE3 AODMODE2 AODMODE1 T850 FSNTC OCNFRAC TS PRECSH PRECT SOLIN PCONVB dst_a3SF FSDSC WGUSTD AODDUST1 AODDUST3 PBLH TREFHT FSUTOA FLUT TSMX PHIS LHFLX FSNT TGCLDLWP SSTSFMBL FLNT ATMEINT ICEFRAC PRECL PRECC TROP_T BURDEN2 BURDEN1 FLNSC CDNUMC CLDHGH QFLX PSL FLNS PS CLDTOT SWCF AEROD_v LANDFRAC dst_a1SF CLDMED FSDS QREFHT FSNS PRECSC LND_MBL TAUY SRFRAD TREFMXAV BURDEN3 FSNTOA CLDLOW PRECSL FLDS FSNTOAC FLUTC SHFLX T700 TGCLDCWP TAUX AODABS".split()
# change this line for selecting a subset of the variables
variables = variables_3d + variables_3di + variables_2d

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
args = parser.parse_args()

# open files
datasets = [nc.Dataset(f, "r", format="NETCDF4") for f in args.files]

print("VARIABLE", "ENSEMBLE", "RMSZ", "EXCLUDED", sep="\t")
for variable in variables:

    # combine variable data in common array
    if variable in variables_2d:
        variable_data = np.dstack((d.variables[variable][time_index,:] for d in datasets))
    else:
        variable_data = np.dstack((d.variables[variable][time_index,:,:] for d in datasets))
    n_others = variable_data.shape[2]-1

    for i, dataset in enumerate(datasets):

        # parse filename
        n = args.files[i].split("/")[-1].split("_")[2]

        # we do the mean and std manually so we can use slicing and avoid data
        # being copied. this is much faster than explicitly constructing the
        # array with the entries for all other ensemble members.
        this = variable_data[:,:,i]
        mean_others = (variable_data[:,:,:i].sum(axis=2) + variable_data[:,:,(i+1):].sum(axis=2)) / n_others
        std_others = np.sqrt((np.square(variable_data[:,:,:i] - mean_others[:,:,None]).sum(axis=2) + np.square(variable_data[:,:,(i+1):] - mean_others[:,:,None]).sum(axis=2)) / n_others)

        # we exclude the entries for which the standard deviation is very
        # small. however, it seems that the rmsz values for these variables
        # are all over the place and thus not really of any use.
        mask_others = np.abs(std_others) > threshold
        n_excluded = np.sum(~mask_others)

        if n_excluded < std_others.size:
            if n_excluded:
                print("WARNING: some stddev are 0 for variable", variable, "run", n, file=sys.stderr)
            rmsz = np.sqrt(np.square( (this[mask_others] - mean_others[mask_others]) / std_others[mask_others]  ).mean())
        else:
            print("WARNING: all stddev are 0 for variable", variable, "run", n, file=sys.stderr)
            rmsz = 0

        # parse filename and print results
        print(variable, n, rmsz, n_excluded, sep="\t")
