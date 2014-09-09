#!/bin/bash

# This script can be used as a template for generating batch files to run
# the compression with different parameters. For each scenario, it creates
# a *.batch file to be used with the SLURM sbatch command. It also creates
# a .run file with the actual command to run, as the long command with all the
# variable names gets truncated otherwise. Everything except for the output
# folder for the batch/run files is hardcoded and has to be adapted to the
# desired configuration.
#
# Usage: generate_batch_files.sh FOLDER 

# configuration
KMIN=8
KMAX=8
KSTEP=1
MMIN=3
MMAX=3
MSTEP=1
DIRECTION=vertical
EXEC="/scratch/daint/schmidma/usi_compression/usi_compression_minlin_host"
DATA="/scratch/daint/schmidma/usi_compression/data/cesm1_1.FC5.ne30_g16.000.cam.h0.0001-01-01-00000.time1.nc"
OUTPUT_DIR="/scratch/daint/schmidma/usi_compression/output"

# read output dir from arguments
if [ $# -ne 1 ]
then
    echo "Usage: generate_batch_files.sh OUTPUT_DIRECTORY"
    exit 1
fi
BATCH_DIR=$(pwd)/$1

# specify variables (for the CESM model)
VAR_3D="CCN3 dst_a1 dst_a3 CLDICE IWC CLDLIQ DCQ T CMFDQ pom_a1 wat_a2 wat_a3 wat_a1 SLV DMS CLDFSNOW DTV H2O2 ANRAIN dgnd_a03 FREQL ICIMR Z3 WSUB FREQS AQSNOW NUMICE Q VQ UU WTKE num_a2 num_a3 num_a1 ANSNOW FREQI TOT_CLD_VISTAU ABSORB dgnd_a02 VD01 TOT_ICLD_VISTAU dgnd_a01 soa_a2 soa_a1 AREL AREI OMEGA RELHUM bc_a1 FICE AWNC DTCOND V CONCLD AWNI CMFDQR NUMLIQ ICWMR H2SO4 LCLOUD ncl_a3 ncl_a2 ncl_a1 ICLDIWP SL ICLDTWP so4_a1 so4_a2 so4_a3 QRL QRS OMEGAT EXTINCT CLOUD FREQR SO2 VT VU VV U CMFDT AQRAIN QT dgnw_a02 dgnw_a03 SOAG dgnw_a01 QC"
VAR_3Di="SLFLX VFLX KVM KVH TKE CMFMCDZM QTFLX UFLX CMFMC"
VAR_2D="FLNTC FREQSH TMQ FREQZM LWCF SSAVIS TGCLDIWP U10 SNOWHLND TREFMNAV TAUTMSX TROP_Z DSTSFMBL SNOWHICE AODVIS TSMN PRECCDZM TROP_P FSNSC TAUTMSY RHREFHT PCONVT AODMODE3 AODMODE2 AODMODE1 T850 FSNTC OCNFRAC TS PRECSH PRECT SOLIN PCONVB dst_a3SF FSDSC WGUSTD AODDUST1 AODDUST3 PBLH TREFHT FSUTOA FLUT TSMX PHIS LHFLX FSNT TGCLDLWP SSTSFMBL FLNT ATMEINT ICEFRAC PRECL PRECC TROP_T BURDEN2 BURDEN1 FLNSC CDNUMC CLDHGH QFLX PSL FLNS PS CLDTOT SWCF AEROD_v LANDFRAC dst_a1SF CLDMED FSDS QREFHT FSNS PRECSC LND_MBL TAUY SRFRAD TREFMXAV BURDEN3 FSNTOA CLDLOW PRECSL FLDS FSNTOAC FLUTC SHFLX T700 TGCLDCWP TAUX AODABS"
VAR_2Dnw="DSTODXC SSTODXC SFCLDLIQ ORO AODDUST2 DSTSFWET SSTSFDRY SFNUMICE SSTSFWET SFNUMLIQ DSTSFDRY SFCLDICE" # these variables were not working correctly

for K in $(seq -w $KMIN $KSTEP $KMAX)
do
    for M in $(seq -w $MMIN $MSTEP $MMAX)
    do
        # create file names
        BATCH_FILE="${BATCH_DIR}/compression_K${K}_M${M}.batch"
        RUN_FILE="${BATCH_DIR}/compression_K${K}_M${M}.run"
        OUTPUT_FILE="${OUTPUT_DIR}/compression_K${K}_M${M}.out"
        ERROR_FILE="${OUTPUT_DIR}/compression_K${K}_M${M}.err"

        # set compression options
        if [ $direction == "horizontal" ]
        then
            OPTS="-i time=0 -c ncol -h -s"
        else
            OPTS="-i time=0 -c lev ilev -s"
        fi

        # write run file (we use this to avoid aprun truncating the command)
        echo "#!/bin/bash" > $RUN_FILE
        echo "$EXEC $DATA $OPTS -K $K -M $M -v $VAR_3D $VAR_3Di $VAR_2D" >> $RUN_FILE
        chmod +x $RUN_FILE

        # write batch file
        echo "#!/bin/bash" > $BATCH_FILE
        echo "#SBATCH --job-name=K${K}M${M}${direction:0:1}" >> $BATCH_FILE
        echo "#SBATCH --ntasks=8" >> $BATCH_FILE
        echo "#SBATCH --ntasks-per-node=1" >> $BATCH_FILE
        echo "#SBATCH --time=00:30:00" >> $BATCH_FILE
        echo "#SBATCH --output=${OUTPUT_FILE}" >> $BATCH_FILE
        echo "#SBATCH --error=${ERROR_FILE}" >> $BATCH_FILE
        echo "aprun -n8 -N1 ${RUN_FILE}" >> $BATCH_FILE
    done
done
