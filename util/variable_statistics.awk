#!/bin/awk -f

# This awk script parses the output files to create a narrow table of
# all statistics about individual variables.
#
# run with: ./variable_statistics.awk FILE1.out FILE2.out ...

BEGIN {
    OFS = "\t";
    print "VARIABLE", "K", "M", "STATISTIC", "VALUE";
}

# read information from filename when processing first line
FNR == 1 {
    split(FILENAME,fn,"[_.]");
    K = substr(fn[2], 2, 10) + 0;
    M = substr(fn[3], 2, 10) + 0;
    direction = fn[4];
}

/^ Variable/ {
    variable = $2;
    sub(/:$/, "", variable); # remove trailing ":"
}

/(original data)/ {
    print variable, K, M, "min_original", $1;
    print variable, K, M, "max_original", $2;
    print variable, K, M, "mean_original", $3;
    print variable, K, M, "std_original", $4;
}

/(reconstructed data)/ {
    print variable, K, M, "min_reconstructed", $1;
    print variable, K, M, "max_reconstructed", $2;
    print variable, K, M, "mean_reconstructed", $3;
    print variable, K, M, "std_reconstructed", $4;
}

/maximum error/ {
    print variable, K, M, "max_error", $3;
}

/RMS error/ {
    print variable, K, M, "rms_error", $3;
}

/correlation/ {
    print variable, K, M, "correlation", $2;
}

/SRR/ {
    print variable, K, M, "srr", $2;
}

/PrecisionBits/ {
    print variable, K, M, "precisionbits", $2;
}

END {
}

