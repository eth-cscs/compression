#!/bin/awk -f

# This awk script parses the output files to create a narrow table of
# all statistics about overall compression.
#
# run with: ./overall_statistics.awk FILE1.out FILE2.out ...

BEGIN {
    OFS = "\t";
    print "K", "M", "DIRECTION", "STATISTIC", "VALUE";
}

# read information from filename when processing first line
FNR == 1 {
    split(FILENAME,fn,"[_.]");
    K = substr(fn[2], 2, 10) + 0;
    M = substr(fn[3], 2, 10) + 0;
    direction = fn[4];
    converged = 0;
    lanczos_n = 0;
    lanczos_sum = 0;
    lanczos_min = 1000;
    lanczos_max = 0;
    lanczos_n_converged = 0;
    lanczos_sum_converged = 0;
    lanczos_max_converged = 0;
    lanczos_min_converged = 1000;
}

/^lanczos/ {
    it = $4 + 0;
    if (converged) {
        lanczos_n_converged++;
        lanczos_sum_converged += it;
        if (it < lanczos_min_converged)
            lanczos_min_converged = it;
        if (it > lanczos_max_converged)
            lanczos_max_converged = it;
    }
    else {
        lanczos_n++;
        lanczos_sum += it;
        if (it < lanczos_min)
            lanczos_min = it;
        if (it > lanczos_max)
            lanczos_max = it;
    }
}

/^ Statistics/ {
    if (lanczos_n) {
        print K, M, direction, "lanczos_mean", lanczos_sum / lanczos_n;
        print K, M, direction, "lanczos_min", lanczos_min;
        print K, M, direction, "lanczos_max", lanczos_max;
    }
    else {
        print K, M, direction, "lanczos_mean", 0;
        print K, M, direction, "lanczos_min", 0;
        print K, M, direction, "lanczos_max", 0;
    }
    if (lanczos_n_converged) {
        print K, M, direction, "lanczos_mean_converged", lanczos_sum_converged / lanczos_n_converged;
        print K, M, direction, "lanczos_min_converged", lanczos_min_converged;
        print K, M, direction, "lanczos_max_converged", lanczos_max_converged;
    }
    else {
        print K, M, direction, "lanczos_mean_converged", 0;
        print K, M, direction, "lanczos_min_converged", 0;
        print K, M, direction, "lanczos_max_converged", 0;
    }
}

/L value final/ {
    print K, M, direction, "L_final", $4;
}

/Maximum time for input:/ {
    print K, M, direction, "time_input", $5;
}

/Maximum time for solve:/ {
    print K, M, direction, "time_solve", $5;
}

/Compression ratio:/ {
    print K, M, direction, "compression_ratio", $3;
}

/^ Converged/ {
    print K, M, direction, "iterations", $6;
    converged = 1;
}
