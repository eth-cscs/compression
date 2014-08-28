#!/bin/awk -f

# This awk script parses the output of ncdump -h to read information about
# the variables.
# The current version reads 2D and 3D variables from a CESM file.
#
# run with: ncdump -h FILE.nc4 | ./variable_information.awk > out.txt

BEGIN {
    OFS = "\t";
    print "VARIABLE", "INFO", "VALUE";
}

/float .*(time, lev, ncol)/ {
    variable = $2;
    sub(/\(.*/,"",variable); # remove dimension information
    print variable, "levels", 30;
}

/float .*(time, ilev, ncol)/ {
    variable = $2;
    sub(/\(.*/,"",variable); # remove dimension information
    print variable, "levels", 31;
}

/float .*(time, ncol)/ {
    variable = $2;
    sub(/\(.*/,"",variable); # remove dimension information
    print variable, "levels", 1;
}

/.*:long_name = / {
    var = $1;
    sub(/:long_name/,"",var); # remove ":long_name"
    if (var == variable) { # only print names when levels have been printed just before
        name = $3;
        for (i=4; i<=NF; i++) name = name " " $i;
        gsub(/"/,"",name); # remove quotes from variable name
        sub(/ ;/,"",name); # remove trailing ";" from variable name
        print variable, "name", name;
    }
}
