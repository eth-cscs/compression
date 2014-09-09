Notes/Documentation for pyStats.py
==================================

This file contains some notes from going through the code of pyStats.py. It
was mainly created as a tool for working with the code so it is not very
polished and some things might not be clear to other people.


General Information
-------------------

* files read:
  * summary files: original ensemble (ens_file argument of main()), reconstructed ensemble (recon_ens_file argument of main())
    * contain variables with information about runs
    * variable names (vars), 3d variables (var3d)
    * ens_avg_3d, ens_avg_2d: ensemble average for 3d and 2d variables (one value per variable and point)
    * ens_stddev3d, ens_stddev2d: ensemble standard deviation for 3d and 2d variables (one value per variable and point)
    * RMSZ: ensemble rmsz values for all variables (one value per variable and ensemble member)
  * 101 run files found in inputdir
    * have variables with values from runs
* where do z-values come from?
  * there is a RMSZ field in the ensemble summaries, one for each variable and ensemble member, used for rmszrange and bias
  * there is a function calculate_raw_score, used for rmszens
* functionality: all options use orig/recon files (single file/run) with time tstart to tend and run for all variables
  * min finds minimal value of original
  * max finds maximum value of original
  * mean finds average value of original
  * std finds standard deviation of original
  * mnorm compares reconstructed and original file, calculating a normalized maximum difference between both
  * rmse compares reconstructed and original file, calculating a normalized rms error between both
  * corr compares reconstructed and original file, calculating the pearson correlation between both
  * rmszens uses average and std from original ensemble summary file to calculate the (single) rmsz score for each variable, either for the original data (if label=="orig") or for the reconstructed data
  * rmszrange uses ens_rmsz from the original ensemble summary file to find the maximum & minimum rmsz scores
  * bias uses original and reconstructed ensemble summary files, fitting a linear function to ens_rmsz and recon_ens_rmsz. it calculates both the slope and intercept with error intervals.-
  * normens uses inputdir (101 run files) to calculate the normalized maximum difference between any two runs. for each runs, it prints out/writes the maximum difference to any of the other runs. it also prints out the smallest and largest of these max. difference. it is similar to mnorm, but instead of comparing the original and reconstructed file, it compares all files (either original or reconstructed ensemble) with each other.


Functions
---------

* notSupportedQuality(): warning/error that quality metric is not supported yet
* notSupportedGenerate(): warning/error that generation of reconstructed data files is not supported yet
* calc_Z(val, arg, stddev, count, flag): calculate (val-arg)/stddev and exclude zero values
  * val, arg, stddev: used for calculating standardized value
  * count: is incremented with number of stddevs below tolerance 1e-12
  * flag: if set to True, warnings for stddevs below tolerance are printed (once)
  * if only some stddevs are below tolerance, the function calculates the z-value of the others and sets those below tolerance to zero
* get_ncol_nlev(frun): goes through the dimensions of “frun” and reads the length of the dimensions “lev” and “ncol”. if either of those isn’t found, it exits. otherwise, it returns (nlev*ncol, ncol), i.e. the number of 3d points and the number of 2d points.
* myLinearFunc(x, a, b): returns a*x+b, used bf curve_fit
* read_ensemble_summary(ens_file):
  * opens file ens_file (string with file path)
  * creates 3 dictionaries: ens_avg, ens_stddev, ens_rmsz
  * add all variable names (saved in variable “vars”) to ens_var_name list
  * set num_var3d to number of 3d variables, given by length of variable “var3d”
  * it seems like the 3d variables appear first in the ens_var_name list
  * save ens_avg3d and ens_avg2d to the dictionary ens_avg, using the variable names from ens_var_name as keys
  * do the same for ens_stddev3d and ens_stddev2d
  * save RMSZ variable to ens_rmsz
* calculate_raw_score(k, v, npts3d, npts2d, ens_avg, ens_stddev): calculate rmsz score by comparing the run file with the ensemble summary file
  * k: key (variable name) that should appear in ens_avg, otherwise has_zscore is set to False
  * v: values that will be normalized with ensemble average and stddev
  * npts3d, npts2d: number of points for 3d/2d variables (see get_ncol_nlev)
  * Zscore is calculated as zero if all elements have stddev below tolerance, otherwise it is the root mean square of all z values of the elements with a stddev above tolerance
* calculate_maxnormens(inputdir, mkeysTimeSeries): calculate max norm ensemble value for each variable based on 101 run files. the inputdir should only have 101 run files
  * inputdir: directory with 101 ensemble run files
  * mkeysTimeSeries: all variable names with time series, read from json file
  * returns nothing, results printed/written to file
  * build a list ifiles with the Nio file objects from all files in inputdir
  * go through dictionary mkeysTimeSeries
    * build a list “output” with all the variables[key][1] from all the files
* probably all ensemble values for the variable, the 1 is probably selecting the second timestep
    * create a file KEY_ens_maxnorm_new.txt
    * go through files, building list Maxnormens[key], this value is the maximum difference between two values from different runs
    * normalize this difference by the total range of the variable, setting it to zero if the total range is very small
* calculate_bias(ens_file, recon_ens_file, label, space, delim, outfile)
  * ens_file, recon_ens_file: file paths, used for read_ensemble_summary
  * label: used for string written to outfile
  * space, delim: used for space & delimiter in output file
  * outfile: output file name/path
  * returns nothing, results printed/written to file
  * go through keys/variable names
    * check for non-zero ens_rmsz values
    * fit linear curve to data pairs (ens_rmsz[variable], recon_ens_rmsz[variable]), finding slope & intercept
    * calculate 95% error intervals on slope & intercept
    * write everything to file
* main(argv)
  * analyze command line arguments
  * initialize dicts startTime & elapseTime and slice for saving timing information
  * read mkeysTimeInvarient, mkeysTimeVarient, mkeysTimeSeries from json file
  * if option is set, calculate maximum ensemble norm from mkeysTimeSeries
  * if option is set, calculate bias between original and reconstructed ensemble (if normens not set, why??)
  * calculate single file metrics: only done if one of the following options is set: min, max, mean, std, corr, rmse, mnorm, rmszrange, rmszens
    * set otimeSeries, rtimeSeries to variable dict; odimensions, rdimensions to dimension dict (only if recon is set)
    * go through otimeSeries dict (iterate), skipping variables that don’t have a time-dimension, are in mkeysTimeInvarient or in mkeysTimeVarient, or have only one dimension
* do this for all time steps that are within the time range
* for all of these, the results are printed and optionally written to the output file. if the option isn’t set, they are skipped
* read/calculate metrics from original data: minimum, maximum, mean, stddev
* mnorm: maximum difference between original and reconstructed, divided by total span (max-min) of original values
* rmse: root mean square error, divided by total span (max-min) of original values
* corr: calculate correlation & p-value between 2 sets, return only correlation. all values are divided by “overflow” if the original stddev is larger than “overflow”.
* rmszens: calculate zscore with calculate_raw_score, using original file if label==orig, reconstructed file otherwise
* rmszrange: read max and min from ens_rmsz[variable]
