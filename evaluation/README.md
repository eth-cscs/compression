Evaluation of Compression Algorithm (Summer 2014)
=================================================

This folder contains some files from the evaluation of the USI compression
algorithm carried out by Manuel Schmid in summer 2014.

The following files can be found in this folder:

* Presentation_Internship_CSCS.pdf: The slides for the final presentation of
  the internship of Manuel Schmid in summer 2014.
* overall_statistics_*.txt, variable_statistics_*.txt: The parsed output of
  the runs done for evaluating the algorithm.
  * klarge: K = [10 20 ... 200], M = [10 20 ... 200], horizontal/vertical
    stacking, all variables. Warning: compression_ratio is wrong!
  * ksmall: K = [1 2 ... 10], M = [10 20 ... 200], vertical stacking, all
    variables. Warning: compression_ratio is wrong!
  * alldistributed_3d: K = [1 2 ... 10], M = [10 20 ... 200], vertical
    stacking, 3D variables, ncol*lev as distributed dimensions. Warning:
    comperssion_ratio is wrong!
  * 3d: K = [6 7 ... 15], M = [160 170 ... 250], vertical stacking, 3D
    variables.
  * ensemble: K = 8, M = 200, vertical stacking, all variables, 101 ensemble
    files.
* rmsz_*_4var.txt: RMSZ values for the variables U, Z3, FSDSC, and CCN3 (like
  in the paper by Baker et al.) for the K=8, M=200 ensemble run.
* *.ipynb: IPython Notebook files for the analysis/statistics of the *.txt
  files in this folder. The "USI Compression Data Analysis" file is a general
  file used for an initial look at the variables (and is therefore a bit messy)
  whereas the other files have statistics for a specific analysis question.
* [pyStats_documentation.md](pyStats_documentation.md): Notes on the pyStats.py script.
