To-Do List
==========

Fixing problems in the current algorithm
----------------------------------------
* Make release build of minlin version behave like debug build.
* Add warning/error when a dimension with a small length is distributed.
  This prevents problems when an indexed dimension has not been specified
  by accident.


Problems with dependencies
--------------------------
* Wait for [minlin issue](https://github.com/bcumming/minlin/issues/11) to be
  fixed so statistics in minlin_device work too.


Small improvements to the algorithm
-----------------------------------
* Calculate a normalized L-norm (normalized by the sum of the original vector
  norms).
* Find a better/more solid way to estimate the maximum iterations for the
  Lanczos algorithm.
* Handle errors consistently (no simple asserts, error messages to std::err...).
* Distribute data more evenly amongst processes. Instead of adding the
  remainder to the last process, make some processes just have 1 more than the
  others. For horizontal stacking, the processes getting this additional data
  can vary using the variable process_getting_more_data_.


Expanding the algorithm
-----------------------
* Save compressed data to a file.
* Use lossless compression to further compress the resulting file.
* Use more than one eigenvector during the clustering.
* Add OpenMP for loops that can be easily parallelized. Some of them are marked
  as TODO in the code.
