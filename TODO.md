To-Do List
==========

Fixing problems in the current algorithm
----------------------------------------
* Make release build of minlin version behave like debug build.


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
* Do not attempt to compress clusters with less than M vectors. Instead, just
  save all the uncompressed vectors. The easiest way of implementing this
  would probably involve adding a vector to save the number of final
  eigenvectors in each cluster. Then, we can just save all vectors as
  eigenvectors and save [1 0 ... 0], [0 1 0 ... 0] etc. as the reduced
  form of the vectors. Alternatively, we can just add logic that when the
  number of eigenvectors for a cluster is smaller than M, the reconstruction
  is done by copying the eigenvectors instead of multiplying them with the
  reduced form.
* Distribute data more evenly amongst processes. Instead of adding the
  remainder to the last process, make some processes just have 1 more than the
  others. For horizontal stacking, the processes getting this additional data
  can vary using the variable process_getting_more_data_. This would involve
  changing the function calculate_distributed_dims_data_range() to calculate
  the start and count accordingly. Other parts should not be affected but it
  might still be wise to check whether different numbers of processes still
  produce the same result after this change.
* Add OpenMP for loops that can be easily parallelized. Some of them are marked
  as TODO in the code.


Expanding the algorithm
-----------------------
* Save compressed data to a file. For this, we basically need to dump the two
  objects NetCDFInterface and CompressedMatrix, i.e. save all their member
  variables. A simple implementation would involve all processes saving their
  data separately, possibly to a shared file. However, this would only allow
  reading the file using the same number of processes. A more complete
  implementation would thus write all the information as if there was just a
  single process getting all the data. In this case, reading the file would
  become more complex, as it involves the same splitting of the data that we
  already use for reading the original data. We would not have to save the
  start and count vectors, instead we rebuild them when loading the file,
  according to the number of processes used for reading. The mapping should
  not be affected.
* Use lossless compression to further compress the resulting file.
* Use more than one eigenvector during the clustering. This probably already
  works and would just involve changing the number passed to the Lanczos
  algorithm in do_iterative_clustering().
