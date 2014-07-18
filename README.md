USI Compression
===============

This repository contains a Matlab and C++ implementation of the compression
algorithm developped at USI by Prof. Illia Horenko.


Requirements
------------
The following setup is necessary to compile and run the algorithm. This list is
not complete yet.

* The minlin library is included as a Git submodule, so it has to be initialized
  and updated if the repository is cloned.
* We use the Thrust from cudatoolkit, so the corresponding module has to be
  activated.
