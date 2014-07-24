#pragma once

#include <omp.h>

#include <mkl.h>

#include <mkl.h>
#include <minlin/minlin.h>
#include <minlin/modules/threx/threx.h>
using namespace minlin::threx; // just dump the namespace for this example

template <typename T>
void matlab_vector(const DeviceVector<T> &v, const char *name ) {
    std::cout << name << " = [";
    for(int i=0; i<v.size(); i++)
        std::cout << v(i) << " ";
    std::cout << "]';" << std::endl;
}