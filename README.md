# Introduction

spGPU is a set of custom matrix storages and CUDA kernels for sparse linear algebra computing on GPU. It isn't a replacement for cuBLAS/cuSPARSE that should be used for a full featured linear algebra environment on GPU.

The main matrix storage used by spGPU is a GPU-friendly ELLpack format, as well as our HELL (Hacked ELLpack) and our HDIA formats, two enhanced versions of ELLpack and DIA with some interesting memory saving properties.

HELL format provides a better memory storage compared to ELL (it avoids allocation inefficency provided by spikes in row sizes), while providing just quite the same or superior performances for the sparse matrix-vector multiply routine. HDIA format applies the same design choices to DIA.

Documentation: http://rawgit.com/davidebarbieri/spgpu/master/doc/html/index.html

Developer:
* Davide Barbieri - University of Rome Tor Vergata

Advisors:
* Valeria Cardellini - University of Rome Tor Vergata
* Salvatore Filippone - University of Rome Tor Vergata

# How to build spgpu
## Linux (and other unix systems)
Requirements: CUDA 6.5, cmake 2.8.4

```
cd spgpu/build/cmake
sh configure.sh
make
```
