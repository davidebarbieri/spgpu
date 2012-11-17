#pragma once

/*
 * spGPU - Sparse matrices on GPU library.
 * 
 * Copyright (C) 2010 - 2012 
 *     Davide Barbieri - University of Rome Tor Vergata
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 3 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
 
// Used to avoid lower precision MAD
#if __CUDA_ARCH__ >= 200
#define PREC_FADD(a,b) ((a) + (b))
#define PREC_FMUL(a,b) ((a) * (b))
#define PREC_DADD(a,b) ((a) + (b))
#define PREC_DMUL(a,b) ((a) * (b))
#else
#define PREC_FADD(a,b) __fadd_rn((a),(b))
#define PREC_FMUL(a,b) __fmul_rn((a),(b))
#define PREC_DADD(a,b) __dadd_rn((a),(b))
#define PREC_DMUL(a,b) __dmul_rn((a),(b))
#endif


#define ENABLE_CACHE