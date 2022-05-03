cmake_minimum_required (VERSION 2.8.4)

project(spgpu)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

# Useful macro to add sources
macro (add_global_sources)
	foreach (_src ${ARGN})
        list (APPEND GLOBAL_SOURCES "${_src}")
    endforeach()
endmacro()

macro (add_sm13_sources)
	foreach (_src ${ARGN})
        list (APPEND CUDA_SM13_SOURCES "${_src}")
    endforeach()
endmacro()

macro (add_sm20_sources)
	foreach (_src ${ARGN})
        list (APPEND CUDA_SM20_SOURCES "${_src}")
    endforeach()
endmacro()

macro (add_include_dirs)
	foreach (_inc ${ARGN})
        list (APPEND GLOBAL_INCLUDES "${_inc}")
    endforeach()
endmacro()

macro (add_library_dir)
	foreach (_lib ${ARGN})
        list (APPEND GLOBAL_LIBRARIES "${_lib}")
    endforeach()
endmacro()

macro(select_directories dirs files)
	foreach(f ${${files}}) 
		if(IS_DIRECTORY "${f}")
			# avoid hidden directories (like .svn)
			get_filename_component(_justfile ${f} NAME)
			if (NOT ${_justfile} MATCHES "^[.]")
				list (APPEND ${dirs} "${f}")
			endif()
		endif()
	endforeach(f) 
endmacro()

function(get_sub_tree subtree dir)
	set(_treelist)
	set(_tree)
	set(_subtree)
	
	file( GLOB _treelist "${dir}/*")
	select_directories(_tree _treelist)
		
	foreach (subdir ${_tree})
		get_sub_tree(_curr ${subdir})
		list (APPEND _subtree ${_curr})
	endforeach()
	
	list (APPEND _subtree "${_tree}")
	
	set(${subtree} ${_subtree} PARENT_SCOPE)
endfunction()

########

# adding subtree

file( GLOB_RECURSE CXX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" )

# exclude tests from library creation
file( GLOB_RECURSE TESTS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp" )
list (REMOVE_ITEM CXX_SOURCES ${TESTS_SRCS})

add_global_sources(${CXX_SOURCES})

file( GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" )

# just sm13
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ell_dspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ell_zspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ell_dcsput.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ell_zcsput.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/hell_dspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/hell_zspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dia_dspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dia_zspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/hdia_dspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/hdia_zspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/daxpby.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zaxpby.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/daxy.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zaxy.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ddot.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zdot.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dnrm2.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/znrm2.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dscal.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zscal.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dscat.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zscat.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dgath.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zgath.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/damax.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zamax.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dasum.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zasum.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dabs.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zabs.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dsetscal.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/zsetscal.cu")


list (REMOVE_ITEM CUDA_SOURCES ${SM13_SRCS}) # compile just for sm13
add_sm13_sources(${SM13_SRCS})

get_sub_tree(subtree ${CMAKE_CURRENT_SOURCE_DIR})
add_include_dirs(${subtree})
add_include_dirs(${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA)

# Set CUDA_cusparse_LIBRARY if not defined
SET(CUDA_lib_directory "${CUDA_TOOLKIT_ROOT_DIR}/lib/")
if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
	SET(CUDA_lib_directory "${CUDA_TOOLKIT_ROOT_DIR}/lib64/")
endif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
	
IF(NOT DEFINED CUDA_cusparse_LIBRARY)
	SET(CUDA_cusparse_LIBRARY "${CUDA_lib_directory}libcusparse.so")
ENDIF(NOT DEFINED CUDA_cusparse_LIBRARY)

# include directories
include_directories (${GLOBAL_INCLUDES})
cuda_include_directories (${GLOBAL_INCLUDES})
# library directories
link_directories (${GLOBAL_LIBRARIES}) 
IF (CUDA_VERSION_MAJOR LESS 7)
set(GENCODE -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=sm_50 )
set(GENCODE_DOUBLE -gencode=arch=compute_13,code=sm_13
-gencode=arch=compute_20,code=sm_20
-gencode=arch=compute_30,code=sm_30
-gencode=arch=compute_50,code=sm_50)
ELSEIF (CUDA_VERSION_MAJOR LESS 11)
set(GENCODE -gencode=arch=compute_30,code=sm_30
-gencode=arch=compute_35,code=sm_35
-gencode=arch=compute_50,code=sm_50 ) 
set(GENCODE_DOUBLE -gencode=arch=compute_30,code=sm_30
-gencode=arch=compute_35,code=sm_35
-gencode=arch=compute_50,code=sm_50)
ELSE (CUDA_VERSION_MAJOR LESS 11)
set(GENCODE  -gencode=arch=compute_50,code=sm_50
-gencode=arch=compute_75,code=sm_75 ) 
set(GENCODE_DOUBLE -gencode=arch=compute_50,code=sm_50
-gencode=arch=compute_75,code=sm_75 ) 
ENDIF (CUDA_VERSION_MAJOR LESS 11)

#set(ADDITIONAL_CUDA_OPTIONS -DDEBUG -G)
#set(ADDITIONAL_OPTIONS -DDEBUG -G -DASSUME_LOCK_SYNC_PARALLELISM)
#set(ADDITIONAL_OPTIONS -DASSUME_LOCK_SYNC_PARALLELISM)

cuda_compile(cuda_sm13_objs ${CUDA_SM13_SOURCES} OPTIONS ${GENCODE_DOUBLE} ${ADDITIONAL_CUDA_OPTIONS} --ptxas-options=-v)
cuda_compile(cuda_objs ${CUDA_SOURCES} OPTIONS ${GENCODE} ${ADDITIONAL_CUDA_OPTIONS} --ptxas-options=-v)
cuda_add_library (spgpu STATIC ${cuda_sm13_objs} ${cuda_objs} ${GLOBAL_SOURCES} )

add_executable(hellperf_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/hellPerf.cpp)
add_executable(hellperf_d ${CMAKE_CURRENT_SOURCE_DIR}/tests/hellPerf.cpp)
add_executable(hellperf_norowsize_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/hellPerf.cpp)
add_executable(hellperf_norowsize_d ${CMAKE_CURRENT_SOURCE_DIR}/tests/hellPerf.cpp)
add_executable(diaPerf_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/diaPerf.cpp)
add_executable(diaPerf_d ${CMAKE_CURRENT_SOURCE_DIR}/tests/diaPerf.cpp)
add_executable(ctest ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest.c)
add_executable(testSpVec_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/testSparseVector.c)
add_executable(testSpVec_d ${CMAKE_CURRENT_SOURCE_DIR}/tests/testSparseVector.c)
add_executable(testDeVec_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/testDenseVector.c)
#add_executable(cusparsePerf_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/cusparsePerf.cpp)
#add_executable(cusparsePerf_d ${CMAKE_CURRENT_SOURCE_DIR}/tests/cusparsePerf.cpp)
set_target_properties(hellperf_norowsize_s PROPERTIES COMPILE_FLAGS -DNO_ROW_SIZE)
set_target_properties(hellperf_d PROPERTIES COMPILE_FLAGS -DTEST_DOUBLE)
set_target_properties(hellperf_norowsize_d PROPERTIES COMPILE_FLAGS "-DNO_ROW_SIZE -DTEST_DOUBLE")
set_target_properties(diaPerf_d PROPERTIES COMPILE_FLAGS -DTEST_DOUBLE)
set_target_properties(testSpVec_d PROPERTIES COMPILE_FLAGS -DTEST_DOUBLE)
#set_target_properties(cusparsePerf_d PROPERTIES COMPILE_FLAGS -DTEST_DOUBLE)

CUDA_ADD_CUBLAS_TO_TARGET(hellperf_s)
CUDA_ADD_CUBLAS_TO_TARGET(hellperf_d)
CUDA_ADD_CUBLAS_TO_TARGET(hellperf_norowsize_s)
CUDA_ADD_CUBLAS_TO_TARGET(hellperf_norowsize_d)
CUDA_ADD_CUBLAS_TO_TARGET(testDeVec_s)
#CUDA_ADD_CUBLAS_TO_TARGET(cusparsePerf_s)
#CUDA_ADD_CUBLAS_TO_TARGET(cusparsePerf_d)
target_link_libraries(hellperf_s spgpu)
target_link_libraries(hellperf_d spgpu)
target_link_libraries(hellperf_norowsize_s spgpu)
target_link_libraries(hellperf_norowsize_d spgpu)
target_link_libraries(diaPerf_s spgpu)
target_link_libraries(diaPerf_d spgpu)
target_link_libraries(ctest spgpu)
target_link_libraries(testSpVec_s spgpu)
target_link_libraries(testSpVec_d spgpu)
target_link_libraries(testDeVec_s spgpu)
#target_link_libraries(cusparsePerf_s spgpu)
#target_link_libraries(cusparsePerf_s ${CUDA_cusparse_LIBRARY})
#target_link_libraries(cusparsePerf_d spgpu)
#target_link_libraries(cusparsePerf_d ${CUDA_cusparse_LIBRARY})






