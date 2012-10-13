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
add_global_sources(${CXX_SOURCES})

file( GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" )

# just sm13
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ell_dspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/hell_dspmv.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/daxpby.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/daxy.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/ddot.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dnrm2.cu")
list(APPEND SM13_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/core/kernels/dscal.cu")


list (REMOVE_ITEM CUDA_SOURCES ${SM13_SRCS}) # compile just for sm13
add_sm13_sources(${SM13_SRCS})

get_sub_tree(subtree ${CMAKE_CURRENT_SOURCE_DIR})
add_include_dirs(${subtree})
add_include_dirs(${CMAKE_CURRENT_SOURCE_DIR})



find_package(CUDA)

# include directories
include_directories (${GLOBAL_INCLUDES})
cuda_include_directories (${GLOBAL_INCLUDES})
# library directories
link_directories (${GLOBAL_LIBRARIES}) 

set(GENCODE -gencode=arch=compute_11,code=sm_11  -gencode=arch=compute_20,code=sm_20)

message(STATUS "13" ${CUDA_SM13_SOURCES})
message(STATUS "XX" ${CUDA_SOURCES})

cuda_compile(cuda_sm13_objs ${CUDA_SM13_SOURCES} OPTIONS -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_20,code=sm_20)
cuda_compile(cuda_objs ${CUDA_SOURCES} OPTIONS ${GENCODE})
cuda_add_library (spgpu STATIC ${cuda_sm13_objs} ${cuda_objs} ${GLOBAL_SOURCES} )

add_executable(hellperf_s ${CMAKE_CURRENT_SOURCE_DIR}/tests/hellPerf.cpp)
add_executable(hellperf_d ${CMAKE_CURRENT_SOURCE_DIR}/tests/hellPerf.cpp)
add_executable(ctest ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest.c)
set_target_properties(hellperf_d PROPERTIES COMPILE_FLAGS -DTEST_DOUBLE)

CUDA_ADD_CUBLAS_TO_TARGET(hellperf_s)
CUDA_ADD_CUBLAS_TO_TARGET(hellperf_d)
target_link_libraries(hellperf_s spgpu)
target_link_libraries(hellperf_d spgpu)
target_link_libraries(ctest spgpu)







