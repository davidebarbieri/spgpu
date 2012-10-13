#!/bin/sh
rm Makefile
rm *.cmake
rm -R CMakeFiles
rm -R CMakeCache.txt
cmake ../../src
