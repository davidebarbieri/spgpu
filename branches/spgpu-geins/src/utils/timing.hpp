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
 

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif



#include <time.h>

/// @cond

class Clock
{
private:
#ifdef _WIN32
	unsigned long long frequency_;
	unsigned long long initialValue_;
#else
	struct timeval initialValue_;
#endif
public:
	inline Clock(void)
	{
#ifdef _WIN32
		LARGE_INTEGER f, c;

		QueryPerformanceFrequency(&f) ;
		frequency_ = f.QuadPart;

		QueryPerformanceCounter(&c);
		initialValue_ = c.QuadPart;
#else
		
	 	gettimeofday(&initialValue_, NULL);
#endif
		
	}


	// in seconds
	inline float getTime()
	{
#ifdef _WIN32
		LARGE_INTEGER time;
		QueryPerformanceCounter(&time);
		unsigned long long current = time.QuadPart - initialValue_;

		return (float)current/(float)frequency_;
#else
	 	struct timeval time;
	 	gettimeofday(&time, NULL);
	 	time.tv_sec -= initialValue_.tv_sec;
	 	time.tv_usec -= initialValue_.tv_usec;
	 	
		float tf = time.tv_sec + (time.tv_usec)*0.000001f;
		return tf;
#endif
	}

};
/// @endcond
