Payload
=======

Repository for the payload software.

Check out match.c for an example of how to use the RANSAC function.  Try 
`match beaver.png beaver_xform.png` to see it work.

Documentation is included in the docs/ directory.  If it is not there, 
use `make doc` to build it (you need Doxygen).

Help is available for executables using the '-h' command line option.


Requirements
============

Requres CMake:<BR>
http://www.cmake.org/

All code in this package requires the OpenCV library (known working 
version is 2.3):
http://sourceforge.net/projects/opencvlibrary/

Building
========

<b>Compile & Run Sift:</b>
<pre>
mkdir build
cd build
cmake ..
make
../bin/siftfeat
</pre>

<b>Compile & Run Tests:</b>
<pre>
mkdir build
cd build
cmake ..
make
../bin/runUnitTests
</pre>

<b>Compile Documentation:</b>
<pre>
mkdir build
cd build
cmake ..
make doc
</pre>

This should produce a few executables in bin/, a static library 
lib/libsift.a, and some HTML documentation in docs/.  You can use the -h 
argument to get help with any of the executables.  libsift.a can be 
compiled into your own code using the standard method:

	gcc -I/path/to/sift/include/ -L/path/to/sift/lib/ yourcode.c -o yourexecutable -lsift

The documentation in docs/ describes all of the functions available in 
libopensift.a as well as #defines, etc.  Use the documentation to determine 
what header files from include/ to include in your code.

License
=======

See the file LICENSE for more information on the legal terms of the use 
of this package.
