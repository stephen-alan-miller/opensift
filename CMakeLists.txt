cmake_minimum_required (VERSION 2.6)
project (OO_PAYLOAD)
# The version number.
set (OO_PAYLOAD_VERSION_MAJOR 0)
set (OO_PAYLOAD_VERSION_MINOR 1)

# Set target directories
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)

# Set directories
include_directories(${OO_PAYLOAD_SOURCE_DIR}/include)

# Set platform
SET (PLATFORM 64)

#This may need to be updated or Linux/Windows
SET (CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

find_package(OpenCV REQUIRED)
find_package(Doxygen)

#include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
#link_directories (${OPENGL_LIBRARY_DIRS} ${GLUT_LIBRARY_DIRS})

include_directories("${OO_PAYLOAD_SOURCE_DIR}/sift")

add_subdirectory(src)

if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)

add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)