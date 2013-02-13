SET (SIFT_LIB_SOURCES
        imgfeatures
        utils
        sift
        kdtree
        minpq
        xform
)

SET (SIFT_FEAT_SOURCES siftfeat)

SET (MATCH_SOURCES match)

SET (DSP_FEAT_SOURCES dspfeat)


add_library(sift ${SIFT_LIB_SOURCES})


add_executable(siftfeat ${SIFT_FEAT_SOURCES})
target_link_libraries(siftfeat sift ${OpenCV_LIBS})
  
add_executable(match ${MATCH_SOURCES})
target_link_libraries(match sift ${OpenCV_LIBS})

add_executable(dspfeat ${DSP_FEAT_SOURCES})
target_link_libraries(dspfeat sift ${OpenCV_LIBS})