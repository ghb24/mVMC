# include guard
cmake_minimum_required(VERSION 2.8.0 )

if(${CMAKE_PROJECT_NAME} STREQUAL "Project")
  message(FATAL_ERROR "cmake should be executed not for 'src' subdirectory, but for the top directory of GPW.")
endif(${CMAKE_PROJECT_NAME} STREQUAL "Project")

include_directories(include)
include_directories(../mVMC/include)

set(SOURCES_GPW_KERN gpw_kernel.c)
set(SOURCES_GPW_TRN gpw_training.c)

add_library(gpw_kernel SHARED ${SOURCES_GPW_KERN})

target_link_libraries(gpw_kernel ${LAPACK_LIBRARIES} m)

if(USE_SCALAPACK)
  string(REGEX REPLACE "-L[ ]+" "-L" sc_libs ${SCALAPACK_LIBRARIES})
  string(REGEX REPLACE "[ ]+" ";" sc_libs ${sc_libs})
  foreach(sc_lib IN LISTS sc_libs)
    target_link_libraries(gpw_kernel ${sc_lib})
  endforeach(sc_lib)
  message(STATUS "SCALAPACK_LIBRARIES: ${SCALAPACK_LIBRARIES}")
endif(USE_SCALAPACK)

if(MPI_FOUND)
  target_link_libraries(gpw_kernel ${MPI_C_LIBRARIES})
endif(MPI_FOUND)

install(TARGETS gpw_kernel
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
