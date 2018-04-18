
set(SRCS
  cfosls_integrators.cpp
  cfosls_testsuite.cpp  
  cfosls_tools.cpp
  )

set(HDRS
  cfosls_integrators.hpp 
  cfosls_testsuite.hpp  
  cfosls_tools.hpp 
  testhead.hpp
  )

convert_filenames_to_full_paths(SRCS)
convert_filenames_to_full_paths(HDRS)

set(SOURCES ${SOURCES} ${SRCS} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${HDRS} PARENT_SCOPE)
