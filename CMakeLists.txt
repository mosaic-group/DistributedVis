cmake_minimum_required(VERSION 3.19)
project(DistributedVis)

#set(JAVA_HOME "/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/")

find_package(Java)
find_package(JNI)

#set(JNI_LIBRARIES /usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/lib/libjawt.so;/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/lib/server/libjvm.so)

find_package(MPI REQUIRED)

if(MPI_FOUND)
    message( "MPI has been found! Library: ${MPI_CXX_LIBRARIES}" )
else()
    message( "MPI not found!" )
endif()

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(TestLibrary)

add_executable(DistributedVis main.cpp TestLibrary/test.cpp)

target_include_directories(DistributedVis PUBLIC
        "${CMAKE_SOURCE_DIR}/TestLibrary"
        ${MPI_C_INCLUDE_PATH}
        )

target_include_directories(DistributedVis PUBLIC ${JNI_INCLUDE_DIRS})


#target_compile_options(DistributedVis PRIVATE ${MPI_CXX_COMPILE_FLAGS})
#target_link_libraries(DistributedVis ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})

target_link_libraries(DistributedVis PUBLIC ${JNI_LIBRARIES})
target_link_libraries(DistributedVis PUBLIC pthread)
target_link_directories(DistributedVis PUBLIC TestLibrary)
target_link_libraries(DistributedVis PUBLIC ${MPI_CXX_LIBRARIES})