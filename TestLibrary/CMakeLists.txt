#set(JAVA_HOME "/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/")

find_package(Java)
find_package(JNI)

find_package(MPI)

#set(JNI_LIBRARIES /usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/lib/libjawt.so;/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/lib/server/libjvm.so)

if(Java_FOUND)
    message("java version: ${Java_VERSION} was found. Java executable: ${Java_JAVA_EXECUTABLE}")
else()
    message("java not found!!")
endif()

if(JNI_FOUND)
    message("jni was found. Include dirs: ${JNI_INCLUDE_DIRS}) and lib dirs: ${JNI_LIBRARIES}")
else()
    message("jni not found!! JAVA_HOME: ${JAVA_HOME}")
endif()

add_library(TestLibrary test.cpp)
add_library(MPILibrary MPINatives.cpp)
add_library(RenderingLibrary ManageRendering.cpp)

target_include_directories(TestLibrary PUBLIC ${JNI_INCLUDE_DIRS})
target_include_directories(MPILibrary PUBLIC ${JNI_INCLUDE_DIRS})
target_include_directories(RenderingLibrary PUBLIC ${JNI_INCLUDE_DIRS})

#target_include_directories (TestLibrary PUBLIC ${JNI_INCLUDE_DIRS})

target_include_directories(TestLibrary PUBLIC
        ${MPI_C_INCLUDE_PATH}
        )

target_include_directories(MPILibrary PUBLIC
        ${MPI_C_INCLUDE_PATH}
        )

target_link_libraries(TestLibrary ${JNI_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(MPILibrary ${JNI_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(RenderingLibrary ${JNI_LIBRARIES})