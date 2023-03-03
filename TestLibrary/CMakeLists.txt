#set(JAVA_HOME "/usr/lib/jvm/adoptopenjdk-11-hotspot-amd64/")

find_package(Java)
find_package(JNI)

find_package(MPI)

#set(IceT_DIR $ENV{ICET_PATH})
set(IceT_DIR "/home/aryaman/Repositories/icet/build/lib/")

FIND_PACKAGE(IceT REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)

message("IceT path: ${IceT_DIR}")
message("IceT lib: ${ICET_CORE_LIBS}")

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
target_include_directories(MPILibrary PUBLIC ${ICET_INCLUDE_DIRS})
target_include_directories(RenderingLibrary PUBLIC ${JNI_INCLUDE_DIRS})
message("changed icet dirs")
target_include_directories(RenderingLibrary PUBLIC ${ICET_INCLUDE_DIRS})

#target_include_directories (TestLibrary PUBLIC ${JNI_INCLUDE_DIRS})

target_include_directories(TestLibrary PUBLIC
        ${MPI_C_INCLUDE_PATH}
        )

target_include_directories(MPILibrary PUBLIC
        ${MPI_C_INCLUDE_PATH}
        )

target_include_directories(RenderingLibrary PUBLIC
        ${MPI_C_INCLUDE_PATH}
        )

target_link_libraries(TestLibrary ${JNI_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(MPILibrary ${JNI_LIBRARIES} ${MPI_CXX_LIBRARIES})
target_link_libraries(RenderingLibrary ${JNI_LIBRARIES})
target_link_libraries(RenderingLibrary ${ICET_CORE_LIBS} ${ICET_GL_LIBS} ${ICET_MPI_LIBS})