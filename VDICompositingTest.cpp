#include <iostream>
#include "ManageRendering.hpp"
#include <mpi.h>
#include "MPINatives.hpp"
#include "VDIParams.hpp"
#include <thread>
#include <fstream>
#include <zconf.h>

int main() {
    std::cout << "Hello, World!" << std::endl;

    std::string dataset = "Kingsnake";

    bool isCluster = false;

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);

    std::cout << "Got MPI thread level: " << provided << std::endl;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    MPI_Comm nodeComm;
    MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                         MPI_INFO_NULL, &nodeComm );

    int node_rank;
    MPI_Comm_rank(nodeComm,&node_rank);

    if(!isCluster) {
        node_rank = 0;
    }

    JVMData jvmData = setupJVM(isCluster, "VDICompositingTest");

    registerNatives(jvmData);

    setPointerAddresses(jvmData, MPI_COMM_WORLD);

    setMPIParams(jvmData, rank, node_rank, num_processes);

    std::thread render(&doRender, jvmData);

    waitRendererConfigured(jvmData);

    //launch the compositing

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jmethodID compositeMethod = env->GetMethodID(jvmData.clazz, "compositeVDIs", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;II)V");

    if(compositeMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function compositeVDIs not found!";
        }
    } else {
        std::cout << "compositeVDIs function successfully found!";
    }

    std::string basePath = "/home/aryaman/TestingData/";

    std::string filePath = basePath + dataset + "_" + std::to_string(num_processes) + "_" + std::to_string(rank)
                           + "SubVDI4_ndc_";

    std::ifstream colorFile( filePath + "col", std::ios::in | std::ios::binary);
    std::ifstream depthFile( filePath + "depth", std::ios::in | std::ios::binary);

    if(!colorFile.is_open() || !depthFile.is_open()) {
        std::cerr<< "Could not open the file! " << std::endl;
        std::exit(-1);
    }


    char * colorBuffer = new char[colorSize];
    char * depthBuffer = new char[depthSize];

    colorFile.read(colorBuffer, colorSize);
    depthFile.read(depthBuffer, depthSize);

    jobject jcolorBuffer = env->NewDirectByteBuffer(colorBuffer, colorSize);
    jobject jdepthBuffer = env->NewDirectByteBuffer(depthBuffer, depthSize);


    int iterations = 5;

    env->CallVoidMethod(jvmData.obj, compositeMethod, jcolorBuffer, jdepthBuffer, rank, iterations);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }


    std::cout<<"Back after calling do Render" <<std::endl;

    sleep(60);
    std::cout<<"Calling stopRendering!" <<std::endl;
    stopRendering(jvmData);

    render.join();

    MPI_Finalize();

    return 0;
}
