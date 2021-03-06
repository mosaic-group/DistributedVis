#include <iostream>
#include <dirent.h>
#include <fstream>
#include "TestLiibrary.hpp"
#include <mpi.h>
#include <ctime>
#include <chrono>
#include <string.h>

#define VERBOSE false
#define USE_VULKAN true
#define SEPARATE_DEPTH true
#define SAVE_FILES false

int count = 0;

int windowWidth = 1920;
int windowHeight = 1080;
int numSupersegments = 20;
int numOutputSupsegs = 20;

std::string datasetName = "";

auto begin_copy1 = std::chrono::high_resolution_clock::now();
auto end_copy1 = std::chrono::high_resolution_clock::now();

auto begin_copy2 = std::chrono::high_resolution_clock::now();
auto end_copy2 = std::chrono::high_resolution_clock::now();

auto begin_whole = std::chrono::high_resolution_clock::now();
auto begin = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();

auto begin1 = std::chrono::high_resolution_clock::now();
auto end1 = std::chrono::high_resolution_clock::now();

auto begin2 = std::chrono::high_resolution_clock::now();
auto end2 = std::chrono::high_resolution_clock::now();

auto begin3 = std::chrono::high_resolution_clock::now();
auto end3 = std::chrono::high_resolution_clock::now();

double total_alltoall = 0;
double total_gather = 0;
double total_overall = 0;
double total_whole = 0;
long int num_alltoall = 0;
long int num_gather = 0;

int warm_up_iterations = 20;

void * subColor_copy;
void * subDepth_copy;

void * compositedColor_copy;
void * compositedDepth_copy;

auto begin4 = std::chrono::high_resolution_clock::now();
auto end4 = std::chrono::high_resolution_clock::now();

//int totalSupersegments = windowWidth * windowHeight * numSupersegments *

//void * a = nullptr;
//void * b = nullptr;
//void * c = nullptr;
//void * d = nullptr;

enum vis_type
{
    particles = 0x0,
    grid = 0x1,
};

bool isMain = false;

vis_type getVisType() {
    return vis_type::grid;
}

void registerNatives(JVMData jvmData);

JVMData setupJVM(bool isCluster) {
    JVMData jvmData{};

    DIR *dir;
    struct dirent *ent;
    std::string classPath = "-Djava.class.path=";
    std::string directory = "/home/aryaman/Repositories/scenery-insitu/build/libs/";

    if ((dir = opendir (directory.c_str())) != nullptr) {
        while ((ent = readdir (dir)) != nullptr) {
            classPath.append(directory);
            classPath.append(ent->d_name);
            classPath.append(":");
        }
        closedir (dir);
    } else {
        /* could not open directory */
        std::cerr << "ERROR: Could not open directory "<< directory <<std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string className;

    if(isMain) {
        className = "graphics/scenery/insitu/InSituMaster";
    } else {
        // determine whether particle or grid data is to be rendered
        vis_type visType = getVisType();

        std::cout<<"Got the data type "<<std::endl;

        if(visType == vis_type::grid) className = "graphics/scenery/insitu/DistributedVolumes";
        else className = "graphics/scenery/insitu/InVisRenderer";
    }

    JavaVMInitArgs vm_args;                        // Initialization arguments
    auto *options = new JavaVMOption[7];   // JVM invocation options
    options[0].optionString = (char *)classPath.c_str();

    #if USE_VULKAN
    options[1].optionString = (char *)
            "-Dscenery.VulkanRenderer.EnableValidations=false";
    #else
    options[1].optionString = (char *)
            "-Dscenery.Renderer=OpenGLRenderer";
    #endif


    options[2].optionString = (char *)
            "-Dorg.lwjgl.system.stackSize=1000";

    options[3].optionString = (char *)
            "-Dscenery.Headless=true";

    options[4].optionString = (char *)
                                      "-Dscenery.LogLevel=error";

//    options[4].optionString = (char *)
//                                      "-Dscenery.LogLevel=debug";

    if(isCluster) {
        options[5].optionString = (char *)
                "-Dorg.lwjgl.system.SharedLibraryExtractPath=/scratch/ws/1/argupta-distributed_vdis/";
        options[6].optionString = (char *)
                "-Dorg.lwjgl.librarypath=/scratch/ws/1/argupta-distributed_vdis/";
    } else {
        options[5].optionString = (char *)
                "-Dorg.lwjgl.system.SharedLibraryExtractPath=/tmp/";
        options[6].optionString = (char *)
                "-Dorg.lwjgl.librarypath=/tmp/";
    }


    vm_args.version = JNI_VERSION_1_6;
    vm_args.nOptions = 4;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = false;

    jint rc = JNI_CreateJavaVM(&jvmData.jvm, (void **) &jvmData.env, &vm_args);

    #if VERBOSE
    std::cout<<"Hello world"<<std::endl;
    #endif

    delete[] options;

    if (rc != JNI_OK) {
        // TODO: error processing...
        std::cin.get();
        std::exit(EXIT_FAILURE);
    }

    std::cout << "JVM load succeeded: Version " << std::endl;
    jint ver = jvmData.env->GetVersion();
    std::cout << ((ver >> 16) & 0x0f) << "." << (ver & 0x0f) << std::endl;


    jclass localClass;
    localClass = jvmData.env->FindClass(className.c_str());  // try to find the class
    if(localClass == nullptr) {
        if( jvmData.env->ExceptionOccurred() ) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cerr << "ERROR: class "<< className << " not found !" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // if class found, continue
    #if VERBOSE
    std::cout << "Class found " << className << std::endl;
    #endif

    jmethodID constructor = jvmData.env->GetMethodID(localClass, "<init>", "()V");  // find constructor
    if (constructor == nullptr) {
        if( jvmData.env->ExceptionOccurred() ) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cerr << "ERROR: constructor not found !" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

#if VERBOSE
    std::cout << "Constructor found for " << className << std::endl;
#endif

    //if constructor found, continue
    jobject localObj;
    localObj = jvmData.env->NewObject(localClass, constructor);
    if(jvmData.env->ExceptionOccurred()) {
        jvmData.env->ExceptionDescribe();
        jvmData.env->ExceptionClear();
    }
    if (!localObj) {
        if (jvmData.env->ExceptionOccurred()) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: class object is null !";
            std::exit(EXIT_FAILURE);
        }
    }

    #if VERBOSE
    std::cout << "Object of class has been constructed" << std::endl;
    #endif

    jfieldID clusterField = jvmData.env->GetFieldID(localClass, "isCluster", "Z");
    jvmData.env->SetBooleanField(localObj, clusterField, isCluster);

    if (jvmData.env->ExceptionOccurred()) {
        jvmData.env->ExceptionDescribe();
    }

    jvmData.clazz = reinterpret_cast<jclass>(jvmData.env->NewGlobalRef(localClass));
    jvmData.obj = reinterpret_cast<jobject>(jvmData.env->NewGlobalRef(localObj));

    jvmData.env->DeleteLocalRef(localClass);
    jvmData.env->DeleteLocalRef(localObj);

    #if VERBOSE
    std::cout << "Global references have been created and local ones deleted" << std::endl;
    #endif

    registerNatives(jvmData);

    return jvmData;
}

void distributeVDIs(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer);
void gatherCompositedVDIs(JNIEnv *e, jobject clazzObject, jobject compositedVDIColor, jobject compositedVDIDepth, jint compositedVDILen, jint root, jint myRank, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer);

void registerNatives(JVMData jvmData) {
    JNINativeMethod methods[] { { (char *)"distributeVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIJJJ)V", (void *)&distributeVDIs },
                                { (char *)"gatherCompositedVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIJJJ)V", (void *)&gatherCompositedVDIs }
    };

    int ret = jvmData.env->RegisterNatives(jvmData.clazz, methods, 2);
    if(ret < 0) {
        if( jvmData.env->ExceptionOccurred() ) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cerr << "ERROR: Could not register natives!" <<std::endl;
            //std::exit(EXIT_FAILURE);
        }
    } else {
        if (VERBOSE) std::cout<<"Natives registered. The return value is: "<< ret <<std::endl;
    }

    begin_whole = std::chrono::high_resolution_clock::now();
}

void stopRendering(JVMData jvmData) {

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jmethodID terminateMethod = env->GetMethodID(jvmData.clazz, "stopRendering", "()V");
    if(terminateMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function stopRendering not found!";
        }
    } else {
        std::cout << "stopRendering function successfully found!";
    }

    env->CallVoidMethod(jvmData.obj, terminateMethod);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}

void setDatasetParams(JVMData jvmData, std::string dataset, float pixelToWorld, int dimensions[]) {

    jstring jdataset = jvmData.env->NewStringUTF(dataset.c_str());
    jfieldID datasetField = jvmData.env->GetFieldID(jvmData.clazz, "dataset", "Ljava/lang/String;");
    jvmData.env->SetObjectField(jvmData.obj, datasetField, jdataset);

    datasetName = dataset;

    jfieldID pixelToWorldField = jvmData.env->GetFieldID(jvmData.clazz, "pixelToWorld", "F");
    jvmData.env->SetFloatField(jvmData.obj, pixelToWorldField, pixelToWorld);

    jintArray jdims = jvmData.env->NewIntArray(3);
    jvmData.env->SetIntArrayRegion(jdims, 0, 3, dimensions);

    std::cout << "Trying to set the volume dims" <<std::endl;

    jmethodID setVolumeDimsMethod = jvmData.env->GetMethodID(jvmData.clazz, "setVolumeDims", "([I)V");
    if(setVolumeDimsMethod == nullptr) {
        if (jvmData.env->ExceptionOccurred()) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function setVolumeDims not found!";
        }
    } else {
        std::cout << "setVolumeDims function successfully found!";
    }

    jvmData.env->CallVoidMethod(jvmData.obj, setVolumeDimsMethod, jdims);
    if(jvmData.env->ExceptionOccurred()) {
        jvmData.env->ExceptionDescribe();
        jvmData.env->ExceptionClear();
    }
}

void setVDIGeneration(JVMData jvmData, bool generateVDIs) {
    jfieldID vdiField = jvmData.env->GetFieldID(jvmData.clazz, "generateVDIs", "Z");
    jvmData.env->SetBooleanField(jvmData.obj, vdiField, generateVDIs);
}

void setPointerAddresses(JVMData jvmData, MPI_Comm renderComm) {
    void * allToAllColorPointer = malloc(windowHeight * windowWidth * numSupersegments * 4 * 4);
    void * allToAllDepthPointer = malloc(windowWidth * windowHeight * numSupersegments * 4 * 2);
    void * gatherColorPointer = malloc(windowHeight * windowWidth * numOutputSupsegs * 4 * 4);
    void * gatherDepthPointer = malloc(windowHeight * windowWidth * numOutputSupsegs * 4 * 2);

    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    subColor_copy = malloc(windowHeight * windowWidth * numSupersegments * 4 * 4);
    subDepth_copy = malloc(windowHeight * windowWidth * numSupersegments * 2 * 4);

    compositedColor_copy = malloc(windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize);
    compositedDepth_copy = malloc(windowWidth * windowHeight * numOutputSupsegs * 4 * 2 / commSize);

//    a = malloc(windowHeight * windowWidth * numSupersegments * 4 * 4);
//    b = malloc(windowWidth * windowHeight * numSupersegments * 4 * 2);
//    c = malloc(windowHeight * windowWidth * numOutputSupsegs * 4 * 4);
//    d = malloc(windowHeight * windowWidth * numOutputSupsegs * 4 * 2);

    std::cout << "During initialization, gather color pointer: " << gatherColorPointer << std::endl;
    std::cout << "In long, color is: " << reinterpret_cast<long>(gatherColorPointer) << std::endl;
    std::cout << "During initialization, gather depth pointer: " << gatherDepthPointer << std::endl;
    std::cout << "In long, depth is: " << reinterpret_cast<long>(gatherDepthPointer) << std::endl;

    MPI_Comm * mpiPointer = &renderComm;

    jfieldID allC = jvmData.env->GetFieldID(jvmData.clazz, "allToAllColorPointer", "J");
    jvmData.env->SetLongField(jvmData.obj, allC, reinterpret_cast<long>(allToAllColorPointer));

    jfieldID allD = jvmData.env->GetFieldID(jvmData.clazz, "allToAllDepthPointer", "J");
    jvmData.env->SetLongField(jvmData.obj, allD, reinterpret_cast<long>(allToAllDepthPointer));

    jfieldID gatherC = jvmData.env->GetFieldID(jvmData.clazz, "gatherColorPointer", "J");
    jvmData.env->SetLongField(jvmData.obj, gatherC, reinterpret_cast<long>(gatherColorPointer));

    jfieldID gatherD = jvmData.env->GetFieldID(jvmData.clazz, "gatherDepthPointer", "J");
    jvmData.env->SetLongField(jvmData.obj, gatherD, reinterpret_cast<long>(gatherDepthPointer));

    jfieldID mpiPtr = jvmData.env->GetFieldID(jvmData.clazz, "mpiPointer", "J");
    jvmData.env->SetLongField(jvmData.obj, mpiPtr, reinterpret_cast<long>(mpiPointer));

    if(jvmData.env->ExceptionOccurred()) {
        jvmData.env->ExceptionDescribe();
        jvmData.env->ExceptionClear();
    }
}

void setRendererConfigured(JVMData jvmData) {

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jboolean ready = true;
    jfieldID readyField = env->GetFieldID(jvmData.clazz, "rendererConfigured", "Z");
    env->SetBooleanField(jvmData.obj, readyField, ready);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

void setMPIParams(JVMData jvmData , int rank, int node_rank, int commSize) {
    jfieldID rankField = jvmData.env->GetFieldID(jvmData.clazz, "rank", "I");
    jvmData.env->SetIntField(jvmData.obj, rankField, rank);

    jfieldID nodeRankField = jvmData.env->GetFieldID(jvmData.clazz, "nodeRank", "I");
    jvmData.env->SetIntField(jvmData.obj, nodeRankField, node_rank);

    jfieldID sizeField = jvmData.env->GetFieldID(jvmData.clazz, "commSize", "I");
    jvmData.env->SetIntField(jvmData.obj, sizeField, commSize);
}

void createVolume(JVMData jvmData, int volumeID, int dimensions[], float pos[], bool is16bit) {

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jmethodID addVolumeMethod = env->GetMethodID(jvmData.clazz, "addVolume", "(I[I[FZ)V");
    if(addVolumeMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function addVolume not found!";
        }
    } else {
        std::cout << "addVolume function successfully found!";
    }

    jintArray jdims = env->NewIntArray(3);
    jfloatArray jpos = env->NewFloatArray(3);
//    jboolean is16bit = true;

    env->SetIntArrayRegion(jdims, 0, 3, dimensions);
    env->SetFloatArrayRegion(jpos, 0, 3, pos);

    env->CallVoidMethod(jvmData.obj, addVolumeMethod, volumeID, jdims, jpos, is16bit);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}

void updateVolume(JVMData jvmData, int volumeID, char * buffer, long int buffer_size) {

    std::cout << "Buffer size is: " << buffer_size << std::endl;

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jmethodID updateVolumeMethod = env->GetMethodID(jvmData.clazz, "updateVolume", "(ILjava/nio/ByteBuffer;)V");
    if(updateVolumeMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function updateVolume not found!";
        }
    } else {
        std::cout << "updateVolume function successfully found!";
    }

    jobject jbuffer = env->NewDirectByteBuffer(buffer, buffer_size);

    env->CallVoidMethod(jvmData.obj, updateVolumeMethod, volumeID, jbuffer);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();

}

void doRender(JVMData jvmData) {
    JNIEnv *env;
//    jvmData.jvm->GetEnv((void **)&env, JNI_VERSION_1_6);
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);


    jmethodID mainMethod = env->GetMethodID(jvmData.clazz, "main", "()V");
    if(mainMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function main not found!";
//            std::exit(EXIT_FAILURE);
        }
    }
    env->CallVoidMethod(jvmData.obj, mainMethod);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}

void distributeVDIs(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer) {
    if (VERBOSE) std::cout<<"In distribute VDIs function. Comm size is "<<commSize<<std::endl;

    void *ptrCol = e->GetDirectBufferAddress(subVDICol);
    void *ptrDepth = nullptr;

#if SEPARATE_DEPTH
    ptrDepth = e->GetDirectBufferAddress(subVDIDepth);
#endif

//    begin_copy1 = std::chrono::high_resolution_clock::now();
//    memcpy(subColor_copy, ptrCol, windowHeight * windowWidth * numSupersegments * 4 * 4);
//    end_copy1 = std::chrono::high_resolution_clock::now();
//
//    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_copy1 - begin_copy1);
//    std::cout<<"Color copy in distributed took: " << elapsed.count() * 1e-9 << std::endl;
//
//    begin_copy2 = std::chrono::high_resolution_clock::now();
//    memcpy(subDepth_copy, ptrDepth, windowHeight * windowWidth * numSupersegments * 4 * 2);
//    end_copy2 = std::chrono::high_resolution_clock::now();
//
//    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_copy2 - begin_copy2);
//    std::cout<<"Depth copy in distributed took: " << elapsed.count() * 1e-9 << std::endl;

    void * recvBufCol;
    recvBufCol = reinterpret_cast<void *>(colPointer);

    if(recvBufCol == nullptr) {
        std::cout<<"allocating color buffer in distributeVDIs"<<std::endl;
        recvBufCol = malloc(sizePerProcess * 4 * commSize);
    }

#if SEPARATE_DEPTH
    void * recvBufDepth;
    recvBufDepth = reinterpret_cast<void *>(depthPointer);
    if(recvBufDepth == nullptr) {
        std::cout<<"allocating depth buffer in distributeVDIs"<<std::endl;
        recvBufDepth = malloc(sizePerProcess * 2 * commSize);
    }
#endif

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();
    begin1 = std::chrono::high_resolution_clock::now();
//    MPI_Alltoall(subColor_copy, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, recvBufCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);
    MPI_Alltoall(ptrCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, recvBufCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);
    end1 = std::chrono::high_resolution_clock::now();

    auto elapsed_col = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1);

//    std::cout<<"Iteration: "<< num_alltoall << " AllToAll color took in seconds: " << elapsed_col.count() * 1e-9 << std::endl;

#if SEPARATE_DEPTH
    begin2 = std::chrono::high_resolution_clock::now();
//    MPI_Alltoall(subDepth_copy, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, recvBufDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, MPI_COMM_WORLD);
    MPI_Alltoall(ptrDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, recvBufDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, MPI_COMM_WORLD);
    end2 = std::chrono::high_resolution_clock::now();

    auto elapsed_depth = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2);

//    std::cout<<"AllToAll depth took seconds: " << elapsed_depth.count() * 1e-9 << std::endl;

#endif

    double local_alltoall = (elapsed_col.count() + elapsed_depth.count()) * 1e-9;

    double global_sum;
    MPI_Reduce(&local_alltoall, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_alltoall = global_sum / commSize;

    if(num_alltoall > warm_up_iterations) {
        total_alltoall += global_alltoall;
    }

    num_alltoall++;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(((num_alltoall % 50) == 0) && (rank == 0)) {
        int iterations = num_alltoall - warm_up_iterations;
        double average_alltoall = total_alltoall / (double) iterations;
        std::cout<< "Number of alltoalls: " << num_alltoall << " average alltoall time so far: " << average_alltoall << std::endl;
    }

    if(VERBOSE) printf("Finished both alltoalls\n");

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "compositeVDIs", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V");

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, sizePerProcess * commSize * 4);

    jobject bbDepth;

#if SEPARATE_DEPTH
    bbDepth = e->NewDirectByteBuffer( recvBufDepth, sizePerProcess * commSize * 2);
#else
    bbDepth = e->NewDirectByteBuffer(recvBufDepth, 0);
#endif

    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }

    if (VERBOSE) std::cout<<"Finished distributing the VDIs. Calling the Composite method now!"<<std::endl;

    e->CallVoidMethod(clazzObject, compositeMethod, bbCol, bbDepth);
    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }
}

void gatherCompositedVDIs(JNIEnv *e, jobject clazzObject, jobject compositedVDIColor, jobject compositedVDIDepth, jint compositedVDILen, jint root, jint myRank, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer) {

    if (VERBOSE) std::cout<<"In Gather function " <<std::endl;

//    std::cout << "color pointer in long: " << colPointer << std::endl;
//    std::cout << "In void *, color is: " << reinterpret_cast<void *>(colPointer) << std::endl;
//    std::cout << "depth pointer in long: " << depthPointer << std::endl;
//    std::cout << "In void *, depth is: " << reinterpret_cast<void *>(depthPointer) << std::endl;

    if (VERBOSE) std::cout<<"Col buff capacity: " << e->GetDirectBufferCapacity(compositedVDIColor) <<std::endl;
    if (VERBOSE) std::cout<<"Depth buff capacity: " << e->GetDirectBufferCapacity(compositedVDIDepth) <<std::endl;

    if (VERBOSE) std::cout<<"CompositedVDILen: " << compositedVDILen << "root" << root << " myRank: " << myRank << " commSize: " << commSize << std::endl;

    void *ptrCol = e->GetDirectBufferAddress(compositedVDIColor);
    void *ptrDepth = e->GetDirectBufferAddress(compositedVDIDepth);

//    begin_copy1 = std::chrono::high_resolution_clock::now();
//    memcpy(compositedColor_copy, ptrCol, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize);
//    end_copy1 = std::chrono::high_resolution_clock::now();
//
//    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_copy1 - begin_copy1);
//    std::cout<<"Color copy in gather took: " << elapsed.count() * 1e-9 << std::endl;
//
//    begin_copy2 = std::chrono::high_resolution_clock::now();
//    memcpy(compositedDepth_copy, ptrDepth, windowWidth * windowHeight * numOutputSupsegs * 2 * 4 / commSize);
//    end_copy2 = std::chrono::high_resolution_clock::now();
//
//    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_copy2 - begin_copy2);
//    std::cout<<"Depth copy in gather took: " << elapsed.count() * 1e-9 << std::endl;


    void * gather_recv_color = reinterpret_cast<void *>(colPointer);
    void * gather_recv_depth = reinterpret_cast<void *>(depthPointer);

    if (myRank == 0) {
        if(gather_recv_color == nullptr) {
            std::cout<<"allocating color receive buffer in gather"<<std::endl;
            gather_recv_color = malloc(compositedVDILen * commSize * 4);
        }
        if(gather_recv_depth == nullptr) {
            std::cout<<"allocating depth receive buffer in gather"<<std::endl;
            gather_recv_depth = malloc(compositedVDILen * commSize * 2);
        }
    }

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    MPI_Barrier(MPI_COMM_WORLD);
    begin3 = std::chrono::high_resolution_clock::now();
//    MPI_Gather(compositedColor_copy, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, gather_recv_color, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);
    MPI_Gather(ptrCol, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, gather_recv_color, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);
    end3 = std::chrono::high_resolution_clock::now();

    auto elapsed_col = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3);

//    std::cout<<"Iteration: " << num_gather << " Gather color took seconds: " << elapsed_col.count() * 1e-9 << std::endl;

    begin4 = std::chrono::high_resolution_clock::now();
//    MPI_Gather(compositedDepth_copy, windowWidth  * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE,  gather_recv_depth, windowWidth * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);
    MPI_Gather(ptrDepth, windowWidth  * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE,  gather_recv_depth, windowWidth * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);
    end4 = std::chrono::high_resolution_clock::now();

    end = std::chrono::high_resolution_clock::now();

    auto elapsed_depth = std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - begin4);

//    std::cout<<"Gather depth took seconds: " << elapsed_depth.count() * 1e-9 << std::endl;
    //The data is here now!

    double local_gather = (elapsed_col.count() + elapsed_depth.count()) * 1e-9;

    double global_sum;

    MPI_Reduce(&local_gather, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_gather = global_sum / commSize;

    auto elapsed_overall = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    auto elapsed_whole = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin_whole);

//    printf("Time measured: %.3f seconds.\n", elapsed_overall.count() * 1e-9);
    double local_overall = elapsed_overall.count() * 1e-9;

    global_sum = 0;
    MPI_Reduce(&local_overall, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_overall = global_sum / commSize;

    if(num_gather > warm_up_iterations) {
        total_gather += global_gather;
        total_overall += global_overall;
        total_whole += elapsed_whole.count() * 1e-9;
    }

    num_gather++;

    if(((num_gather % 50) == 0) && (myRank == 0)) {
        int iterations = num_gather - warm_up_iterations;
        double average_gather = total_gather / (double)iterations;
        double average_overall = total_overall / (double) iterations;
        double average_whole = total_whole / (double) iterations;
        std::cout<< "Number of gathers: " << num_gather << " average_gather gather time so far: " << average_gather << " average overall so far: " << average_overall << " average whole " << average_whole<<std::endl;
    }

    std::string dataset = datasetName;

    dataset += "_" + std::to_string(commSize) + "_" + std::to_string(myRank);

    std::string basePath = "/home/aryaman/TestingData/";

    if(num_gather % 20 == 0) {

    }

    if(myRank == 0) {
//        //send or store the VDI

        if(SAVE_FILES) {
            std::time_t t = std::time(0);

            std::cout<<"Writing the final gathered VDI now"<<std::endl;

            std::string filename = basePath + dataset + "FinalVDI" + std::to_string(count) + "_ndc_col";
            std::ofstream b_stream(filename.c_str(),
                                   std::fstream::out | std::fstream::binary);
            std::string filenameDepth = basePath + dataset + "FinalVDI" + std::to_string(count) + "_ndc_depth";
            std::ofstream b_streamDepth(filenameDepth.c_str(),
                                   std::fstream::out | std::fstream::binary);

            if (b_stream)
            {
                b_stream.write(static_cast<const char *>(gather_recv_color), windowHeight * windowWidth * numOutputSupsegs * 4 * 4);
                b_streamDepth.write(static_cast<const char *>(gather_recv_depth), windowHeight * windowWidth * numOutputSupsegs * 4 * 2);

                if (b_stream.good()) {
                    std::cout<<"Writing was successful"<<std::endl;
                }
            }
            count++;
        }
    }

    begin_whole = std::chrono::high_resolution_clock::now();
}