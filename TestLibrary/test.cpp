#include <iostream>
#include <dirent.h>
#include <fstream>
#include "TestLiibrary.hpp"
#include <mpi.h>
#include <ctime>

#define VERBOSE true
#define USE_VULKAN true
#define SEPARATE_DEPTH true
#define SAVE_FILES true

int count = 0;

int windowWidth = 1280;
int windowHeight = 720;
int numSupersegments = 20;
int numOutputSupsegs = 40;

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

JVMData func(int rank, bool isCluster) {
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
    auto *options = new JavaVMOption[4];   // JVM invocation options
    options[0].optionString = (char *)classPath.c_str();

    #if USE_VULKAN
    options[1].optionString = (char *)
            "-Dscenery.Renderer=VulkanRenderer";
    #else
    options[1].optionString = (char *)
            "-Dscenery.Renderer=OpenGLRenderer";
    #endif

    #if VERBOSE
    options[2].optionString = (char *)
            "-Dorg.lwjgl.system.stackSize=1000";
    #else
    options[2].optionString = (char *)
            "-Dscenery.LogLevel=warn";
    #endif

    options[3].optionString = (char *)
            "-Dscenery.Headless=true";

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

void setPixelToWorld(JVMData jvmData, float pixelToWorld) {
    jfieldID pixelToWorldField = jvmData.env->GetFieldID(jvmData.clazz, "pixelToWorld", "F");
    jvmData.env->SetFloatField(jvmData.obj, pixelToWorldField, pixelToWorld);
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

void setMPIParams(JVMData jvmData , int rank, int commSize) {
    jfieldID rankField = jvmData.env->GetFieldID(jvmData.clazz, "rank", "I");
    jvmData.env->SetIntField(jvmData.obj, rankField, rank);

    jfieldID sizeField = jvmData.env->GetFieldID(jvmData.clazz, "commSize", "I");
    jvmData.env->SetIntField(jvmData.obj, sizeField, commSize);
}

void createVolume(JVMData jvmData, int volumeID, int dimensions[], float pos[]) {

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
    jboolean is16bit = true;

    env->SetIntArrayRegion(jdims, 0, 3, dimensions);
    env->SetFloatArrayRegion(jpos, 0, 3, pos);

    env->CallVoidMethod(jvmData.obj, addVolumeMethod, volumeID, jdims, jpos, is16bit);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}

void updateVolume(JVMData jvmData, int volumeID, char * buffer, long buffer_size) {

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

    printf("Here 1\n");

    void *ptrCol = e->GetDirectBufferAddress(subVDICol);
    void *ptrDepth = nullptr;

    printf("Here 1.5\n");

#if SEPARATE_DEPTH
    ptrDepth = e->GetDirectBufferAddress(subVDIDepth);
#endif

    printf("Here 2\n");

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

    printf("Trying to get the MPI comm\n");

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    printf("Got MPI comm, trying alltoall\n");

    MPI_Alltoall(ptrCol, sizePerProcess * 4, MPI_BYTE, recvBufCol, sizePerProcess * 4, MPI_BYTE, MPI_COMM_WORLD);

#if SEPARATE_DEPTH
        MPI_Alltoall(ptrDepth, sizePerProcess * 2, MPI_BYTE, recvBufDepth, sizePerProcess * 2, MPI_BYTE, MPI_COMM_WORLD);
#endif

    printf("Finished both alltoalls\n");

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "compositeVDIs", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V");

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, sizePerProcess * commSize * 4);

    jobject bbDepth;

#if SEPARATE_DEPTH
    bbDepth = e->NewDirectByteBuffer(recvBufDepth, sizePerProcess * commSize * 2);
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

    std::cout<<"Col buff capacity: " << e->GetDirectBufferCapacity(compositedVDIColor) <<std::endl;
    std::cout<<"Depth buff capacity: " << e->GetDirectBufferCapacity(compositedVDIDepth) <<std::endl;

    std::cout<<"CompositedVDILen: " << compositedVDILen << "root" << root << " myRank: " << myRank << " commSize: " << commSize << std::endl;

    void *ptrCol = e->GetDirectBufferAddress(compositedVDIColor);
    void *ptrDepth = e->GetDirectBufferAddress(compositedVDIDepth);

//    ptrCol = malloc(294912000);
//    ptrDepth = malloc(147456000);

    void * gather_recv_color = reinterpret_cast<void *>(colPointer);
    void * gather_recv_depth = reinterpret_cast<void *>(depthPointer);

    if (myRank == 0) {
//        if(gather_recv_color == nullptr) {
            std::cout<<"allocating color receive buffer in gather"<<std::endl;
            gather_recv_color = malloc(compositedVDILen * commSize * 4);
//        }
//        if(gather_recv_depth == nullptr) {
            std::cout<<"allocating depth receive buffer in gather"<<std::endl;
            gather_recv_depth = malloc(compositedVDILen * commSize * 2);
//        }
    }

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    MPI_Gather(ptrCol, compositedVDILen * 4, MPI_BYTE, gather_recv_color, compositedVDILen * 4, MPI_BYTE, root, MPI_COMM_WORLD);
    MPI_Gather(ptrDepth, compositedVDILen * 2, MPI_BYTE,  gather_recv_depth, compositedVDILen * 2, MPI_BYTE, root, MPI_COMM_WORLD);
    //The data is here now!

    std::string dataset = "DistributedStagbeetle";

    dataset += "_" + std::to_string(commSize) + "_" + std::to_string(myRank);

    std::string basePath = "/home/aryaman/TestingData/";

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
                b_stream.write(static_cast<const char *>(gather_recv_color), compositedVDILen * commSize * 4);
                b_streamDepth.write(static_cast<const char *>(gather_recv_depth), compositedVDILen * commSize * 2);

                if (b_stream.good()) {
                    std::cout<<"Writing was successful"<<std::endl;
                }
            }
            count++;
        }
    }
}