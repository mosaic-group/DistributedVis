#include <mpi.h>
#include "MPINatives.hpp"
#include "VDIParams.hpp"
#include <fstream>
#include <chrono>

int count = 0;

void setPointerAddresses(JVMData jvmData, MPI_Comm renderComm) {
    void * allToAllColorPointer = malloc(windowHeight * windowWidth * numSupersegments * 4 * 4);
    void * allToAllDepthPointer = malloc(windowWidth * windowHeight * numSupersegments * 4 * 2);
    void * gatherColorPointer = malloc(windowHeight * windowWidth * numOutputSupsegs * 4 * 4);
    void * gatherDepthPointer = malloc(windowHeight * windowWidth * numOutputSupsegs * 4 * 2);

    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

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

void setMPIParams(JVMData jvmData , int rank, int node_rank, int commSize) {
    jfieldID rankField = jvmData.env->GetFieldID(jvmData.clazz, "rank", "I");
    jvmData.env->SetIntField(jvmData.obj, rankField, rank);

    jfieldID nodeRankField = jvmData.env->GetFieldID(jvmData.clazz, "nodeRank", "I");
    jvmData.env->SetIntField(jvmData.obj, nodeRankField, node_rank);

    jfieldID sizeField = jvmData.env->GetFieldID(jvmData.clazz, "commSize", "I");
    jvmData.env->SetIntField(jvmData.obj, sizeField, commSize);
}

void registerNatives(JVMData jvmData) {
    JNINativeMethod methods[] { { (char *)"distributeVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIJJJ)V", (void *)&distributeVDIs },
                                { (char *)"distributeVDIsForBenchmark", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIJJJII)V", (void *)&distributeVDIsForBenchmark },
                                { (char *)"distributeVDIsWithVariableLength", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;[I[IIJJJZII)V", (void *)&distributeVDIsWithVariableLength },
                                { (char *)"gatherCompositedVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIJJJ)V", (void *)&gatherCompositedVDIs },

    };

    int ret = jvmData.env->RegisterNatives(jvmData.clazz, methods, 4);
    if(ret < 0) {
        if( jvmData.env->ExceptionOccurred() ) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cerr << "ERROR: Could not register natives!" <<std::endl;
            //std::exit(EXIT_FAILURE);
        }
    } else {
        std::cout<<"Natives registered. The return value is: "<< ret <<std::endl;
    }
}



void distributeVDIs(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer) {
    std::cout<<"In distribute VDIs function. Comm size is "<<commSize<<std::endl;


    void *ptrCol = e->GetDirectBufferAddress(subVDICol);
    void *ptrDepth = e->GetDirectBufferAddress(subVDIDepth);

    void * recvBufCol;
    recvBufCol = reinterpret_cast<void *>(colPointer);

    if(recvBufCol == nullptr) {
        std::cout<<"allocating color buffer in distributeVDIs"<<std::endl;
        recvBufCol = malloc(sizePerProcess * 4 * commSize);
    }

    void * recvBufDepth;
    recvBufDepth = reinterpret_cast<void *>(depthPointer);
    if(recvBufDepth == nullptr) {
        std::cout<<"allocating depth buffer in distributeVDIs"<<std::endl;
        recvBufDepth = malloc(sizePerProcess * 2 * commSize);
    }

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    std::cout<<"Starting all to all"<<std::endl;

    MPI_Alltoall(ptrCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, recvBufCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);

    std::cout<<"Finished color all to all"<<std::endl;

    MPI_Alltoall(ptrDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, recvBufDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, MPI_COMM_WORLD);

    printf("Finished both alltoalls\n");

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "uploadForCompositing", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V");

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, sizePerProcess * commSize * 4);

    jobject bbDepth;

    bbDepth = e->NewDirectByteBuffer( recvBufDepth, sizePerProcess * commSize * 2);

    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }

    std::cout<<"Finished distributing the VDIs. Calling the Composite method now!"<<std::endl;

    e->CallVoidMethod(clazzObject, compositeMethod, bbCol, bbDepth);
    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }
}

void distributeVDIsForBenchmark(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer, jint rank, jint iteration) {
    std::cout<<"In distribute VDIs function. Comm size is "<<commSize<<std::endl;

    auto beginAllToAll = std::chrono::high_resolution_clock::now();

    void *ptrCol = e->GetDirectBufferAddress(subVDICol);
    void *ptrDepth = e->GetDirectBufferAddress(subVDIDepth);

    void * recvBufCol;
    recvBufCol = reinterpret_cast<void *>(colPointer);

    if(recvBufCol == nullptr) {
        std::cout<<"allocating color buffer in distributeVDIs"<<std::endl;
        recvBufCol = malloc(sizePerProcess * 4 * commSize);
    }

    void * recvBufDepth;
    recvBufDepth = reinterpret_cast<void *>(depthPointer);
    if(recvBufDepth == nullptr) {
        std::cout<<"allocating depth buffer in distributeVDIs"<<std::endl;
        recvBufDepth = malloc(sizePerProcess * 2 * commSize);
    }

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    MPI_Alltoall(ptrCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, recvBufCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);

    MPI_Alltoall(ptrDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, recvBufDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, MPI_COMM_WORLD);

    auto endAllToAll = std::chrono::high_resolution_clock::now();

    auto elapsed_AllToAll = std::chrono::duration_cast<std::chrono::nanoseconds>(endAllToAll - beginAllToAll);
    std::cout << "AllToAll Values took in seconds: #ALLVAL:"<< rank << ":" << iteration << ":" << elapsed_AllToAll.count() * 1e-9 << "#"<< std::endl;


    printf("Finished both alltoalls\n");

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "uploadForCompositing", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V");

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, sizePerProcess * commSize * 4);

    jobject bbDepth;

    bbDepth = e->NewDirectByteBuffer( recvBufDepth, sizePerProcess * commSize * 2);

    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }

    std::cout<<"Finished distributing the VDIs. Calling the Composite method now!"<<std::endl;

    e->CallVoidMethod(clazzObject, compositeMethod, bbCol, bbDepth);
    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }
}

// isBenchmark, rank & iteration don't need to be set -> results in no benchmark
void distributeVDIsWithVariableLength(JNIEnv *e, jobject clazzObject, jobject colorVDI, jobject depthVDI, jintArray colorLimits, jintArray depthLimits , jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer, jboolean isBenchmark, jint rank , jint iteration ) {
    std::cout<<"In distribute Compressed VDIs (Benchmark) function. Comm size is "<<commSize<<std::endl;

    auto beginAllToAllLimits = std::chrono::high_resolution_clock::now();

    jint *colLimits = e->GetIntArrayElements(colorLimits, NULL);
    jint *depLimits = e->GetIntArrayElements(depthLimits, NULL);

    //first send the limits to each process
    int colorLimitsRecv[commSize];
    int depthLimitsRecv[commSize];

    MPI_Alltoall(colLimits, 1, MPI_INT, colorLimitsRecv, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoall(depLimits, 1, MPI_INT, depthLimitsRecv, 1, MPI_INT, MPI_COMM_WORLD);

    auto endAllToAllLimits = std::chrono::high_resolution_clock::now();

    if(isBenchmark){
        auto elapsed_AllToAllLimits = std::chrono::duration_cast<std::chrono::nanoseconds>(endAllToAllLimits - beginAllToAllLimits);
        std::cout << "AllToAll Limits took in seconds: #ALLLIM:"<< rank << ":" << iteration << ":" << elapsed_AllToAllLimits.count() * 1e-9 << "#" << std::endl;
    }

    //sync all processes
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout<<"Starting all to all with Compression"<<std::endl;

    auto beginAllToAll = std::chrono::high_resolution_clock::now();

    void *ptrCol = e->GetDirectBufferAddress(colorVDI);
    void *ptrDepth = e->GetDirectBufferAddress(depthVDI);

    void * recvBufCol;
    recvBufCol = reinterpret_cast<void *>(colPointer);

    if(recvBufCol == nullptr) {
        std::cout<<"allocating color buffer in distributeVDIs"<<std::endl;
        int sum = 0;
        for( int i = 0 ; i < commSize ; i++) {
            sum += colorLimitsRecv[i];
        }
        recvBufCol = malloc(sum);
    }

    void * recvBufDepth;
    recvBufDepth = reinterpret_cast<void *>(depthPointer);
    if(recvBufDepth == nullptr) {
        std::cout<<"allocating depth buffer in distributeVDIs"<<std::endl;
        int sum = 0;
        for( int i = 0 ; i < commSize ; i++) {
            sum += depthLimitsRecv[i];
        }
        recvBufDepth = malloc(sum);
    }

    auto * renComm = reinterpret_cast<MPI_Comm *>(mpiPointer);

    //set up the AllToAllv
    int displacementSendSumColor = 0;
    int displacementSendSumDepth = 0;
    int displacementSendColor[commSize];
    int displacementSendDepth[commSize];

    int displacementRecvSumColor = 0;
    int displacementRecvSumDepth = 0;
    int displacementRecvColor[commSize];
    int displacementRecvDepth[commSize];

    for( int i = 0 ; i < commSize ; i ++){
        displacementSendColor[i] = displacementSendSumColor;
        displacementSendDepth[i] = displacementSendSumDepth;
        displacementSendSumColor += colLimits[i];
        displacementSendSumDepth += depLimits[i];

        displacementRecvColor[i] = displacementRecvSumColor;
        displacementRecvDepth[i] = displacementRecvSumDepth;
        displacementRecvSumColor += colorLimitsRecv[i];
        displacementRecvSumDepth += depthLimitsRecv[i];
    }

    MPI_Alltoallv(ptrCol, colLimits, displacementSendColor,MPI_BYTE, recvBufCol, colorLimitsRecv, displacementRecvColor,MPI_BYTE, MPI_COMM_WORLD);

    MPI_Alltoallv(ptrDepth, depLimits, displacementSendDepth, MPI_BYTE, recvBufDepth, depthLimitsRecv, displacementRecvDepth,  MPI_BYTE, MPI_COMM_WORLD);

    auto endAllToAll = std::chrono::high_resolution_clock::now();

    if(isBenchmark){
        auto elapsed_AllToAll = std::chrono::duration_cast<std::chrono::nanoseconds>(endAllToAll - beginAllToAll);
        std::cout << "AllToAll Values took in seconds: #ALLVAL:"<< rank << ":" << iteration << ":" << elapsed_AllToAll.count() * 1e-9 << "#"<< std::endl;
    }

    printf("Finished both alltoalls with Compression\n");

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "handleReceivedBuffersAndUploadForCompositing", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;[I[I)V");

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, displacementRecvSumColor);

    jobject bbDepth = e->NewDirectByteBuffer( recvBufDepth, displacementRecvSumDepth);

    jintArray javaColorLimits = e->NewIntArray(commSize);
    e->SetIntArrayRegion(javaColorLimits, 0, commSize, colorLimitsRecv);

    jintArray javaDepthLimits = e->NewIntArray(commSize);
    e->SetIntArrayRegion(javaDepthLimits, 0, commSize, depthLimitsRecv);


    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }

    std::cout<<"Finished distributing the VDIs. Calling the decompression Composite method now!"<<std::endl;

    e->CallVoidMethod(clazzObject, compositeMethod, bbCol, bbDepth, javaColorLimits, javaDepthLimits);
    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }
}



void gatherCompositedVDIs(JNIEnv *e, jobject clazzObject, jobject compositedVDIColor, jobject compositedVDIDepth, jint compositedVDILen, jint root, jint myRank, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer) {

    std::cout<<"In Gather function " <<std::endl;

    void *ptrCol = e->GetDirectBufferAddress(compositedVDIColor);
    void *ptrDepth = e->GetDirectBufferAddress(compositedVDIDepth);

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

    MPI_Gather(ptrCol, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, gather_recv_color, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);

    MPI_Gather(ptrDepth, windowWidth  * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE,  gather_recv_depth, windowWidth * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);

    std::string dataset = datasetName;

    dataset += "_" + std::to_string(commSize) + "_" + std::to_string(myRank);

    std::string basePath = "/home/aryaman/TestingData/";

    if(myRank == 0) {
//        //send or store the VDI

        if(true) {

            std::cout<<"Writing the final gathered VDI now"<<std::endl;

            std::string filename = basePath + dataset + "FinalVDI_" + std::to_string(windowWidth) + "_" + std::to_string(windowHeight) + "_" + std::to_string(numOutputSupsegs)
                                   + "_0_" + std::to_string(count) + "_ndc";

            std::string filenameCol = filename + "_col";

            std::ofstream b_stream(filenameCol.c_str(),
                                   std::fstream::out | std::fstream::binary);
            std::string filenameDepth = filename + "_depth";
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

}

void setProgramSettings(JVMData jvmData, std::string dataset, bool withCompression, bool benchmarkValues){

    jstring jdataset = jvmData.env->NewStringUTF(dataset.c_str());
    jfieldID datasetField = jvmData.env->GetFieldID(jvmData.clazz, "dataset", "Ljava/lang/String;");
    jvmData.env->SetObjectField(jvmData.obj, datasetField, jdataset);

    jboolean jWithCompression = withCompression;
    jfieldID withCompressionField = jvmData.env->GetFieldID(jvmData.clazz, "isCompressed", "Z");
    jvmData.env->SetBooleanField(jvmData.obj, withCompressionField, jWithCompression);

    jboolean jBenchmarkValues = benchmarkValues;
    jfieldID benchmarkValuesField = jvmData.env->GetFieldID(jvmData.clazz, "isBenchmark", "Z");
    jvmData.env->SetBooleanField(jvmData.obj, benchmarkValuesField, jBenchmarkValues);

}