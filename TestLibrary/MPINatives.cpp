#include <mpi.h>
#include "MPINatives.hpp"
#include "VDIParams.hpp"
#include <fstream>
#include <chrono>
#include <cmath>

int count = 0;

void setPointerAddresses(JVMData jvmData, MPI_Comm renderComm) {
    void * allToAllColorPointer = malloc(windowHeight * windowWidth * numSupersegments * 4 * 4);
    void * allToAllDepthPointer = malloc(windowWidth * windowHeight * numSupersegments * 4 * 2);
    void * allToAllPrefixPointer = malloc(windowWidth * windowHeight * 4);
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

    jfieldID allP = jvmData.env->GetFieldID(jvmData.clazz, "allToAllPrefixPointer", "J");
    jvmData.env->SetLongField(jvmData.obj, allP, reinterpret_cast<long>(allToAllPrefixPointer));

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
                                { (char *)"distributeDenseVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;[IIJJJJ)V", (void *)&distributeDenseVDIs},

//                                { (char *)"distributeVDIsForBenchmark", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIJJJII)V", (void *)&distributeVDIsForBenchmark },
//                                { (char *)"distributeVDIsWithVariableLength", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;[I[IIJJJZII)V", (void *)&distributeVDIsWithVariableLength },
                                { (char *)"gatherCompositedVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIJJJ)V", (void *)&gatherCompositedVDIs },

    };

    int ret = jvmData.env->RegisterNatives(jvmData.clazz, methods, 3);
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

int distributeVariable(int *counts, int *countsRecv, void * sendBuf, void * recvBuf, int commSize, const std::string& purpose = "") {

    std::cout<<"Performing distribution of " << purpose <<std::endl;

    MPI_Alltoall(counts, 1, MPI_INT, countsRecv, 1, MPI_INT, MPI_COMM_WORLD);

    //set up the AllToAllv
    int displacementSendSum = 0;
    int displacementSend[commSize];

    int displacementRecvSum = 0;
    int displacementRecv[commSize];

    for( int i = 0 ; i < commSize ; i ++){
        displacementSend[i] = displacementSendSum;
        displacementSendSum += counts[i];

        displacementRecv[i] = displacementRecvSum;
        displacementRecvSum += countsRecv[i];
    }

    if(recvBuf == nullptr) {
        std::cout<<"This is an error! Receive buffer needs to be preallocated with sufficient size"<<std::endl;
        int sum = 0;
        for( int i = 0 ; i < commSize ; i++) {
            sum += countsRecv[i];
        }
        recvBuf = malloc(sum);
    }

    MPI_Alltoallv(sendBuf, counts, displacementSend, MPI_BYTE, recvBuf, countsRecv, displacementRecv, MPI_BYTE, MPI_COMM_WORLD);

    return displacementRecvSum;
}

void distributeDenseVDIs(JNIEnv *e, jobject clazzObject, jobject colorVDI, jobject depthVDI, jobject prefixSums, jintArray supersegmentCounts, jint commSize, jlong colPointer, jlong depthPointer, jlong prefixPointer, jlong mpiPointer) {
    std::cout<<"In distribute dense VDIs function. Comm size is "<<commSize<<std::endl;

    int *supsegCounts = e->GetIntArrayElements(supersegmentCounts, NULL);

    auto beginAllToAll = std::chrono::high_resolution_clock::now();

    void *ptrCol = e->GetDirectBufferAddress(colorVDI);
    void *ptrDepth = e->GetDirectBufferAddress(depthVDI);

    void * recvBufCol;
    recvBufCol = reinterpret_cast<void *>(colPointer);

    void * recvBufDepth;
    recvBufDepth = reinterpret_cast<void *>(depthPointer);

    int * colorCounts = new int[commSize];
    int * depthCounts = new int[commSize];

    for(int i = 0; i < commSize; i++) {
        colorCounts[i] = supsegCounts[i] * 4 * 4;
        depthCounts[i] = supsegCounts[i] * 4 * 2;
    }

    int * colorCountsRecv = new int[commSize];
    int * depthCountsRecv = new int[commSize];

    int totalRecvdColor = distributeVariable(colorCounts, colorCountsRecv, ptrCol, recvBufCol, commSize, "color");
    int totalRecvdDepth = distributeVariable(depthCounts, depthCountsRecv, ptrDepth, recvBufDepth, commSize, "depth");

    std::cout << "total bytes recvd: color: " << totalRecvdColor << " depth: " << totalRecvdDepth << std::endl;

    void * recvBufPrefix = reinterpret_cast<void *>(prefixPointer);
    void *ptrPrefix = e->GetDirectBufferAddress(prefixSums);

    MPI_Alltoall(ptrPrefix, windowWidth * windowHeight * 4 / commSize, MPI_BYTE, recvBufPrefix, windowWidth * windowHeight * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);

    auto endAllToAll = std::chrono::high_resolution_clock::now();

    printf("Finished both alltoalls for the dense VDIs\n");

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "uploadForCompositingDense", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;[I[I)V");

    int supsegsRecvd = totalRecvdColor / (4 * 4);

    long supsegsInBuffer = 512 * 512 * (long)ceil((double)supsegsRecvd / (512.0*512.0));

    std::cout << "The number of supsegs recvd: " << supsegsRecvd << " and stored: " << supsegsInBuffer << std::endl;

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, supsegsInBuffer * 4 * 4);

    jobject bbDepth = e->NewDirectByteBuffer(recvBufDepth, supsegsInBuffer * 4 * 2);

    jobject bbPrefix = e->NewDirectByteBuffer( recvBufPrefix, windowWidth * windowHeight * 4);

    jintArray javaColorCounts = e->NewIntArray(commSize);
    e->SetIntArrayRegion(javaColorCounts, 0, commSize, colorCountsRecv);

    jintArray javaDepthCounts = e->NewIntArray(commSize);
    e->SetIntArrayRegion(javaDepthCounts, 0, commSize, depthCountsRecv);

    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }

    std::cout<<"Finished distributing the VDIs. Calling the dense Composite method now!"<<std::endl;

    e->CallVoidMethod(clazzObject, compositeMethod, bbCol, bbDepth, bbPrefix, javaColorCounts, javaDepthCounts);
    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }
}

// isBenchmark, rank & iteration don't need to be set -> results in no benchmark
void distributeVDIsWithVariableLength(JNIEnv *e, jobject clazzObject, jobject colorVDI, jobject depthVDI, jintArray colorLimits, jintArray depthLimits , jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer, jboolean isBenchmark, jint rank , jint iteration ) {
    std::cout<<"In distribute Compressed VDIs (Benchmark) function. Comm size is "<<commSize<<std::endl;

    int *colLimits = e->GetIntArrayElements(colorLimits, NULL);
    int *depLimits = e->GetIntArrayElements(depthLimits, NULL);

    auto beginAllToAll = std::chrono::high_resolution_clock::now();

    void *ptrCol = e->GetDirectBufferAddress(colorVDI);
    void *ptrDepth = e->GetDirectBufferAddress(depthVDI);

    void * recvBufCol;
    recvBufCol = reinterpret_cast<void *>(colPointer);

    void * recvBufDepth;
    recvBufDepth = reinterpret_cast<void *>(depthPointer);

    int * colorLimitsRecv = new int[commSize];
    int * depthLimitsRecv = new int[commSize];

    int displacementRecvSumColor = distributeVariable(colLimits, colorLimitsRecv, ptrCol, recvBufCol, commSize, "color");
    int displacementRecvSumDepth = distributeVariable(depLimits, depthLimitsRecv, ptrDepth, recvBufDepth, commSize, "depth");

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