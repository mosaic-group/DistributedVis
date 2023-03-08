#include <mpi.h>
#include "MPINatives.hpp"
#include "VDIParams.hpp"
#include <cmath>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <IceT.h>
#include <IceTGL.h>
#include <IceTMPI.h>
#include <IceTDevCommunication.h>
#include <IceTDevState.h>
#include <zconf.h>
#include <vector>


#define VERBOSE false

#define PROFILING true

int count = 0;

std::vector<float> distributeTimes;
std::vector<float> gatherTimes;
std::vector<float> wholeCompositeTimes;
std::vector<float> wholeVDITimes;
std::vector<long> numSupsegsGenerated;
std::vector<float> globalDistributeTimes;
std::vector<float> globalGatherTimes;
std::vector<float> globalWholeCompositeTimes;
std::vector<float> globalWholeVDITimes;
std::vector<long> globalNumSupsegsGenerated;

auto begin_whole_vdi = std::chrono::high_resolution_clock::now();
auto end_whole_vdi = std::chrono::high_resolution_clock::now();
auto begin = std::chrono::high_resolution_clock::now();
auto end = std::chrono::high_resolution_clock::now();

auto begin_whole_compositing = std::chrono::high_resolution_clock::now();
auto end_whole_compositing = std::chrono::high_resolution_clock::now();

double total_alltoall = 0;
double total_gather = 0;
double total_whole_compositing = 0;
double total_whole_vdi = 0;
long int num_alltoall = 0;
long int num_gather = 0;
long int num_whole_vdi = 0;

int warm_up_iterations = 10;
int total_iterations = 144;

extern std::string datasetName;
extern bool dataset16bit;

void writeBenchmarkFile(const std::string& description, const std::vector<float>& data, int commSize, int myRank) {

    std::ofstream benchmarkFile(datasetName + "_" + std::to_string(numOutputSupsegs) + "_" + description + "_" + std::to_string(commSize) + "_" + std::to_string(myRank)
                                 + ".csv");

    for(auto val: data) {
        benchmarkFile << std::to_string(val) << ", ";
    }
    benchmarkFile << std::endl;
    benchmarkFile.close();
}

void writeBenchmarkFile(const std::string& description, const std::vector<long>& data, int commSize, int myRank) {

    std::ofstream benchmarkFile(datasetName + "_" + std::to_string(numOutputSupsegs) + "_" + description + "_" + std::to_string(commSize) + "_" + std::to_string(myRank)
                                + ".csv");

    for(auto val: data) {
        benchmarkFile << std::to_string(val) << ", ";
    }
    benchmarkFile << std::endl;
    benchmarkFile.close();
}

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
                                { (char *)"gatherCompositedVDIs", (char *)"(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;IIIIJJIJ)V", (void *)&gatherCompositedVDIs },
                                { (char *)"compositeImages", (char *)"(Ljava/nio/ByteBuffer;II[FJ)V", (void *) &compositeImages },
                                {(char *)"reduceAcrossPEs", (char *)"(D)D", (void *)&reduce},

    };

    int ret = jvmData.env->RegisterNatives(jvmData.clazz, methods, 5);
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

std::vector<std::vector<float>> proc_positions;

void setCentroids(std::vector<std::vector<float>> ptr) {
    proc_positions = ptr;
}


void distributeVDIs(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer) {
#if VERBOSE
    std::cout<<"In distribute VDIs function. Comm size is "<<commSize<<std::endl;
#endif

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

#if PROFILING
    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();
    begin_whole_compositing = std::chrono::high_resolution_clock::now();
#endif

#if VERBOSE
    std::cout<<"Starting all to all"<<std::endl;
#endif

    MPI_Alltoall(ptrCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, recvBufCol, windowHeight * windowWidth * numSupersegments * 4 * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);

#if VERBOSE
    std::cout<<"Finished color all to all"<<std::endl;
#endif

    MPI_Alltoall(ptrDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, recvBufDepth, windowHeight * windowWidth * numSupersegments * 4 * 2 / commSize, MPI_BYTE, MPI_COMM_WORLD);

#if VERBOSE
    printf("Finished both alltoalls\n");
#endif

#if PROFILING
    {
        end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        double local_alltoall = (elapsed.count()) * 1e-9;

        double global_sum;

        MPI_Reduce(&local_alltoall, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        double global_alltoall = global_sum / commSize;

        if (num_alltoall > warm_up_iterations) {
            total_alltoall += global_alltoall;

            distributeTimes.push_back((float)local_alltoall);
            globalDistributeTimes.push_back((float)global_alltoall);
        }

        num_alltoall++;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if VERBOSE
        std::cout << "Distribute time (full-res) at process " << rank << " was " << local_alltoall << std::endl;

        if(rank == 0) {
            std::cout << "global alltoall time: " << global_alltoall << std::endl;
        }
#endif

        if (((num_alltoall % 50) == 0) && (rank == 0)) {
            int iterations = num_alltoall - warm_up_iterations;
            double average_alltoall = total_alltoall / (double) iterations;
            std::cout << "Number of alltoalls: " << num_alltoall << " average alltoall time so far: "
                      << average_alltoall << std::endl;
        }

    }
#endif

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "uploadForCompositing", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)V");

    jobject bbCol = e->NewDirectByteBuffer(recvBufCol, sizePerProcess * commSize * 4);

    jobject bbDepth;

    bbDepth = e->NewDirectByteBuffer( recvBufDepth, sizePerProcess * commSize * 2);

    if(e->ExceptionOccurred()) {
        e->ExceptionDescribe();
        e->ExceptionClear();
    }

#if VERBOSE
    std::cout<<"Finished distributing the VDIs. Calling the Composite method now!"<<std::endl;
#endif

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

#if VERBOSE
    std::cout<<"Performing distribution of " << purpose <<std::endl;
#endif

    MPI_Alltoall(counts, 1, MPI_INT, countsRecv, 1, MPI_INT, MPI_COMM_WORLD);

    //set up the AllToAllv
    int displacementSendSum = 0;
    int displacementSend[commSize];

    int displacementRecvSum = 0;
    int displacementRecv[commSize];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

void compositeImages(JNIEnv *e, jobject clazzObject, jobject subImage, jint myRank, jint commSize, jfloatArray camPos, jlong imagePointer) {
#if VERBOSE
    std::cout<<"In image compositing function. Comm size is "<<commSize<<std::endl;
    IceTInt global_viewport[4];
    icetGetIntegerv(ICET_GLOBAL_VIEWPORT, global_viewport);

    std::cout<<"The global viewport is: width: "<<global_viewport[2] << " height: " << global_viewport[3] <<std::endl;
#endif

    IceTFloat background_color[4] = { 0.0, 0.0, 0.0, 0.0 };

    void *imageBuffer = e->GetDirectBufferAddress(subImage);

    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetSetDepthFormat(ICET_IMAGE_DEPTH_NONE);

    icetStrategy(ICET_STRATEGY_SEQUENTIAL);
//    icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_RADIXKR)

    IceTImage image;

    icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
    icetCompositeMode(ICET_COMPOSITE_MODE_BLEND);
    icetEnable(ICET_ORDERED_COMPOSITE);

    int order[commSize];

    float * cam = e->GetFloatArrayElements(camPos, NULL);
    std::vector<float> distances;
    std::vector<int> procs;

    for(int i = 0; i < commSize; i++) {
        float distance = std::sqrt(
                (proc_positions[i][0] - cam[0])*(proc_positions[i][0] - cam[0]) +
                (proc_positions[i][1] - cam[1])*(proc_positions[i][1] - cam[1]) +
                (proc_positions[i][2] - cam[2])*(proc_positions[i][2] - cam[2])
                );
        distances.push_back(distance);

        procs.push_back(i);
    }

    std::cout << "Cam pos: " << cam[0] << " " << cam[1] << " " << cam[2] << std::endl;

    if(myRank == 0) {
        for(int k = 0; k < commSize; k++) {
            std::cout << "Proc: " << k << " has centroid: " << proc_positions[k][0] << " " << proc_positions[k][1] << " " << proc_positions[k][2] << std::endl;
            std::cout << "Distance of proc " << k << ": " << distances[k] << std::endl;
        }
    }

    std::sort(procs.begin(), procs.end(), [&](int i, int j){return distances[i] < distances[j];});

    if(myRank == 0) {
        std::cout << "Order is: " << std::endl;
    }
    for(int i = 0; i < commSize; i++) {
        order[i] = procs[i];
        if(myRank == 0) {
            std::cout << "At pos: " << i << " proc: " << order[i] << std::endl;
        }
    }

    icetCompositeOrder(order);

    image = icetCompositeImage(
            imageBuffer,
            NULL,
            NULL,
            NULL,
            NULL,
            background_color
            );


    if(myRank == 0) {
        const char *color_buffer = (char *)icetImageGetColorcui(image);

        IceTSizeType width;
        IceTSizeType height;
        width = icetImageGetWidth(image);
        height = icetImageGetHeight(image);

#if VERBOSE
        std::cout << "Composited the image with dimensions: " << width << " " << height << std::endl;
#endif

        std::string dataset = datasetName;

        dataset += "_" + std::to_string(commSize) + "_" + std::to_string(myRank);

        std::string basePath = "/home/aryaman/TestingData/";

        if ((count % 10) == 0) {

            std::cout << "Writing the composited image " << count << " now" << std::endl;

            std::string filename = basePath + dataset + "FinalImage_" + std::to_string(count) + ".raw";

            std::ofstream b_stream(filename.c_str(),
                                   std::fstream::out | std::fstream::binary);
            if (b_stream) {
                b_stream.write(static_cast<const char *>(color_buffer),
                               windowHeight * windowWidth * 4);

                if (b_stream.good()) {
                    std::cout << "Writing was successful" << std::endl;
                }
            }
        }
        count++;

        jclass clazz = e->GetObjectClass(clazzObject);
        jmethodID displayMethod = e->GetMethodID(clazz, "displayComposited", "(Ljava/nio/ByteBuffer;)V");

        jobject bbCcomposited = e->NewDirectByteBuffer((void *)color_buffer, windowHeight * windowWidth * 4);
        if(e->ExceptionOccurred()) {
            e->ExceptionDescribe();
            e->ExceptionClear();
        }

        e->CallVoidMethod(clazzObject, displayMethod, bbCcomposited);
        if(e->ExceptionOccurred()) {
            e->ExceptionDescribe();
            e->ExceptionClear();
        }
    }
}

void distributeDenseVDIs(JNIEnv *e, jobject clazzObject, jobject colorVDI, jobject depthVDI, jobject prefixSums, jintArray supersegmentCounts, jint commSize, jlong colPointer, jlong depthPointer, jlong prefixPointer, jlong mpiPointer) {
#if VERBOSE
    std::cout<<"In distribute dense VDIs function. Comm size is "<<commSize<<std::endl;
#endif

    int *supsegCounts = e->GetIntArrayElements(supersegmentCounts, NULL);

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

#if PROFILING
    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();
    begin_whole_compositing = std::chrono::high_resolution_clock::now();
#endif

    int totalRecvdColor = distributeVariable(colorCounts, colorCountsRecv, ptrCol, recvBufCol, commSize, "color");
    int totalRecvdDepth = distributeVariable(depthCounts, depthCountsRecv, ptrDepth, recvBufDepth, commSize, "depth");

#if VERBOSE
    std::cout << "total bytes recvd: color: " << totalRecvdColor << " depth: " << totalRecvdDepth << std::endl;
#endif

    void * recvBufPrefix = reinterpret_cast<void *>(prefixPointer);
    void *ptrPrefix = e->GetDirectBufferAddress(prefixSums);

    MPI_Alltoall(ptrPrefix, windowWidth * windowHeight * 4 / commSize, MPI_BYTE, recvBufPrefix, windowWidth * windowHeight * 4 / commSize, MPI_BYTE, MPI_COMM_WORLD);

#if PROFILING
    {
        end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        double local_alltoall = (elapsed.count()) * 1e-9;

        double global_sum;

        MPI_Reduce(&local_alltoall, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        double global_alltoall = global_sum / commSize;

        if (num_alltoall > warm_up_iterations) {
            total_alltoall += global_alltoall;

            distributeTimes.push_back((float)local_alltoall);
            globalDistributeTimes.push_back((float)global_alltoall);
        }

        num_alltoall++;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if VERBOSE
        std::cout << "Distribute time (dense) at process " << rank << " was " << local_alltoall << std::endl;

        if(rank == 0) {
            std::cout << "global alltoall time: " << global_alltoall << std::endl;
        }
#endif

        if (((num_alltoall % 50) == 0) && (rank == 0)) {
            int iterations = num_alltoall - warm_up_iterations;
            double average_alltoall = total_alltoall / (double) iterations;
            std::cout << "Number of alltoalls: " << num_alltoall << " average alltoall time so far: "
                      << average_alltoall << std::endl;
        }

    }
#endif

#if VERBOSE
    printf("Finished both alltoalls for the dense VDIs\n");
#endif

    jclass clazz = e->GetObjectClass(clazzObject);
    jmethodID compositeMethod = e->GetMethodID(clazz, "uploadForCompositingDense", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;[I[I)V");

    int supsegsRecvd = totalRecvdColor / (4 * 4);

#if PROFILING
    if(num_alltoall % 50 == 0) {
        long global_sum;
        long local = (long)supsegsRecvd;

        numSupsegsGenerated.push_back(local);

        MPI_Reduce(&local, &global_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        long global_avg = global_sum/commSize;

        globalNumSupsegsGenerated.push_back(global_avg);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(rank == 0) {
            std::cout << "The average number of supersegments generated per PE " << global_avg << std::endl;
        }

        std::cout << "Number of supersegments received by this process: " << supsegsRecvd << std::endl;
    }
#endif

    long supsegsInBuffer = 512 * 512 * (std::max((long)ceil((double)supsegsRecvd / (512.0*512.0)), 2L));

#if VERBOSE
    std::cout << "The number of supsegs recvd: " << supsegsRecvd << " and stored: " << supsegsInBuffer << std::endl;
#endif

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

#if VERBOSE
    std::cout<<"Finished distributing the VDIs. Calling the dense Composite method now!"<<std::endl;
#endif

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

void gatherCompositedVDIs(JNIEnv *e, jobject clazzObject, jobject compositedVDIColor, jobject compositedVDIDepth, jint compositedVDILen, jint root, jint myRank, jint commSize, jlong colPointer, jlong depthPointer, jint vo, jlong mpiPointer) {

#if VERBOSE
    std::cout<<"In Gather function " <<std::endl;
#endif

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

#if PROFILING
    MPI_Barrier(MPI_COMM_WORLD);
    begin = std::chrono::high_resolution_clock::now();
#endif

    MPI_Gather(ptrCol, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, gather_recv_color, windowWidth * windowHeight * numOutputSupsegs * 4 * 4 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);

    MPI_Gather(ptrDepth, windowWidth  * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE,  gather_recv_depth, windowWidth * windowHeight * numOutputSupsegs * 4 * 2 / commSize, MPI_BYTE, root, MPI_COMM_WORLD);

#if PROFILING
    end = std::chrono::high_resolution_clock::now();
    end_whole_compositing = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    double local_gather = (elapsed.count()) * 1e-9;

#if VERBOSE
    std::cout << "Gather time at process " << myRank << " was " << local_gather << std::endl;
#endif

    double global_sum;

    MPI_Reduce(&local_gather, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_gather = global_sum / commSize;

    auto elapsed_whole_compositing = std::chrono::duration_cast<std::chrono::nanoseconds>(end_whole_compositing - begin_whole_compositing);

    double local_whole_compositing = (elapsed_whole_compositing.count()) * 1e-9;

#if VERBOSE
    std::cout << "Whole compositing time at process " << myRank << " was " << local_whole_compositing << std::endl;
#endif

    MPI_Reduce(&local_whole_compositing, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_whole_compositing = global_sum / commSize;

#if VERBOSE
    if(myRank == 0) {
        std::cout<<"Global gather: " << global_gather <<std::endl;
        std::cout<<"Global whole compositing: " << global_whole_compositing <<std::endl;
    }
#endif

    if(num_gather > warm_up_iterations) {
        total_gather += global_gather;
        total_whole_compositing += global_whole_compositing;

        gatherTimes.push_back((float)local_gather);
        globalGatherTimes.push_back((float)global_gather);

        wholeCompositeTimes.push_back((float)(local_whole_compositing));
        globalWholeCompositeTimes.push_back((float)global_whole_compositing);
    }

    num_gather++;
    if(((num_gather % 50) == 0) && (myRank == 0)) {
        int iterations = num_gather - warm_up_iterations;
        double average_gather = total_gather / (double)iterations;
        double average_whole_compositing = total_whole_compositing / (double)iterations;
        std::cout<< "Number of gathers: " << num_gather << " average gather time so far: " << average_gather
                << " average whole compositing time so far: " << average_whole_compositing <<std::endl;
    }
#endif

    end_whole_vdi = std::chrono::high_resolution_clock::now();

    auto elapsed_overall = std::chrono::duration_cast<std::chrono::nanoseconds>(end_whole_vdi - begin_whole_vdi);

    double local_overall = elapsed_overall.count() * 1e-9;

#if VERBOSE
    std::cout << "Whole VDI generation time at process " << myRank << " was " << local_overall << std::endl;
#endif

    double global_overall_sum = 0;
    MPI_Reduce(&local_overall, &global_overall_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_overall = global_overall_sum / commSize;

#if VERBOSE
    if(myRank == 0) {
        std::cout<<"Global overall time: " << global_overall <<std::endl;
    }
#endif

    if(num_whole_vdi > warm_up_iterations) {
        total_whole_vdi += global_overall;

        wholeVDITimes.push_back((float)local_overall);
        globalWholeVDITimes.push_back((float)global_overall);
    }

    num_whole_vdi++;

    if(((num_whole_vdi % 50)==0) && (myRank == 0)) {
        int iterations = num_whole_vdi - warm_up_iterations;
        double average_overall = total_whole_vdi / (double) iterations;

        std::cout<< "Number of VDIs generated: " << num_whole_vdi << " average time so far: " << average_overall << std::endl;
    }

    if((num_whole_vdi - warm_up_iterations - 1) == total_iterations) {
        //the benchmark is complete

#if PROFILING
        writeBenchmarkFile("distribute_full", distributeTimes, commSize, myRank);
        writeBenchmarkFile("gather_full", gatherTimes, commSize, myRank);
        writeBenchmarkFile("whole_composite_full", wholeCompositeTimes, commSize, myRank);
//        writeBenchmarkFile("num_supsegs", numSupsegsGenerated, commSize, myRank);

        if(myRank == 0) {
            writeBenchmarkFile("global_distr_full", globalDistributeTimes, commSize, myRank);
            writeBenchmarkFile("global_gather_full", globalGatherTimes, commSize, myRank);
            writeBenchmarkFile("global_whole_comp_full", globalWholeCompositeTimes, commSize, myRank);
//            writeBenchmarkFile("global_num_supsegs", globalNumSupsegsGenerated, commSize, myRank);
        }
#else //writing whole VDI timings only if not profiling
        writeBenchmarkFile("whole_vdi", wholeVDITimes, commSize, myRank);

        if(myRank == 0) {
            writeBenchmarkFile("global_whole_vdi", globalWholeVDITimes, commSize, myRank);
        }
#endif
        std::cout<< "The benchmark files have been written"<<std::endl;

        //benchmark complete so exit
        std::exit(0);
    }

    std::string dataset = datasetName;

    dataset += "_" + std::to_string(commSize) + "_" + std::to_string(myRank);

    std::string basePath = "/home/aryaman/TestingData/";

    if(myRank == 0) {
//        //send or store the VDI

        if(!benchmarking && (count % 2 == 0)) {

            std::cout<<"Writing the final gathered VDI now"<<std::endl;

            std::string filename = basePath + dataset + "FinalVDI_" + std::to_string(windowWidth) + "_" + std::to_string(windowHeight) + "_" + std::to_string(numOutputSupsegs)
                                   + "_" + std::to_string(vo) + "_4_ndc";

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
            if(count >= 10) {
                std::exit(1);
            }
        }
    }
    count++;
    begin_whole_vdi = std::chrono::high_resolution_clock::now();
}

double reduce(JNIEnv *e, jobject clazzObject, jdouble value) {

    double local_val = value;
    double global_min;

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Reduce(&local_val, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        return global_min;
    } else {
        return 0.0;
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