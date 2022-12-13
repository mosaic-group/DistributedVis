//
// Created by aryaman on 11/10/22.
//

#ifndef DISTRIBUTEDVIS_MPINATIVES_HPP
#define DISTRIBUTEDVIS_MPINATIVES_HPP

#include <iostream>
#include "JVMData.hpp"

void distributeVDIs(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer);
void distributeVDIsForBenchmark(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer, jint rank, jint iteration);
void distributeVDIsWithVariableLength(JNIEnv *e, jobject clazzObject, jobject colorVDI, jobject depthVDI, jintArray colorLimits, jintArray depthLimits , jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer, jboolean isBenchmark = false, jint rank = 0, jint iteration = 0);
void gatherCompositedVDIs(JNIEnv *e, jobject clazzObject, jobject compositedVDIColor, jobject compositedVDIDepth, jint compositedVDILen, jint root, jint myRank, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer);
void setPointerAddresses(JVMData jvmData, MPI_Comm renderComm);
void setMPIParams(JVMData jvmData , int rank, int node_rank, int commSize);
void registerNatives(JVMData jvmData);
void setProgramSettings(JVMData jvmData, std::string dataset, bool withCompression, bool benchmarkValues);

#endif //DISTRIBUTEDVIS_MPINATIVES_HPP
