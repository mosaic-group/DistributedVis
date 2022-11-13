//
// Created by aryaman on 11/10/22.
//

#ifndef DISTRIBUTEDVIS_MPINATIVES_HPP
#define DISTRIBUTEDVIS_MPINATIVES_HPP

#include "JVMData.hpp"

void distributeVDIs(JNIEnv *e, jobject clazzObject, jobject subVDICol, jobject subVDIDepth, jint sizePerProcess, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer);
void gatherCompositedVDIs(JNIEnv *e, jobject clazzObject, jobject compositedVDIColor, jobject compositedVDIDepth, jint compositedVDILen, jint root, jint myRank, jint commSize, jlong colPointer, jlong depthPointer, jlong mpiPointer);

void setPointerAddresses(JVMData jvmData, MPI_Comm renderComm);
void setMPIParams(JVMData jvmData , int rank, int node_rank, int commSize);
void registerNatives(JVMData jvmData);

#endif //DISTRIBUTEDVIS_MPINATIVES_HPP
