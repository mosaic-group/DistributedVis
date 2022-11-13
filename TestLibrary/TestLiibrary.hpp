//
// Created by aryaman on 5/9/22.
//

#ifndef DISTRIBUTEDVIS_TESTLIIBRARY_HPP
#define DISTRIBUTEDVIS_TESTLIIBRARY_HPP

#include "JVMData.hpp"
#include <mpi.h>

void setDatasetParams(JVMData jvmData, std::string dataset, float pixelToWorld, int dimensions[]);
void setVDIGeneration(JVMData jvmData , bool generateVDIs);
void createVolume(JVMData jvmData, int volumeID, int dimensions[], float pos[], bool is16bit);
void updateVolume(JVMData jvmData, int volumeID, char * buffer, long int buffer_size);
void setSceneConfigured(JVMData jvmData);

#endif //DISTRIBUTEDVIS_TESTLIIBRARY_HPP
