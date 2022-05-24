//
// Created by aryaman on 5/9/22.
//

#ifndef DISTRIBUTEDVIS_TESTLIIBRARY_HPP
#define DISTRIBUTEDVIS_TESTLIIBRARY_HPP

#include <jni.h>

struct JVMData {
    JavaVM *jvm;
    jclass clazz;
    jobject obj;
    JNIEnv *env;
};

JVMData func(int rank);
void setPixelToWorld(JVMData jvmData , float pixelToWorld);
void createVolume(JVMData jvmData, int volumeID, int dimensions[], float pos[]);
void updateVolume(JVMData jvmData, int volumeID, char * buffer, long buffer_size);
void doRender(JVMData jvmData);

#endif //DISTRIBUTEDVIS_TESTLIIBRARY_HPP
