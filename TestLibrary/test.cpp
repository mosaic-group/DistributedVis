#include <iostream>
#include <dirent.h>
#include "TestLiibrary.hpp"
#include <mpi.h>
#include <string.h>
#include "VDIParams.hpp"

#define USE_VULKAN true

enum vis_type
{
    particles = 0x0,
    grid = 0x1,
};

vis_type getVisType() {
    return vis_type::grid;
}

void setBenchmarking(JVMData jvmData) {
    jfieldID vdiField = jvmData.env->GetFieldID(jvmData.clazz, "benchmarking", "Z");
    jvmData.env->SetBooleanField(jvmData.obj, vdiField, benchmarking);
}

void setDatasetParams(JVMData jvmData, std::string dataset, float pixelToWorld, int dimensions[]) {

    jstring jdataset = jvmData.env->NewStringUTF(dataset.c_str());
    jfieldID datasetField = jvmData.env->GetFieldID(jvmData.clazz, "dataset", "Ljava/lang/String;");
    jvmData.env->SetObjectField(jvmData.obj, datasetField, jdataset);

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

    setBenchmarking(jvmData);

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