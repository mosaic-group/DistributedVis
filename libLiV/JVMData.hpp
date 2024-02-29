#ifndef DISTRIBUTEDVIS_JVMDATA_HPP
#define DISTRIBUTEDVIS_JVMDATA_HPP

#include <jni.h>

struct JVMData {
    JavaVM *jvm;
    jclass clazz;
    jobject obj;
    JNIEnv *env;
};

#endif //DISTRIBUTEDVIS_JVMDATA_HPP
