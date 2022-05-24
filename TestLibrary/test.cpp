#include <iostream>
#include <dirent.h>
#include <fstream>
#include "TestLiibrary.hpp"

#define VERBOSE true
#define USE_VULKAN true

enum vis_type
{
    particles = 0x0,
    grid = 0x1,
};

bool isMain = false;

vis_type getVisType() {
    return vis_type::grid;
}

JVMData func(int rank) {
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
            "-Dorg.slf4j.simpleLogger.defaultLogLevel=info";
    #else
    options[2].optionString = (char *)
            "-Dscenery.LogLevel=warn";
    #endif

    options[3].optionString = (char *)
            "-Dscenery.Headless=false";

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

    jvmData.clazz = reinterpret_cast<jclass>(jvmData.env->NewGlobalRef(localClass));
    jvmData.obj = reinterpret_cast<jobject>(jvmData.env->NewGlobalRef(localObj));

    jvmData.env->DeleteLocalRef(localClass);
    jvmData.env->DeleteLocalRef(localObj);

    #if VERBOSE
    std::cout << "Global references have been created and local ones deleted" << std::endl;
    #endif

    return jvmData;
}

void setPixelToWorld(JVMData jvmData , float pixelToWorld) {
    jfieldID pixelToWorldField = jvmData.env->GetFieldID(jvmData.clazz, "pixelToWorld", "F");
    jvmData.env->SetFloatField(jvmData.obj, pixelToWorldField, 3.84 / 832.f);
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
