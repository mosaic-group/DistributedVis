#include "ManageRendering.hpp"
#include <dirent.h>

JVMData setupJVM(bool isCluster, std::string className, int rank) {
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

    className = "graphics/scenery/insitu/" + className;

    JavaVMInitArgs vm_args;                        // Initialization arguments
    auto *options = new JavaVMOption[7];   // JVM invocation options
    options[0].optionString = (char *)classPath.c_str();

    options[1].optionString = (char *)
            "-Dscenery.VulkanRenderer.EnableValidations=false";

    options[2].optionString = (char *)
            "-Dorg.lwjgl.system.stackSize=1000";

    options[3].optionString = (char *)
            "-Dscenery.Headless=false";

    options[4].optionString = (char *)
            "-Dscenery.LogLevel=info";

//    options[4].optionString = (char *)
//                                      "-Dscenery.LogLevel=debug";

    if(isCluster) {
        options[5].optionString = (char *)
                (std::string("-Dorg.lwjgl.system.SharedLibraryExtractPath=/beegfs/ws/1/argupta-vdi_generation/lw_files/rank") + std::to_string(rank) + "/").c_str();
        options[6].optionString = (char *)
                (std::string("-Dorg.lwjgl.librarypath=/beegfs/ws/1/argupta-vdi_generation/lw_files/rank") + std::to_string(rank) + "/").c_str();
    } else {
        options[5].optionString = (char *)
                "-Dorg.lwjgl.system.SharedLibraryExtractPath=/tmp/";
        options[6].optionString = (char *)
                "-Dorg.lwjgl.librarypath=/tmp/";
    }


    vm_args.version = JNI_VERSION_1_6;
    vm_args.nOptions = 7;
    vm_args.options = options;
    vm_args.ignoreUnrecognized = false;

    jint rc = JNI_CreateJavaVM(&jvmData.jvm, (void **) &jvmData.env, &vm_args);

    std::cout<<"Hello world"<<std::endl;

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

    std::cout << "Class found " << className << std::endl;

    jmethodID constructor = jvmData.env->GetMethodID(localClass, "<init>", "()V");  // find constructor
    if (constructor == nullptr) {
        if( jvmData.env->ExceptionOccurred() ) {
            jvmData.env->ExceptionDescribe();
        } else {
            std::cerr << "ERROR: constructor not found !" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << "Constructor found for " << className << std::endl;

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

    std::cout << "Object of class has been constructed" << std::endl;

    jfieldID clusterField = jvmData.env->GetFieldID(localClass, "isCluster", "Z");
    jvmData.env->SetBooleanField(localObj, clusterField, isCluster);

    if (jvmData.env->ExceptionOccurred()) {
        jvmData.env->ExceptionDescribe();
    }

    jvmData.clazz = reinterpret_cast<jclass>(jvmData.env->NewGlobalRef(localClass));
    jvmData.obj = reinterpret_cast<jobject>(jvmData.env->NewGlobalRef(localObj));

    jvmData.env->DeleteLocalRef(localClass);
    jvmData.env->DeleteLocalRef(localObj);

    return jvmData;
}

void setupICET(int windowWidth, int windowHeight) {
    IceTCommunicator comm;

    comm = icetCreateMPICommunicator(MPI_COMM_WORLD);

    IceTBitField diag_level = ICET_DIAG_ALL_NODES | ICET_DIAG_WARNINGS;

    icetCreateContext(comm);
    icetDiagnostics(diag_level);

    icetAddTile(0, 0, windowWidth, windowHeight, 0);
}

void doRender(JVMData jvmData) {

    std::cout << "In doRender function!" << std::endl;

    JNIEnv *env;
//    jvmData.jvm->GetEnv((void **)&env, JNI_VERSION_1_6);
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    std::cout << "Looking for main function!" << std::endl;

    jmethodID mainMethod = env->GetMethodID(jvmData.clazz, "main", "()V");
    if(mainMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            std::cout << "ERROR in finding main!" << std::endl;
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function main not found!" << std::endl;
//            std::exit(EXIT_FAILURE);
        }
    }
    env->CallVoidMethod(jvmData.obj, mainMethod);
    if(env->ExceptionOccurred()) {
        std::cout << "ERROR in calling main!" << std::endl;
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}

void setSceneConfigured(JVMData jvmData) {
    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jboolean ready = true;
    jfieldID readyField = env->GetFieldID(jvmData.clazz, "rendererConfigured", "Z");
    env->SetBooleanField(jvmData.obj, readyField, ready);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

void waitRendererConfigured(JVMData jvmData) {

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jmethodID rendererMethod = env->GetMethodID(jvmData.clazz, "rendererReady", "()V");
    if(rendererMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function rendererReady not found!";
        }
    }

    env->CallVoidMethod(jvmData.obj, rendererMethod);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}

void stopRendering(JVMData jvmData) {

    JNIEnv *env;
    jvmData.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), NULL);

    jmethodID terminateMethod = env->GetMethodID(jvmData.clazz, "stopRendering", "()V");
    if(terminateMethod == nullptr) {
        if (env->ExceptionOccurred()) {
            env->ExceptionDescribe();
        } else {
            std::cout << "ERROR: function stopRendering not found!";
        }
    } else {
        std::cout << "stopRendering function successfully found!";
    }

    env->CallVoidMethod(jvmData.obj, terminateMethod);
    if(env->ExceptionOccurred()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    jvmData.jvm->DetachCurrentThread();
}