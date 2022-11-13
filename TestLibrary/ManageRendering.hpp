//
// Created by aryaman on 11/10/22.
//

#ifndef DISTRIBUTEDVIS_MANAGERENDERING_HPP
#define DISTRIBUTEDVIS_MANAGERENDERING_HPP
#include "JVMData.hpp"
#include <iostream>


JVMData setupJVM(bool isCluster, std::string className);
void doRender(JVMData jvmData);
void setSceneConfigured(JVMData jvmData);
void waitRendererConfigured(JVMData jvmData);
void stopRendering(JVMData jvmData);

#endif //DISTRIBUTEDVIS_MANAGERENDERING_HPP
