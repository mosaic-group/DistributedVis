#ifndef DISTRIBUTEDVIS_VDIPARAMS_HPP
#define DISTRIBUTEDVIS_VDIPARAMS_HPP

#include <iostream>
#include <string.h>

const int windowWidth = 1920;
const int windowHeight = 1080;
const int numSupersegments = 25;
const int numOutputSupsegs = 25;

const int colorSize = windowWidth * windowHeight * numSupersegments * 4 * 4;
const int depthSize = windowWidth * windowHeight * numSupersegments * 4 * 2;

const std::string datasetName = "Rotstrat";
const bool dataset16bit = true;
const bool benchmarking = true;

#endif //DISTRIBUTEDVIS_VDIPARAMS_HPP
