#ifndef DISTRIBUTEDVIS_VDIPARAMS_HPP
#define DISTRIBUTEDVIS_VDIPARAMS_HPP

#include <iostream>
#include <string.h>

const int windowWidth = 1280;
const int windowHeight = 720;
const int numSupersegments = 20;
const int numOutputSupsegs = 20;

const int colorSize = windowWidth * windowHeight * numSupersegments * 4 * 4;
const int depthSize = windowWidth * windowHeight * numSupersegments * 4 * 2;

const std::string datasetName = "Kingsnake";
const bool dataset16bit = false;
const bool benchmarking = false;

#endif //DISTRIBUTEDVIS_VDIPARAMS_HPP
