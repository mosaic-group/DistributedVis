#include <iostream>
#include "TestLiibrary.hpp"
#include <mpi.h>
#include <thread>
#include <fstream>
#include <vector>

void tokenize(std::string const &str, const char delim,
              std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

int * getVolumeDims(const std::string& path) {
    std::ifstream infoFile (path + "/stacks.info", std::ios::in);
    if(!infoFile.is_open()) {
        std::cerr << "Could not find the stacks.info file!" << std::endl;
    }

    int * volume_dimensions = new int[3];

    std::string line;
    std::getline(infoFile, line);

    std::vector<std::string> tokens;
    tokenize(line, ',', tokens);

    volume_dimensions[0] = stoi(tokens.at(0));
    volume_dimensions[1] = stoi(tokens.at(1));
    volume_dimensions[2] = stoi(tokens.at(2));

    std::cout << "Dimensions: 0:" << volume_dimensions[0] << " 1: " << volume_dimensions[1] << " 2: " << volume_dimensions[2] << std::endl;

    return volume_dimensions;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    MPI_Init(NULL, NULL);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    JVMData jvmData = func(rank);

    std::thread render(&doRender, jvmData);

    int * volume_dimensions = getVolumeDims("/home/aryaman/Datasets/Volume/Stagbeetle/Part1");
    float pixelToWorld = 3.84f / (float)volume_dimensions[0]; //empirical

    setPixelToWorld(jvmData, pixelToWorld);

    int num_volumes = 6;

    long volume_sizes[num_volumes];

    int prev_slices = 0;

    std::ifstream volumeFile ("/home/aryaman/Datasets/Volume/Stagbeetle/Part1/stagbeetle832x832x494.raw", std::ios::in | std::ios::binary);
    if(!volumeFile.is_open()) {
        std::cerr<< "Could not open the volume file! " << std::endl;
    }

    int chunks_remaining = num_volumes;
    int slices_remaining = volume_dimensions[2];

    for(int i = 0; i < num_volumes; i++) {
        int chunk_dimensions[3];
        chunk_dimensions[0] = volume_dimensions[0];
        chunk_dimensions[1] = volume_dimensions[1];
        chunk_dimensions[2] = slices_remaining / chunks_remaining;
        slices_remaining -= chunk_dimensions[2];
        chunks_remaining--;

        std::cout << "Chunk " << i << "has dimensions: " << chunk_dimensions[0] << " " << chunk_dimensions[1] << " " << chunk_dimensions[2] << std::endl;

        volume_sizes[i] = chunk_dimensions[0] * chunk_dimensions[1] * chunk_dimensions[2] * 2;

        float pos [3];
        pos[0] = 0.f;
        pos[1] = 0.f;
        pos[2] = 1.f * (float)prev_slices * pixelToWorld;

        createVolume(jvmData, i, chunk_dimensions, pos);
        char * buffer = new char[volume_sizes[i]];
        volumeFile.read (buffer, volume_sizes[i]);
        updateVolume(jvmData, i, buffer, volume_sizes[i]);

        prev_slices += chunk_dimensions[2];
    }

    std::cout<<"Back after calling do Render" <<std::endl;

    render.join();

    MPI_Finalize();

    return 0;
}
