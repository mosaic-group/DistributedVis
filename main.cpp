#include <iostream>
#include "TestLiibrary.hpp"
#include <mpi.h>
#include <thread>
#include <fstream>
#include <vector>
#include <zconf.h>
#include <cmath>

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

    std::string dataset = "Kingsnake";
    const bool is16bit = false;
    bool generateVDIs = true;
    bool isCluster = false;

    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);

    std::cout << "Got MPI thread level: " << provided << std::endl;
//
//    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

//    if(rank == 1) {
//        sleep(10000);
//    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                         MPI_INFO_NULL, &nodeComm );

    int node_rank;
    MPI_Comm_rank(nodeComm,&node_rank);

    if(!isCluster) {
        node_rank = 0;
    }

    JVMData jvmData = setupJVM(isCluster);

    setPointerAddresses(jvmData, MPI_COMM_WORLD);
    setVDIGeneration(jvmData, generateVDIs);

    if(true) {


        int * volume_dimensions = getVolumeDims("/home/aryaman/Datasets/Volume/" + dataset + "/Part1");
        float pixelToWorld = 3.84f / (float)volume_dimensions[0]; //empirical
        setDatasetParams(jvmData, dataset, pixelToWorld);
        setMPIParams(jvmData, rank, node_rank, num_processes);

        std::thread render(&doRender, jvmData);

        int slices_per_process[num_processes];
        int start_slice[num_processes];

        int processes_remaining = num_processes;
        int slices_remaining = volume_dimensions[2];

        int prev_slices = 0;

        for(int i = 0; i < num_processes; i++) {
            slices_per_process[i] = slices_remaining / processes_remaining;
            start_slice[i] = prev_slices;
            slices_remaining -= slices_per_process[i];
            processes_remaining--;
            std::cout << "Process " << i << "will handle: " << slices_per_process[i] << " slices." << std::endl;
            prev_slices += slices_per_process[i];
        }

        long int volume_size = (long int)volume_dimensions[0] * (long int)volume_dimensions[1] * (long int)slices_per_process[rank] * (is16bit? 2: 1);

        int num_volumes = ceil((double)volume_size / 2000000000.0); // Divide by 2 GB. each process will handle num_volumes volumes

        std::cout<<"Volume size is: " << volume_size << std::endl;
        std::cout<<"Num volumes is: " << num_volumes << std::endl;

        long volume_sizes[num_volumes]; // this array will store the size (in Bytes) of each volume in the scene of a given visualization process

        prev_slices = start_slice[rank];

        std::ifstream volumeFile ("/home/aryaman/Datasets/Volume/" + dataset + "/Part1/" + dataset + ".raw", std::ios::in | std::ios::binary);
        if(!volumeFile.is_open()) {
            std::cerr<< "Could not open the volume file! " << std::endl;
        }

        volumeFile.seekg(prev_slices * volume_dimensions[0] * volume_dimensions[1] * (is16bit? 2: 1));

        int chunks_remaining = num_volumes;
        slices_remaining = slices_per_process[rank];

        for(int i = 0; i < num_volumes; i++) {
            int chunk_dimensions[3];
            chunk_dimensions[0] = volume_dimensions[0];
            chunk_dimensions[1] = volume_dimensions[1];
            chunk_dimensions[2] = slices_remaining / chunks_remaining;
            slices_remaining -= chunk_dimensions[2];
            chunks_remaining--;

            std::cout << "Chunk " << i << " has dimensions: " << chunk_dimensions[0] << " " << chunk_dimensions[1] << " " << chunk_dimensions[2] << std::endl;

            volume_sizes[i] = chunk_dimensions[0] * chunk_dimensions[1] * chunk_dimensions[2] * (is16bit? 2: 1);

            float pos [3];
            pos[0] = 0.f;
            pos[1] = 0.f;
            pos[2] = 1.f * (float)prev_slices * pixelToWorld;

            createVolume(jvmData, i, chunk_dimensions, pos, is16bit);
            char * buffer = new char[volume_sizes[i]];
            volumeFile.read (buffer, volume_sizes[i]);
            updateVolume(jvmData, i, buffer, volume_sizes[i]);

            prev_slices += chunk_dimensions[2];
        }

        setRendererConfigured(jvmData);

        std::cout<<"Back after calling do Render" <<std::endl;

        sleep(60);
        std::cout<<"Calling stopRendering!" <<std::endl;
        stopRendering(jvmData);

        render.join();
    }


    MPI_Finalize();

    return 0;
}
