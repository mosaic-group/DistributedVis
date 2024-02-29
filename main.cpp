#include <iostream>
#include "libLiV/libLiV.hpp"
#include "libLiV/MPINatives.hpp"
#include "libLiV/ManageRendering.hpp"
#include "libLiV/VDIParams.hpp"
#include <mpi.h>
#include <thread>
#include <fstream>
#include <vector>
#include <zconf.h>
#include <cmath>

enum decompositionTypes {
    plane,
    block
};

std::string datasetName;
bool dataset16bit;

std::string getEnvVar( std::string const & key )
{
    char * val = getenv( key.c_str() );
    return val == NULL ? std::string("") : std::string(val);
}

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
        std::cerr << "Could not find the stacks.info file! Path: " << path << std::endl;
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

int * getBlockNumbers(const std::string& path) {
    std::ifstream infoFile (path + "/block_div.info", std::ios::in);
    if(!infoFile.is_open()) {
        std::cerr << "Could not find the block numbers info file! Path: " << path << std::endl;
    }

    int * block_divs = new int[3];

    std::string line;
    std::getline(infoFile, line);

    std::vector<std::string> tokens;
    tokenize(line, ',', tokens);

    block_divs[0] = stoi(tokens.at(0));
    block_divs[1] = stoi(tokens.at(1));
    block_divs[2] = stoi(tokens.at(2));

    std::cout << "Block divs: 0:" << block_divs[0] << " 1: " << block_divs[1] << " 2: " << block_divs[2] << std::endl;

    return block_divs;
}

void slice2GB(JVMData jvmData, const int volume_dimensions[], int start_slice, const float pos_offset[], float pixelToWorld, std::string filename) {
    long int volume_size = (long int)volume_dimensions[0] * (long int)volume_dimensions[1] * (long int)volume_dimensions[2] * (dataset16bit? 2: 1);

    int num_volumes = ceil((double)volume_size / 2000000000.0); // Divide by 2 GB. each process will handle num_volumes volumes

    std::cout<<"Volume size is: " << volume_size << std::endl;
    std::cout<<"Num volumes is: " << num_volumes << std::endl;

    long int volume_sizes[num_volumes]; // this array will store the size (in Bytes) of each volume in the scene of a given visualization process

    int prev_slices = start_slice;

    std::cout<<"fetched volume from " << filename << std::endl;

    std::ifstream volumeFile (filename, std::ios::in | std::ios::binary);
    if(!volumeFile.is_open()) {
        std::cerr<< "Could not open the volume file! " << std::endl;
        std::exit(-1);
    }

    volumeFile.seekg((long int)prev_slices * (long int)volume_dimensions[0] * (long int)volume_dimensions[1] * (dataset16bit? 2: 1));

    int chunks_remaining = num_volumes;
    int slices_remaining = volume_dimensions[2];

    for(int i = 0; i < num_volumes; i++) {
        int chunk_dimensions[3];
        chunk_dimensions[0] = volume_dimensions[0];
        chunk_dimensions[1] = volume_dimensions[1];
        chunk_dimensions[2] = slices_remaining / chunks_remaining;
        slices_remaining -= chunk_dimensions[2];
        chunks_remaining--;

        std::cout << "Chunk " << i << " has dimensions: " << chunk_dimensions[0] << " " << chunk_dimensions[1] << " " << chunk_dimensions[2] << std::endl;

        volume_sizes[i] = (long int)chunk_dimensions[0] * (long int)chunk_dimensions[1] * (long int)chunk_dimensions[2] * (dataset16bit? 2: 1);

        float pos [3];
        pos[0] = pos_offset[0] + 0.f;
        pos[1] = pos_offset[1] + 0.f;
        pos[2] = pos_offset[2] + 1.f * (float)prev_slices * pixelToWorld;

        std::cout<< "volume position set to " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
        std::cout<< "volume size " << chunk_dimensions[0] << " " << chunk_dimensions[1] << " " << chunk_dimensions[2] << std::endl;

        createVolume(jvmData, i, chunk_dimensions, pos, dataset16bit);
        char * buffer = new char[volume_sizes[i]];
        volumeFile.read (buffer, volume_sizes[i]);
        updateVolume(jvmData, i, buffer, volume_sizes[i]);

        prev_slices += chunk_dimensions[2];
    }
}

int * getBlockID(int proc_rank, int num_cells_x, int num_cells_y) {
    int *id = new int[3];

    id[2] = proc_rank / (num_cells_x * num_cells_y);
    int remainder = proc_rank % (num_cells_x * num_cells_y);

    id[1] = remainder / num_cells_x;
    id[0] = remainder % num_cells_x;

    return id;
}

float * getBlockPos(int rank, const int * volume_dimensions, float pixelToWorld, int num_x, int num_y, int num_z) {
    //return the Front Bottom Left position for the block
    float * pos = new float[3];
    int * id = getBlockID(rank, num_x, num_y);

    int block_x = id[0];
    int block_y = id[1];
    int block_z = id[2];

    pos[0] = pixelToWorld * (float)(block_x * (int)(volume_dimensions[0] / num_x));
    pos[1] = -1 * pixelToWorld * (float)(block_y * (int)(volume_dimensions[1] / num_y));
    pos[2] = pixelToWorld * (float)(block_z * (int)(volume_dimensions[2] / num_z));
    return pos;
}

std::vector<std::vector<float>> computeCentroids(int num_processes, int * volume_dimensions, float pixelToWorld, int num_cells_x, int num_cells_y, int num_cells_z) {
    // for all ranks, find block ID, then block pos FBL and pos FTR, divide by two

    auto centroids = std::vector<std::vector<float>>(num_processes, std::vector<float>(3));

    for(int i = 0; i < num_processes; i++) {
        float * fbl = getBlockPos(i, volume_dimensions, pixelToWorld, num_cells_x, num_cells_y, num_cells_z);

        float ftr[3];

        ftr[0] = fbl[0] + pixelToWorld * (int)(volume_dimensions[0] / num_cells_x);
        ftr[1] = fbl[1] - pixelToWorld * (int)(volume_dimensions[1] / num_cells_y);
        ftr[2] = fbl[2] + pixelToWorld * (int)(volume_dimensions[2] / num_cells_z);

        for(int j = 0; j < 3; j++) {
            centroids[i][j] = (fbl[j] + ftr[j])/2.0f;
        }

        delete[] fbl;

    }

    return centroids;

}

void decomposeBlocks(JVMData jvmData, int num_processes, int volume_dimensions[], int rank, float pixelToWorld) {

    int * block_numbers = getBlockNumbers(getEnvVar("DATASET_PATH") + "/" + datasetName + "/Cubes" + std::to_string(num_processes));

    int num_x = block_numbers[0];
    int num_y = block_numbers[1];
    int num_z = block_numbers[2];

    float * pos_offset = getBlockPos(rank, volume_dimensions, pixelToWorld, num_x, num_y, num_z);

    std::vector<std::vector<float>> centroids(num_processes, std::vector<float>(3));
    centroids = computeCentroids(num_processes, volume_dimensions, pixelToWorld, num_x, num_y, num_z);

    setCentroids(centroids);

    std::string filepath = getEnvVar("DATASET_PATH") + "/" + datasetName + "/Cubes" + std::to_string(num_processes) + "/Part" + std::to_string(rank);

    std::cout << "Filepath: " << filepath << std::endl;

    int * block_dims = getVolumeDims(filepath);

    std::string volume_path = filepath + "/block.raw";

    std::cout << "pos offset: " << pos_offset[0] << ", " << pos_offset[1] << ", " << pos_offset[2] <<std::endl;

    slice2GB(jvmData, block_dims, 0, pos_offset, pixelToWorld, volume_path);

    std::cout << "Process " << rank << " has read and updated all volumes." << std::endl;
}

void decomposePlanes(JVMData jvmData, int num_processes, const int volume_dimensions[], int rank, float pixelToWorld) {
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

    int proc_vol_dims[] = {volume_dimensions[0], volume_dimensions[1], slices_per_process[rank]};
    std::cout << "Proc dims: " << proc_vol_dims[0] << " " << proc_vol_dims[1] << " " << proc_vol_dims[2] << std::endl;
    int proc_start = start_slice[rank];

    float pos_offset[] = {0.0f, 0.0f, 0.0f};

    slice2GB(jvmData, proc_vol_dims, proc_start, pos_offset, pixelToWorld, getEnvVar("DATASET_PATH") + "/" + datasetName + "/" + datasetName + ".raw");
}

void decomposeDomain(decompositionTypes type, JVMData jvmData, int num_processes, int volume_dimensions[], int rank, float pixelToWorld) {
    if(type == decompositionTypes::plane) {
        decomposePlanes(jvmData, num_processes, volume_dimensions, rank, pixelToWorld);
    } else if(type == decompositionTypes::block) {
        decomposeBlocks(jvmData, num_processes, volume_dimensions, rank, pixelToWorld);
    }
}


int main() {
    std::cout << "Hello, World!" << std::endl;

    datasetName = getEnvVar("DATASET_NAME");
    dataset16bit = getEnvVar("DATASET_16_BIT") == "true";

//    setDatasetProperties(datasetName, dataset16bit);

    std::string dataset = datasetName;
    const bool is16bit = dataset16bit;
    bool generateVDIs = false;
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

    JVMData jvmData = setupJVM(isCluster, "DistributedVolumes", rank);

    registerNatives(jvmData);

    setPointerAddresses(jvmData, MPI_COMM_WORLD);
    setVDIGeneration(jvmData, generateVDIs);

    if(!generateVDIs) {
        setupICET(windowWidth, windowHeight);
    }

    if(true) {
        int * volume_dimensions = getVolumeDims(getEnvVar("DATASET_PATH") + "/" + dataset);
        float pixelToWorld = 3.84f / (float)volume_dimensions[0]; //empirical
        setDatasetParams(jvmData, dataset, pixelToWorld, volume_dimensions);
        setMPIParams(jvmData, rank, node_rank, num_processes);

        std::thread render(&doRender, jvmData);

        decomposeDomain(decompositionTypes::block, jvmData, num_processes, volume_dimensions, rank, pixelToWorld);

        setSceneConfigured(jvmData);

        std::cout<<"Back after calling do Render" <<std::endl;

        sleep(1000);
        std::cout<<"Calling stopRendering!" <<std::endl;
        stopRendering(jvmData);

        render.join();
    }


    MPI_Finalize();

    return 0;
}
