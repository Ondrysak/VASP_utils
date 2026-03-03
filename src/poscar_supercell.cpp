#include "poscar_supercell.h"

#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include "poscar_file.h"

bool readInput(int argc, char* argv[], std::string& inputFile, std::string& outputFile, bool& diagonal, double S[9]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            printHelp();
            return false;
        } else if (arg == "--input") {
            if (i + 1 >= argc)
                return false;
            inputFile = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc)
                return false;
            outputFile = argv[++i];
        } else if (arg == "--dim") {
            if (i + 3 >= argc) {
                std::cerr << "Error: --dim requires three integers\n";
                return false;
            }
            S[0] = std::stoi(argv[++i]);
            S[4] = std::stoi(argv[++i]);
            S[8] = std::stoi(argv[++i]);
            diagonal = true;
        } else {
            std::cerr << "Warning: unknown argument! Ignoring it!\n";
            printHelp();
        }
    }
    return true;
}

bool validateInput(const std::string& inputFile, const std::string& outputFile, double S[9]) {
    std::ifstream file(inputFile);
    if (!file) {
        std::cerr << "Error: cannot open file " << inputFile << "\n";
        return false;
    }
    std::ifstream file_2(outputFile);
    if (file_2) {
        std::cerr << "Warning: output file " << outputFile << " already exists!!! Will be overwritten!!!\n";
    }
    if (S[0] <= 0 || S[4] <= 0 || S[8] <= 0) {
        std::cerr << "Error: supercell dimentions must be positive integer!!!\n";
        return false;
    }

    return true;
}

void printHelp() {
    std::cerr << "Usage:\n"
                 "  poscar_supercell [options]\n\n"
                 "Options:\n"
                 "  --input      input POSCAR file name (default: POSCAR)\n"
                 "  --output     output POSCAR file name (default: POSCAR_nx_ny_nz)\n"
                 "  --dim        dimentions of supercell\n"
                 "  --help       show this help message\n\n"
                 "Example:\n"
                 "  poscar_supercell --input POSCARin --dim 2 2 2 \n";
}

int main(int argc, char* argv[]) {
    std::string inputFile{"POSCAR"};
    std::string outputFile{"POSCAR_"};
    bool diagonal{true};
    double S[9];

    // S matrix inicialization
    for (int k = 0; k < 9; ++k)
        S[k] = 0.0;

    S[0] = 1;
    S[4] = 1;
    S[8] = 1;

    if (!readInput(argc, argv, inputFile, outputFile, diagonal, S)) {
        return 1;
    }

    if (!validateInput(inputFile, outputFile, S)) {
        return 1;
    }
    if (outputFile == "POSCAR_") {
        outputFile += std::to_string((int)S[0]) + "_" + std::to_string((int)S[4]) + "_" + std::to_string((int)S[8]);
    }

    POSCAR poscar;

    if (!poscar.readPOSCAR(inputFile)) {
        std::cerr << "Error: reading POSCAR file " << inputFile << "\n";
        return 1;
    }
    poscar.toDirect();

    POSCAR poscar_supercell;
    poscar_supercell = poscar.makeSupercell(S);
    if (poscar_supercell.total_atoms == 0) {
        std::cerr << "Error: failed to make supercell!!!\n";
        return 1;
    }

    if (!poscar_supercell.writePOSCAR(outputFile)) {
        std::cerr << "Error: failed to write primitie cell POSCAR file to " << outputFile << "\n";
        return 1;
    }

    return 0;
}
