#include <CLI/CLI.hpp>
#include <iostream>
#include <string>

#include "poscar_file.h"

int main(int argc, char* argv[]) {
    CLI::App app{"Randomly displace atoms in a POSCAR file to generate perturbed structures"};

    std::string filename{"POSCAR"};
    int n_atoms{1};
    int n_files{1};
    double amplitude{0.01};
    bool allAtoms{false};

    app.add_option("--input,-i", filename, "Input POSCAR file")->capture_default_str()->check(CLI::ExistingFile);
    app.add_option("--nfiles,-n", n_files, "Number of displaced structure files to create")
        ->capture_default_str()
        ->check(CLI::Range(1, 1000));
    app.add_option("--natoms", n_atoms, "Number of atoms to displace")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    app.add_flag("--allatoms", allAtoms, "Displace all atoms (overrides --natoms)");
    app.add_option("--amp,-a", amplitude, "Maximum displacement amplitude in Angstroms")
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);

    CLI11_PARSE(app, argc, argv);

    if (amplitude > 0.1) {
        std::cerr << "Warning: amplitude is above 10%! That can cause errors running VASP!\n";
    }

    POSCAR original;
    if (!original.readPOSCAR(filename)) {
        std::cerr << "Error: reading POSCAR file " << filename << "\n";
        return 1;
    }

    if (allAtoms) {
        if (n_atoms != 1) {
            std::cerr << "Note: --allatoms overrides --natoms\n";
        }
        n_atoms = original.total_atoms;
        std::cout << "Displacing all atoms in input file.\n";
    } else if (n_atoms > original.total_atoms) {
        std::cerr << "Warning: requested more atoms than available. Displacing all " << original.total_atoms
                  << " atoms.\n";
        n_atoms = original.total_atoms;
    }

    std::cout << "Number of atoms to displace: " << n_atoms << "\n";

    for (int j = 0; j < n_files; j++) {
        POSCAR output(original);
        std::string filenameOut = "POSCAR_modified" + std::to_string(j + 1);
        output.displaceAtoms(n_atoms, amplitude);
        output.writePOSCAR(filenameOut);
    }

    return 0;
}
