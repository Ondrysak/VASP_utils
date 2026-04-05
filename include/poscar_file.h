#ifndef POSCAR_FILE_H_INCLUDED
#define POSCAR_FILE_H_INCLUDED

#include <array>
#include <string>
#include <vector>

struct Atom {
    double x, y, z;                               // fractional or Cartesian
    std::array<bool, 3> selective_flags{true, true, true};
};

struct DisplacementOptions {
    std::vector<size_t> candidate_indices;  // if empty => all atoms are candidates
    bool wrap_direct{false};
    bool zero_net{false};
    double ampx{0.01};
    double ampy{0.01};
    double ampz{0.01};
};

struct POSCAR {
    std::string comment{"System"};      // first line
    double scale{1.000000};             // scaling factor
    double lattice[3][3];               // 3x3 lattice vectors
    std::vector<std::string> elements;  // element symbols
    std::vector<int> num_atoms;         // number of atoms per element
    bool selective_dynamics = false;    // optional
    bool is_direct = true;              // true = Direct, false = Cartesian
    std::vector<Atom> coordinates;      // Nx3 coordinates
    int total_atoms{0};

    bool readPOSCAR(const std::string& filename);
    bool writePOSCAR(const std::string& filenameOut);
    void displaceAtoms(int n_atoms, double amplitude);
    void displaceAtoms(int n_atoms, const DisplacementOptions& options);
    void toDirect();
    void toCartesian();
    bool writeCtrlsFile(const std::string& filenameOut);
    POSCAR makeSupercell(const double S[9]);
    std::vector<std::string> atomSpecies() const;

private:
    Atom displaceAtomWithVector(size_t atom_index, const DisplacementOptions& options);
    void setScaleTo1();
};
#endif  // POSCAR_FILE_H_INCLUDED
