#ifndef POSCAR_FILE_H_INCLUDED
#define POSCAR_FILE_H_INCLUDED

#include <array>
#include <string>
#include <vector>

/**
 * @brief Single atom in a crystal structure.
 *
 * Coordinates are either fractional (Direct) or Cartesian depending on
 * the @c is_direct flag of the parent POSCAR.
 */
struct Atom {
    double x, y, z;                                         ///< Fractional or Cartesian coordinates.
    std::array<bool, 3> selective_flags{true, true, true};  ///< Selective dynamics flags (T/F per axis).
};

/**
 * @brief Options controlling random atom displacement.
 *
 * Passed to POSCAR::displaceAtoms(int, const DisplacementOptions&).
 */
struct DisplacementOptions {
    std::vector<size_t> candidate_indices;  ///< Atom indices to displace (0-based). Empty = all atoms.
    bool wrap_direct{false};                ///< Wrap displaced fractional coordinates back into [0, 1).
    bool zero_net{false};                   ///< Remove net Cartesian displacement after perturbation.
    double ampx{0.01};                      ///< Displacement amplitude along x (Å).
    double ampy{0.01};                      ///< Displacement amplitude along y (Å).
    double ampz{0.01};                      ///< Displacement amplitude along z (Å).
};

/**
 * @brief In-memory representation of a VASP POSCAR / CONTCAR file.
 *
 * Lattice vectors are stored as **row vectors**: @c lattice[i][j] is the
 * j-th Cartesian component of the i-th basis vector.
 *
 * @note When passing the lattice to spglib, transpose to column-vector
 *       layout — see symmetry.cpp for the canonical pattern.
 */
struct POSCAR {
    std::string comment{"System"};      ///< First line of the POSCAR file.
    double scale{1.000000};             ///< Global scaling factor.
    double lattice[3][3];               ///< Row-vector lattice: lattice[i] = i-th basis vector.
    std::vector<std::string> elements;  ///< Element symbols in order.
    std::vector<int> num_atoms;         ///< Number of atoms per element, same order as @c elements.
    bool selective_dynamics = false;    ///< Whether a selective dynamics block is present.
    bool is_direct = true;              ///< true = Direct (fractional), false = Cartesian.
    std::vector<Atom> coordinates;      ///< Atomic positions, length == total_atoms.
    int total_atoms{0};                 ///< Total number of atoms.

    /** @brief Read a POSCAR/CONTCAR file from disk. @return false on parse error. */
    bool readPOSCAR(const std::string& filename);

    /** @brief Write a POSCAR file to disk. @return false on I/O error. */
    bool writePOSCAR(const std::string& filenameOut);

    /**
     * @brief Displace @p n_atoms randomly chosen atoms by a uniform isotropic amplitude.
     * @param n_atoms   Number of atoms to displace.
     * @param amplitude Displacement magnitude (Å).
     */
    void displaceAtoms(int n_atoms, double amplitude);

    /**
     * @brief Displace atoms according to the full option set.
     * @param n_atoms  Number of atoms to displace from the candidate set.
     * @param options  Per-axis amplitudes, index filter, wrap, zero-net.
     */
    void displaceAtoms(int n_atoms, const DisplacementOptions& options);

    /** @brief Convert coordinates from Cartesian to fractional (Direct). */
    void toDirect();

    /** @brief Convert coordinates from fractional to Cartesian. */
    void toCartesian();

    /** @brief Write an atat-style ctrls.in file. @return false on I/O error. */
    bool writeCtrlsFile(const std::string& filenameOut);

    /**
     * @brief Build a supercell from a 3×3 transformation matrix.
     * @param S Column-major supercell matrix (integer entries).
     * @return New POSCAR representing the supercell.
     */
    POSCAR makeSupercell(const double S[9]);

    /** @brief Return the ordered list of unique element symbols. */
    std::vector<std::string> atomSpecies() const;

private:
    Atom displaceAtomWithVector(size_t atom_index, const DisplacementOptions& options);
    void setScaleTo1();
};
#endif  // POSCAR_FILE_H_INCLUDED
