#ifndef SYMMETRY_H_INCLUDED
#define SYMMETRY_H_INCLUDED

#include <spglib.h>

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>

struct POSCAR;

/// @brief RAII owner for a heap-allocated SpglibDataset.
using SpglibDatasetPtr = std::unique_ptr<SpglibDataset, void (*)(SpglibDataset*)>;

/**
 * @brief Run spglib symmetry analysis on a crystal structure.
 *
 * The input POSCAR is converted to fractional coordinates internally if needed.
 *
 * @param poscar   Input crystal structure.
 * @param symprec  Symmetry tolerance (Å).
 * @return         Owning pointer to the spglib dataset, or a null pointer on failure.
 */
SpglibDatasetPtr analyzeSymmetry(const POSCAR& poscar, const double& symprec);

/**
 * @brief Print a human-readable symmetry summary to stdout.
 *
 * @param dataset      spglib dataset from analyzeSymmetry().
 * @param wyckoff      If true, print Wyckoff letters for each atom.
 * @param symoperation If true, print all symmetry operations (rotations + translations).
 */
void printSymmetryInfo(const SpglibDataset& dataset, const bool& wyckoff, const bool& symoperation);

/**
 * @brief Print all symmetry operations (rotation matrices and translation vectors) to stdout.
 * @param dataset  spglib dataset from analyzeSymmetry().
 */
void printSymmetryOperations(const SpglibDataset& dataset);

/**
 * @brief Standardise a crystal cell using spglib.
 *
 * Wraps @c spg_standardize_cell.  Setting @p primitive=1 returns the primitive cell;
 * @p primitive=0 returns the conventional cell.
 *
 * @param poscar    Input crystal structure.
 * @param symprec   Symmetry tolerance (Å).
 * @param primitive 1 = primitive cell, 0 = conventional cell.
 * @param idealize  1 = idealize lattice parameters, 0 = keep input metric.
 * @return          Standardised POSCAR, or @c std::nullopt if spglib fails.
 */
std::optional<POSCAR> standardizeCell(const POSCAR& poscar, const double& symprec, const int& primitive,
                                      const int& idealize);

/**
 * @brief Prepare spglib input arrays from a POSCAR.
 *
 * Transposes the row-vector POSCAR lattice into spglib's column-vector layout,
 * copies fractional coordinates, and assigns integer type indices per element.
 *
 * @param poscarDirect  Input POSCAR (must be in fractional/Direct coordinates).
 * @param lattice       Output 3×3 lattice in spglib column-vector convention.
 * @param positions     Output fractional positions, size >= total_atoms.
 * @param types         Output integer type per atom, size >= total_atoms.
 * @param element_map   Output map from element symbol to type index.
 */
void initializeSpglibInput(POSCAR& poscarDirect, double lattice[3][3], std::vector<std::array<double, 3>>& positions,
                           std::vector<int>& types, std::map<std::string, int>& element_map);

#endif  // SYMMETRY_H_INCLUDED
