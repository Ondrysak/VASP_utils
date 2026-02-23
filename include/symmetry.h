#ifndef SYMMETRY_H_INCLUDED
#define SYMMETRY_H_INCLUDED

#include <spglib.h>

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>

struct POSCAR;

using SpglibDatasetPtr = std::unique_ptr<SpglibDataset, void (*)(SpglibDataset*)>;

SpglibDatasetPtr analyzeSymmetry(const POSCAR& poscar, const double& symprec);
void printSymmetryInfo(const SpglibDataset& dataset, const bool& wyckoff, const bool& symoperation);
void printSymmetryOperations(const SpglibDataset& dataset);
std::optional<POSCAR> standardizeCell(const POSCAR& poscar, const double& symprec, const int& primitive,
                                      const int& idealize);

void initializeSpglibInput(POSCAR& poscarDirect, double lattice[3][3], std::vector<std::array<double, 3>>& positions,
                           std::vector<int>& types, std::map<std::string, int>& element_map);

#endif  // SYMMETRY_H_INCLUDED
