#ifndef POSCAR_SUPERCELL_H_INCLUDED
#define POSCAR_SUPERCELL_H_INCLUDED

#include <string>

bool readInput(int argc, char* argv[], std::string& inputFile, std::string& outputFile, bool& diagonal, double S[9]);
bool validateInput(const std::string& inputFile, const std::string& outputFile, double S[9]);
void printHelp();

#endif  // POSCAR_SUPERCELL_H_INCLUDED
