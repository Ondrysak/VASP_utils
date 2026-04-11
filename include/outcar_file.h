#ifndef OUTCAR_FILE_H_INCLUDED
#define OUTCAR_FILE_H_INCLUDED

#include <string>

/**
 * @brief 6×6 elastic tensor extracted from a VASP OUTCAR file.
 *
 * Index ordering matches VASP's OUTCAR output:
 *   0 = XX,  1 = YY,  2 = ZZ,  3 = XY,  4 = YZ,  5 = ZX
 *
 * This differs from standard Voigt notation (XX,YY,ZZ,YZ,ZX,XY), so when
 * mapping to crystallographic labels use:
 *   C44 (YZ,YZ) → C[4][4],  C55 (ZX,ZX) → C[5][5],  C66 (XY,XY) → C[3][3]
 *
 * Units are GPa (converted from kBar on read: 1 kBar = 0.1 GPa).
 */
struct ElasticTensor {
    double C[6][6]{};  ///< Elastic constants C_ij in GPa.
};

/**
 * @brief Data extracted from a VASP OUTCAR file.
 *
 * Parses the "SYMMETRIZED ELASTIC MODULI (kBar)" block.  If the file contains
 * multiple blocks (e.g. restarted runs) the last occurrence is used.
 */
struct OUTCAR {
    bool has_elastic_tensor{false};  ///< True if an elastic tensor was found and parsed.
    ElasticTensor elastic_tensor;    ///< Symmetrized elastic tensor in GPa.

    /**
     * @brief Parse an OUTCAR file and extract available data.
     * @param filename  Path to the OUTCAR file.
     * @return false on I/O error or if no elastic tensor block was found.
     */
    bool readOUTCAR(const std::string& filename);
};

#endif  // OUTCAR_FILE_H_INCLUDED
