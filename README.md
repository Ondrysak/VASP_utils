Still under development. Currently, these utilities for POSCAR file manipulation are implemented:

- poscar_d2c - fractional coordinates to cartesian
- poscar_c2d - cartesian coordinates to fractional
- poscar_symmetry - find symmetry of a cell
- poscar_supercell - create supercell
- poscar_2primitive - create primitive cell
- poscar_2conventional - create conventional cell
- poscar_2ctrls - create ctrls file for ecalj/Questaal package from POSCAR
- poscar_atom_displace - randomly displace atoms
- poscar_kpath - for automatic generation of path in the Brillouin zone for band structure calculation
- poscar_elastic_deformations - generate a set of deformed POSCAR files for elastic constant fitting
- poscar_elastic_fit - fit elastic constant tensor Cij from VASP OUTCAR files (stress or energy method)

For now, the code is as it is; nothing is guaranteed.

----Plans:----

(Implemented) Deformation generator for elastic constants calculation.

----Installation:----

Requires CMake >= 3.22, a C++17 compiler, and BLAS/LAPACK/LAPACKE installed.
CLI11 and spglib are fetched automatically by CMake.

```
mkdir build && cd build
cmake ..
cmake --build . --parallel
```

Executables are placed in `build/bin/`.

----Usage:----

The following tools use CLI11 for argument parsing (run with `--help` for full options):

**poscar_d2c** — Direct to Cartesian conversion
```
poscar_d2c [--input/-i <file>] [--output/-o <file>]
```
Defaults: input `POSCAR`, output `POSCAR_cartesian`.

**poscar_c2d** — Cartesian to Direct conversion
```
poscar_c2d [--input/-i <file>] [--output/-o <file>]
```
Defaults: input `POSCAR`, output `POSCAR_direct`.

**poscar_atom_displace** — Randomly displace atoms to generate perturbed structures
```
poscar_atom_displace [--input/-i <file>] [--nfiles/-n <int>] [--natoms <int>] [--allatoms]
                     [--amp/-a <float>] [--ampx <float>] [--ampy <float>] [--ampz <float>]
                     [--seed <int>] [--species <Na,Cl,...>] [--indices <1,3-5,...>]
                     [--wrap] [--zero-net]
```

| Option | Default | Description |
|---|---|---|
| `--input/-i` | `POSCAR` | Input POSCAR file |
| `--nfiles/-n` | `1` | Number of output files |
| `--natoms` | `1` | Number of atoms to displace |
| `--allatoms` | off | Displace all eligible atoms (overrides `--natoms`) |
| `--amp/-a` | `0.01` | Isotropic max displacement amplitude in Angstroms |
| `--ampx` / `--ampy` / `--ampz` | `0.01` | Axis-specific max displacement amplitudes (override `--amp`) |
| `--seed` | random | Seed RNG for deterministic perturbations |
| `--species` | all species | Comma-separated element symbols to consider |
| `--indices` | all atoms | 1-based list/ranges (e.g., `1,3-5,8`) to consider |
| `--wrap` | off | Wrap fractional coordinates into `[0,1)` after displacement |
| `--zero-net` | off | Remove net Cartesian translation across displaced atoms |

Examples:
```
# Deterministic perturbation of all Cl atoms in 5 structures
poscar_atom_displace -i POSCAR -n 5 --species Cl --allatoms --seed 2026 --amp 0.02

# Anisotropic displacement on selected atoms and keep atoms wrapped in cell
poscar_atom_displace --indices 1,3-8 --natoms 4 --ampx 0.03 --ampy 0.01 --ampz 0.00 --wrap

# Apply perturbation with zero net translation correction
poscar_atom_displace --allatoms --zero-net --amp 0.015
```


**poscar_elastic_deformations** — Generate independent strain deformation set for elastic constant calculations
```
poscar_elastic_deformations [--input/-i <file>] [--output-prefix/-o <prefix>] [--manifest/-m <csv>]
                            [--norm-strains <list>] [--shear-strains <list>] [--symmetric]
```

| Option | Default | Description |
|---|---|---|
| `--input/-i` | `POSCAR` | Input POSCAR file |
| `--output-prefix/-o` | `POSCAR_def` | Prefix for generated POSCAR files |
| `--manifest/-m` | `elastic_deformations.csv` | CSV metadata file (input for `poscar_elastic_fit`) |
| `--norm-strains` | `-0.01,-0.005,0.005,0.01` | Normal strain amplitudes (e11, e22, e33) |
| `--shear-strains` | `-0.01,-0.005,0.005,0.01` | Shear strain amplitudes (e12, e13, e23) |
| `--symmetric` | off | Use symmetric F = sqrt(I+2E) instead of upper-triangular (Cholesky). Recommended when lattice orientation relative to spin must be preserved. |

Amplitudes above 1% trigger a warning; above 5% the run is aborted. For the energy method at least 7 points per mode are recommended — supply more via `--norm-strains`/`--shear-strains`.

Six independent strain modes are generated (3 normal + 3 shear). The deformation gradient satisfies `F^T F = I + 2E`.

**poscar_elastic_fit** — Fit elastic constants from VASP OUTCAR files
```
poscar_elastic_fit [--manifest/-m <csv>] [--outcar-dir/-d <dir>] [--method stress|energy]
                   [--volume/-V <float>] [--output/-o <file>] [--negate-stress] [--quartic]
```

| Option | Default | Description |
|---|---|---|
| `--manifest/-m` | `elastic_deformations.csv` | Manifest CSV from `poscar_elastic_deformations` |
| `--outcar-dir/-d` | `.` | Root dir; OUTCARs looked up as `<dir>/<name>/OUTCAR` or `<dir>/<name>.OUTCAR` |
| `--method` | `stress` | Fitting method (see table below) |
| `--volume/-V` | — | Reference cell volume in Å³ (required for energy method) |
| `--output/-o` | `elastic_constants.txt` | Output file |
| `--negate-stress` | off | Negate VASP stress values (use if fitted Cij have wrong sign) |
| `--quartic` | off | Energy method: extend polynomial to degree 4 (1+ε+ε²+ε³+ε⁴). Improves accuracy when higher-order elastic effects are present; requires ≥5 points per mode (recommend ≥9). |

| Method | Description |
|---|---|
| `stress` | Linear regression σ = C·ε over all deformations. Gives full 6×6 Cij. Requires `ISIF >= 2` in INCAR. ≥5 data points total recommended. |
| `energy` | Polynomial fit E = a₀ + a₁ε + a₂ε² (+ a₃ε³ + a₄ε⁴ with `--quartic`) per mode. Gives diagonal Cii only. ≥7 points per mode recommended (≥9 for quartic). |

The output `elastic_constants.txt` is a plain 6×6 matrix in GPa (Voigt notation: rows/cols ordered 11 22 33 23 13 12).

**poscar_symmetry** — Symmetry analysis using spglib
```
poscar_symmetry [--input/-i <file>] [--output/-o <file>] [--symprec <float>]
                [--primitive/-p] [--wyckoff/-w] [--symoper/-s]
```

| Option | Default | Description |
|---|---|---|
| `--input/-i` | `POSCAR` | Input POSCAR file |
| `--output/-o` | `POSCAR_primitive` | Output file (used with `--primitive`) |
| `--symprec` | `1e-5` | Symmetry tolerance |
| `--primitive/-p` | off | Generate and write primitive cell |
| `--wyckoff/-w` | off | Print Wyckoff positions |
| `--symoper/-s` | off | Print all symmetry operations |

----Dependencies:----

This project relies on external scientific libraries for symmetry analysis and linear algebra operations.

**SPGLIB** — space group symmetry determination and primitive cell search.
Fetched automatically by CMake. Official: https://github.com/spglib/spglib

**CLI11** — command-line argument parsing.
Fetched automatically by CMake. Official: https://github.com/CLIUtils/CLI11

**BLAS / LAPACK / LAPACKE** — linear algebra; must be installed on the system.

----Development Setup:----

This project uses pre-commit hooks to enforce consistent code formatting and catch common issues before commits.

Prerequisites (Ubuntu/WSL):

```
sudo apt-get install -y clang-format
pip3 install pre-commit
```

Installing the hooks:

```
cd VASP_utils
pre-commit install
```

After installation, the following checks run automatically on every `git commit`:

- **trailing-whitespace** — removes trailing whitespace
- **end-of-file-fixer** — ensures files end with a newline
- **check-merge-conflict** — prevents committing merge conflict markers
- **clang-format** — formats C/C++ code according to `.clang-format`

To run all hooks manually on the entire codebase:

```
pre-commit run --all-files
```

To run clang-tidy (requires a build with `compile_commands.json`):

```
pre-commit run --hook-stage manual clang-tidy
```
