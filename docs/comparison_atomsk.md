# VASP_utils vs Atomsk — Comparative Analysis

## Overview

| | **VASP_utils** | **Atomsk** |
|---|---|---|
| Author | Ondřej / contributors | Pierre Hirel (Univ. Lille) |
| Language | C++17 | Fortran 90 |
| License | — | GPL |
| Design philosophy | Focused POSCAR toolkit, VASP-centric | Universal atomic structure converter and editor |
| Interface | Multiple small CLIs, one per operation | Single binary with modes and chainable options |
| File format support | POSCAR (read/write), ctrls (write) | 37 format variants across ~28 types (see below) |
| Symmetry backend | spglib v2.3.0 | None built-in (only `-spacegroup` expansion) |
| Dependencies | BLAS/LAPACK, spglib, CLI11 | Standard Fortran libs only |
| Build system | CMake, auto-fetches deps | Makefile / configure |
| Test suite | GoogleTest unit tests | — |

---

## Feature Matrix

### File Format Support

| Format | VASP_utils | Atomsk |
|---|---|---|
| POSCAR / CONTCAR (VASP) | Read + Write | Read + Write |
| OUTCAR (VASP) | — | Read (multi-snapshot) |
| ctrls / ctrls.system (Questaal/ecalj) | Write | — |
| CIF | — | Read + Write |
| XYZ / extended XYZ | — | Read + Write |
| LAMMPS data | — | Read + Write |
| LAMMPS dump (custom) | — | Read |
| Quantum ESPRESSO input (`pw`) | — | Read + Write |
| Quantum ESPRESSO output (`out`) | — | Read |
| ABINIT | — | Read + Write |
| SIESTA (`fdf`, `xv`) | — | Read + Write |
| CRYSTAL | — | Read + Write |
| GULP (`gin`) | — | Read + Write |
| DL_POLY (`CONFIG`) | — | Read + Write |
| VESTA | — | Read + Write |
| XCrysDen (`xsf`) | — | Read + Write |
| AtomEye / QSTEM (`cfg`) | — | Read + Write |
| PDB | — | Read + Write |
| Atomsk native binary (`atsk`) | — | Read + Write |
| CSV / plain columns (`dat`) | — | Read + Write |
| IMD, XMD, MOLDY, BOPfox, BOP, MBPP | — | Read + Write |
| ddplot | — | Write |
| Dr Probe (`cel`), JEMS | — | Read + Write |

### Structure Construction

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Create crystal from scratch | — | Yes (`--create`): sc, bcc, fcc, hcp, diamond, rocksalt, perovskite, fluorite, wurtzite, graphite, Laves, A15, L1₀, L1₂, ... |
| Nanotube generation | — | Yes (any chirality m,n) |
| Polycrystal / Voronoi tessellation | — | Yes (`--polycrystal`): 3-D and 2-D columnar grains, random or explicit grain positions/orientations |
| Supercell generation | Yes (`poscar_supercell`, nx×ny×nz) | Yes (`-duplicate`, arbitrary repetitions) |
| Merge / stack structures | — | Yes (`--merge`: stack along axis, combine with optional rescaling) |
| NEB / interpolated structures | — | Yes (`--interpolate`: N images between two endpoints) |
| Crystal orientation control | — | Yes (Miller indices `orient [hkl][hkl][hkl]` for all lattice types) |

### Cell / Box Manipulation

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Direct ↔ Cartesian conversion | Yes (`poscar_d2c`, `poscar_c2d`) | Yes (`-fractional`) |
| Conventional cell standardization | Yes (`poscar_2conventional`, spglib) | Partial (`-reduce-cell`, no spglib) |
| Primitive cell reduction | Yes (`poscar_2primitive`, spglib) | Yes (`-reduce-cell`) |
| Find orthogonal equivalent cell | — | Yes (`-orthogonal-cell`) |
| Minimize box skew | — | Yes (`-unskew`) |
| Align cell vector to X axis | — | Yes (`-alignx`) |
| Modify cell vectors directly | — | Yes (`-cell`) |
| Re-detect / guess box parameters | — | Yes (`-rebox`) |
| Wrap atoms into box (PBC folding) | — | Yes (`-wrap`) |

### Symmetry & Crystallography

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Space group determination | Yes (spglib: number, symbol, Hall, point group) | — |
| Wyckoff positions | Yes (`--wyckoff`) | — |
| Symmetry operations listing | Yes (`--symoper`: rotation + translation) | — |
| Apply space group symmetry to expand cell | — | Yes (`-spacegroup` by name or number 1–230) |
| Symmetry tolerance control | Yes (`--symprec`) | — |
| Primitive cell (spglib standardized) | Yes (`poscar_2primitive`) | Approximate (`-reduce-cell`) |
| Conventional cell (spglib standardized) | Yes (`poscar_2conventional`) | — |
| CIF read with occupancy and Wyckoff data | — | Yes |

### Atomic Perturbation & Displacement

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Random displacement of N atoms | Yes (`poscar_atom_displace --natoms`) | Yes (`-disturb`) |
| Displace all atoms | Yes (`--allatoms`) | Yes (`-disturb`) |
| Amplitude control | Yes (`--amp`, in Å) | Yes |
| Generate N perturbed structures at once | Yes (`--nfiles`) | Requires scripting / `--list` |
| Displacement distribution | Uniform sphere (cbrt-scaled radius) | Not documented in detail |
| Seed for reproducibility | Yes (`seedRandom`) | — |
| Maxwell-Boltzmann velocity assignment | — | Yes (`-velocity`) |

### Geometric Transformations

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Rotation | — | Yes (`-rotate`) |
| Crystallographic reorientation | — | Yes (`-orient`) |
| Mirror / reflection | — | Yes (`-mirror`) |
| Translation / shift | — | Yes (`-shift`) |
| Center structure in box | — | Yes (`-center`) |
| Roll / bend | — | Yes (`-roll`) |
| Torsional deformation | — | Yes (`-torsion`) |
| Uniaxial strain / shear | — | Yes (`-deform`) |
| Stress application (Hooke's law) | — | Yes (`-stress`) |

### Defect Insertion

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Point displacements (perturbation) | Yes | Yes (`-disturb`) |
| Dislocations (screw, edge, mixed) | — | Yes (`-dislocation`, iso & aniso elasticity) |
| Dislocation loops | — | Yes |
| Cracks (Mode I/II/III) | — | Yes (`-crack`) |
| Vacancies / interstitials | — | Via `-remove-atom` / `-add-atom` |
| Species substitution / alloying | — | Yes (`-substitute`) |
| Remove duplicate atoms | — | Yes (`-remove-doubles`) |
| Push apart overlapping atoms | — | Yes (`-separate`) |

### Atom Selection & Sorting

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Select by species | — | Yes (`-select`) |
| Select by geometric region (box/sphere/cylinder) | — | Yes |
| Select by index / range | — | Yes |
| Select random fraction | — | Yes |
| Invert selection | — | Yes |
| Sort atoms | — | Yes (`-sort` by species, coordinate, property) |
| Swap atoms / axes | — | Yes (`-swap`) |

### Analysis & Characterization

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Displacement difference between snapshots | — | Yes (`--diff`: per-atom vectors, statistics, histogram) |
| Radial distribution function (RDF) | — | Yes (`--rdf`) |
| Local symmetry parameter (CSP, sp3, sp2) | — | Yes (`--local-symmetry`) |
| Nye tensor / per-atom strain field | — | Yes (`--nye`) |
| Density maps (1-D/2-D/3-D) | — | Yes (`--density`) |
| Electric dipole moments | — | Yes (`--edm`) |
| Electronic polarization (core-shell) | — | Yes (`--electronic-polarization`) |
| Space group / symmetry analysis | Yes (spglib: full) | — |

### Multi-Snapshot / Batch

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Generate N structures in one command | Yes (`--nfiles`) | Via `--list` scripting |
| Average positions over snapshots | — | Yes (`--average`) |
| Split multi-snapshot → per-frame files | — | Yes (`--one-in-all`) |
| Merge per-frame files → trajectory | — | Yes (`--gather`) |
| Batch-convert a list of files | — | Yes (`--list`) |
| Apply options from a macro file | — | Yes (`-options`) |
| Interactive / scripted REPL | — | Yes (`--interactive`) |
| NEB chain generation | — | Yes (`--interpolate`) |

### Ionic Core-Shell Model

| Feature | VASP_utils | Atomsk |
|---|---|---|
| Add / remove shells | — | Yes |
| Rebind shells to cores | — | Yes |
| Write shell-model input (GULP, etc.) | — | Yes |

### Output & Integration

| Feature | VASP_utils | Atomsk |
|---|---|---|
| VASP POSCAR output | Yes | Yes |
| Questaal / ecalj ctrls output | Yes | — |
| Output to stdout (`-`) | — | Yes |
| Suppress output (`NULL`) | — | Yes |
| Multi-format output in one command | — | Yes (list output formats) |
| Auxiliary property propagation (forces, charges, velocities) | — | Yes |

---

## Summary Assessment

### VASP_utils strengths
- **Rigorous symmetry analysis** via spglib: space group number/symbol, Hall symbol, point group, Wyckoff positions, full symmetry operations list. Atomsk has no equivalent.
- **spglib-standardized cells**: both primitive and conventional cell reduction use the same industry-standard library as PHONOPY, ASE, etc., giving consistent results.
- **VASP-native workflow**: direct POSCAR manipulation without format round-tripping.
- **Questaal/ecalj integration**: unique ctrls output not available in Atomsk.
- **Reproducible perturbations**: seeded RNG + batch `--nfiles` in a single command.
- **Unit-tested codebase**: GoogleTest suite covering I/O, coordinate conversion, and displacement.

### Atomsk strengths
- **Vastly broader format support**: 37 format variants vs 2. Covers all major ab initio, force-field, and EM-simulation codes.
- **Structure creation from scratch**: 20+ crystal types, nanotubes, polycrystals via Voronoi tessellation.
- **Defect physics toolkit**: dislocations (isotropic and anisotropic elasticity), cracks (all three fracture modes), dislocation loops — not trivially replicated.
- **Rich geometric transformations**: rotation, orientation, deformation, strain, torsion, rolling.
- **Analysis modes**: Nye tensor, CSP, RDF, density maps, dipole moments — useful for MD post-processing.
- **Batch and scripting**: `--list`, `-options` macro files, `--interactive`, trajectory handling.
- **Core-shell model**: full support for shell-model potentials used in classical MD.

### When to use which

| Use case | Recommended tool |
|---|---|
| Determine space group / Wyckoff positions | **VASP_utils** |
| Get spglib-standardized primitive or conventional cell | **VASP_utils** |
| Generate perturbed structures for ML training or phonon calculations | **VASP_utils** (`--nfiles` + seeded RNG) |
| Convert POSCAR for Questaal/ecalj | **VASP_utils** |
| Convert between any two of the 28+ supported formats | **Atomsk** |
| Build a crystal from lattice type + parameters | **Atomsk** |
| Insert dislocations or cracks | **Atomsk** |
| Build a polycrystalline model | **Atomsk** |
| Generate NEB images | **Atomsk** |
| Rotate / orient / strain a structure | **Atomsk** |
| Compute Nye tensor or CSP from MD snapshots | **Atomsk** |
| VASP-only workflow needing symmetry data | **VASP_utils** |
| Complex multi-step structure construction | **Atomsk** (chainable options) |
