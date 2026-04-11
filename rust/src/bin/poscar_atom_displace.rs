use std::collections::BTreeSet;

use clap::Parser;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vasp_utils::poscar::{DisplacementOptions, Poscar};

#[derive(Parser)]
#[command(about = "Randomly displace atoms in a POSCAR file to generate perturbed structures")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    /// Number of displaced structure files to create (1–1000)
    #[arg(short = 'n', long, default_value_t = 1)]
    nfiles: u32,
    /// Number of atoms to displace
    #[arg(long, default_value_t = 1)]
    natoms: usize,
    /// Displace all eligible atoms (overrides --natoms)
    #[arg(long, default_value_t = false)]
    allatoms: bool,
    /// Maximum isotropic displacement amplitude in Å
    #[arg(short = 'a', long, default_value_t = 0.01)]
    amp: f64,
    /// Maximum displacement along x in Å
    #[arg(long)]
    ampx: Option<f64>,
    /// Maximum displacement along y in Å
    #[arg(long)]
    ampy: Option<f64>,
    /// Maximum displacement along z in Å
    #[arg(long)]
    ampz: Option<f64>,
    /// Random seed for deterministic displacements
    #[arg(long)]
    seed: Option<u64>,
    /// Comma-separated element symbols to displace
    #[arg(long, value_delimiter = ',')]
    species: Vec<String>,
    /// 1-based atom indices / ranges (e.g. 1,3-5,8)
    #[arg(long, value_delimiter = ',')]
    indices: Vec<String>,
    /// Wrap fractional coordinates into [0, 1)
    #[arg(long, default_value_t = false)]
    wrap: bool,
    /// Remove net Cartesian translation over displaced atoms
    #[arg(long = "zero-net", default_value_t = false)]
    zero_net: bool,
}

fn parse_indices(tokens: &[String], total: i32) -> Result<BTreeSet<usize>, String> {
    let mut out = BTreeSet::new();
    for token in tokens {
        if let Some(dash) = token.find('-') {
            let lo: i32 = token[..dash]
                .parse()
                .map_err(|_| format!("Invalid range token '{token}'"))?;
            let hi: i32 = token[dash + 1..]
                .parse()
                .map_err(|_| format!("Invalid range token '{token}'"))?;
            if lo > hi || lo < 1 || hi > total {
                return Err(format!(
                    "Range {lo}-{hi} out of bounds (1..={total})"
                ));
            }
            for i in lo..=hi {
                out.insert((i - 1) as usize);
            }
        } else {
            let idx: i32 = token
                .parse()
                .map_err(|_| format!("Invalid index '{token}'"))?;
            if idx < 1 || idx > total {
                return Err(format!("Index {idx} out of bounds (1..={total})"));
            }
            out.insert((idx - 1) as usize);
        }
    }
    Ok(out)
}

fn main() {
    let cli = Cli::parse();

    if cli.nfiles < 1 || cli.nfiles > 1000 {
        eprintln!("Error: --nfiles must be in the range 1–1000");
        std::process::exit(1);
    }

    let ampx = cli.ampx.unwrap_or(cli.amp);
    let ampy = cli.ampy.unwrap_or(cli.amp);
    let ampz = cli.ampz.unwrap_or(cli.amp);

    if ampx.max(ampy).max(ampz) > 0.75 {
        eprintln!(
            "Warning: amplitude is above 0.75 Å! That can cause errors running VASP!"
        );
    }

    let original = match Poscar::read(&cli.input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let atom_species = original.atom_species();

    // Build initial candidate set (all atoms)
    let mut selected: BTreeSet<usize> = (0..atom_species.len()).collect();

    // Filter by species
    if !cli.species.is_empty() {
        let by_species: BTreeSet<usize> = atom_species
            .iter()
            .enumerate()
            .filter(|(_, el)| cli.species.contains(el))
            .map(|(i, _)| i)
            .collect();
        selected = selected.intersection(&by_species).copied().collect();
    }

    // Filter by indices
    if !cli.indices.is_empty() {
        let by_indices =
            match parse_indices(&cli.indices, original.total_atoms) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: invalid --indices: {e}");
                    std::process::exit(1);
                }
            };
        selected = selected.intersection(&by_indices).copied().collect();
    }

    let candidate_indices: Vec<usize> = selected.into_iter().collect();
    if candidate_indices.is_empty() {
        eprintln!("Warning: no atoms matched filters; no files generated.");
        return;
    }

    let n_atoms = if cli.allatoms {
        if cli.natoms != 1 {
            eprintln!("Note: --allatoms overrides --natoms");
        }
        println!("Displacing all eligible atoms in input file.");
        candidate_indices.len()
    } else if cli.natoms > candidate_indices.len() {
        eprintln!(
            "Warning: requested more atoms than eligible. \
             Displacing all {} eligible atoms.",
            candidate_indices.len()
        );
        candidate_indices.len()
    } else {
        cli.natoms
    };

    println!("Number of atoms to displace: {n_atoms}");

    let options = DisplacementOptions {
        candidate_indices,
        wrap_direct: cli.wrap,
        zero_net: cli.zero_net,
        ampx,
        ampy,
        ampz,
    };

    macro_rules! run {
        ($rng:expr) => {{
            for j in 1..=cli.nfiles {
                let mut output = original.clone();
                output.displace_atoms(n_atoms, &options, $rng);
                let filename = format!("POSCAR_modified{j}");
                if let Err(e) = output.write(&filename) {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }};
    }

    match cli.seed {
        Some(s) => run!(&mut StdRng::seed_from_u64(s)),
        None    => run!(&mut StdRng::from_entropy()),
    }
}
