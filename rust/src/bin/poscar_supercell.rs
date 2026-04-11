use clap::Parser;
use vasp_utils::poscar::Poscar;

#[derive(Parser)]
#[command(about = "Create a diagonal supercell from a POSCAR")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    #[arg(short, long, default_value = "")]
    output: String,
    /// Supercell dimensions (nx ny nz)
    #[arg(long, num_args = 3, default_values_t = [1i32, 1, 1])]
    dim: Vec<i32>,
}

fn main() {
    let cli = Cli::parse();
    let nx = cli.dim[0];
    let ny = cli.dim[1];
    let nz = cli.dim[2];

    if cli.dim.iter().any(|&d| d <= 0) {
        eprintln!("Error: all dimensions must be positive integers");
        std::process::exit(1);
    }
    if cli.dim.iter().any(|&d| d > 100) {
        eprintln!("Error: one of the dimensions exceeds 100!");
        std::process::exit(1);
    }
    if cli.dim.iter().any(|&d| d > 20) {
        eprintln!("Warning: one of the dimensions exceeds 20. Consider if you need that!");
    }

    let poscar = match Poscar::read(&cli.input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let supercell = poscar.make_supercell(nx, ny, nz);
    if supercell.total_atoms == 0 {
        eprintln!("Error: failed to make supercell");
        std::process::exit(1);
    }

    let out = if cli.output.is_empty() {
        format!("POSCAR_{nx}_{ny}_{nz}")
    } else {
        cli.output.clone()
    };

    if let Err(e) = supercell.write(&out) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    println!("Supercell written to: {out}");
}
