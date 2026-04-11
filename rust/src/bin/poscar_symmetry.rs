use clap::Parser;
use vasp_utils::poscar::Poscar;
use vasp_utils::symmetry::{analyze_symmetry, print_symmetry_info, standardize_cell};

#[derive(Parser)]
#[command(about = "Analyze symmetry of a POSCAR structure using spglib")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    #[arg(short, long, default_value = "POSCAR_primitive")]
    output: String,
    /// Symmetry tolerance (Å)
    #[arg(long, default_value_t = 1e-5)]
    symprec: f64,
    /// Write primitive cell POSCAR
    #[arg(short, long, default_value_t = false)]
    primitive: bool,
    /// Print Wyckoff letters for each atom
    #[arg(short, long, default_value_t = false)]
    wyckoff: bool,
    /// Print all symmetry operations
    #[arg(short = 's', long, default_value_t = false)]
    symoper: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.symprec > 1e-3 {
        eprintln!("Warning: symprec is too high! Consider using the default value 1e-5.");
    }

    let poscar = match Poscar::read(&cli.input) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let dataset = match analyze_symmetry(&poscar, cli.symprec) {
        Some(d) => d,
        None => {
            eprintln!("Error: failed to analyze symmetry.");
            std::process::exit(1);
        }
    };

    print_symmetry_info(&dataset, cli.wyckoff, cli.symoper);

    if cli.primitive {
        println!("Creating the primitive cell file.");
        match standardize_cell(&poscar, cli.symprec, true, false) {
            Some(prim) => {
                if let Err(e) = prim.write(&cli.output) {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
            None => {
                eprintln!("Error: failed to create primitive POSCAR.");
                std::process::exit(1);
            }
        }
    }
}
