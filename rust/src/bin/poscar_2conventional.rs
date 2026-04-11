use clap::Parser;
use vasp_utils::symmetry::standardize_cell;
use vasp_utils::poscar::Poscar;

#[derive(Parser)]
#[command(about = "Transform POSCAR to conventional cell using spglib")]
struct Cli {
    #[arg(short, long, default_value = "POSCAR")]
    input: String,
    #[arg(short, long, default_value = "POSCAR_conventional")]
    output: String,
    /// Symmetry tolerance (Å)
    #[arg(long, default_value_t = 1e-5)]
    symprec: f64,
    /// Idealize the conventional cell
    #[arg(long, default_value_t = false)]
    idealize: bool,
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

    match standardize_cell(&poscar, cli.symprec, false, cli.idealize) {
        Some(conv) => {
            if let Err(e) = conv.write(&cli.output) {
                eprintln!("{e}");
                std::process::exit(1);
            }
            println!("Conventional cell written to: {}", cli.output);
        }
        None => {
            eprintln!("Error: spglib failed to create conventional cell.");
            std::process::exit(1);
        }
    }
}
