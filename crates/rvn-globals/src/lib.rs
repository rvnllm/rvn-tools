use clap::{Args, ValueEnum};
use once_cell::sync::OnceCell;
use std::path::PathBuf;

static INIT_SYNC: std::sync::Once = std::sync::Once::new();
static GLOBAL_OPTS: OnceCell<GlobalOpts> = OnceCell::new();

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum Device {
    Cpu,
    #[value(alias = "gpu")]
    Cuda,
    #[value(alias = "m1", alias = "m2", alias = "m3", alias = "apple")]
    Metal,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum, Debug)]
pub enum OutputFormat {
    Pretty,
    Json,
    Yaml,
}

#[derive(Args, Debug)]
pub struct GlobalOpts {
    // write output to a file instead of a stdout
    #[arg(short = 'o', long, value_name = "PATH", global = true)]
    pub output: Option<PathBuf>,

    // pretty | json | yaml | text
    #[arg(
        short = 'F',
        long,
        value_enum,
        default_value = "pretty",
        value_name = "FMT",
        global = true
    )]
    pub format: OutputFormat,

    // Target device
    #[arg(short = 'd', long, value_enum, default_value = "cpu", global = true)]
    pub device: Device,

    #[arg(short = 't', long, help = "Number of threads (optional)")]
    pub threads: Option<usize>,

    // verbosity (-v, -vv, etc.)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
}

// utility functions
// instead of typing all the damn type RUST_LOG='debug", just add the stupid flag here
pub fn setup_logging(level: u8) {
    let level = match level {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };

    unsafe { std::env::set_var("RUST_LOG", format!("rvn={}", level)) };

    let _ = env_logger::try_init();
}

pub fn setup_rayon(threads: Option<usize>) {
    INIT_SYNC.call_once(|| {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = threads {
            builder = builder.num_threads(n);
        }
        builder
            .build_global()
            .expect("Failed to build global threading pool");
    });
}

pub fn init_globals(opts: GlobalOpts) {
    if GLOBAL_OPTS.set(opts).is_err() {
        panic!("Global options already initialized");
    }

    let opts = GLOBAL_OPTS.get().expect("options just initialized");

    // logging
    setup_logging(opts.verbose);
    // threading
    setup_rayon(opts.threads);
}

pub fn get_globals() -> &'static GlobalOpts {
    GLOBAL_OPTS.get().expect("Global options not initialized")
}
