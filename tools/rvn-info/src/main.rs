mod cli;

use crate::cli::{Cli, InfoArgs};
use clap::Parser;
use log::debug;
use rvn_core_info::render_header;
use rvn_core_parser::Model;
use rvn_globals::{GlobalOpts, get_globals, init_globals};

pub async fn run_info_cmd(cmd: InfoArgs, _globals: &GlobalOpts) -> anyhow::Result<()> {
    debug!("rvn-info::header");
    debug!("rvn-info file: {:#?}", &cmd.file);

    let model = Model::open(&cmd.file)?;

    if cmd.header {
        println!("{}", render_header(&model));
    }

    if cmd.metadata {
        debug!("[DEBUG] cmd.metadata");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let _ = env_logger::try_init();

    init_globals(cli.g);
    run_info_cmd(cli.args, get_globals()).await
}
