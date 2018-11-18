extern crate dlfs;
extern crate structopt;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(parse(from_os_str))]
    mnist_data_dir: PathBuf,

    #[structopt(long = "iters-num", default_value = "10000")]
    iters_num: usize,

    #[structopt(long = "batch-size", default_value = "100")]
    batch_size: usize,

    #[structopt(long = "learning_rate", default_value = "0.1")]
    learning_rate: f64,
}

fn main() {
    let opt = Opt::from_args();
    let mnist = dlfs::data::Mnist::load(opt.mnist_data_dir);
    let mut net = dlfs::ch05::TwoLayerNet::default();
    net.train(&mnist, opt.iters_num, opt.batch_size, opt.learning_rate);
}
