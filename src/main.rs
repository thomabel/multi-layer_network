/*
Thomas Abel
2022-08-09
Machine Learning
*/

mod read;
mod constants;
mod print_data;
mod layer_size;
mod layer;
mod network;

use std::error::Error;
use layer_size::LayerSize;
use ndarray::prelude::*;
use network::Network;
use print_data::*;
use crate::constants::*;
use rand::{self, seq::SliceRandom};

type Input = (Array2<f32>, Array1<String>);

fn main() {
    let temp = read();
    let input;
    match temp {
        Ok(mut o) => {
            //o.0 /= DIVISOR;
            input = o;
        }
        Err(e) => {
            println!("{}", e);
            return;
        }
    }
    _print_matrix(&input.0.view(), "INPUT");

    let network = train(&input);
}

fn train(input: &Input) -> Network {
    let size = [
        LayerSize::new(HIDDEN, INPUT, STORAGE),
        LayerSize::new(OUTPUT, HIDDEN, STORAGE),
    ];
    let mut network = Network::new(&size[..]);

    for i in input.0.rows() {
        network.output(&i);
    }

    network
}

fn random_index(size: usize) -> Vec<usize> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());
    vec
}

fn read() -> Result<Input, Box<dyn Error>> {
    let path_index = 3;
    let path = [
        "./data/mnist_test_short.csv",
        "./data/mnist_test.csv",
        "./data/mnist_train.csv",
        "./data/test.csv",
    ];
    println!("Reading data, please be patient...");
    read::read(path[path_index])
}
