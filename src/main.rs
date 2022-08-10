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

    println!("End program.\n");
}

fn train(input: &Input) -> Network {
    let size = [
        LayerSize::new(HIDDEN, INPUT, STORAGE),
        LayerSize::new(OUTPUT, HIDDEN, STORAGE),
    ];
    let mut network = Network::new(&size[..]);

    // Feed each input into the network and update the weights.
    for i in 0..input.1.len() {
        let row = input.0.row(i);
        let target = input.1[i].parse::<f32>().unwrap();
        
        println!("Output");
        let output = 
            network.output(&row);
        println!();

        let class = classify(&output);

        println!("Error");
        network.error(target);
        println!();

        println!("Weight");
        network.weight(&row, LEARN_RATE, MOMENTUM);
        println!();
    }

    network
}

fn classify(output: &ArrayView1<f32>) -> String {
    let mut index = 0;
    let mut value = 0.;
    let len = output.len();
    for i in 0..len {
        if output[i] > value {
            index = i;
            value = output[i];
        }
    }

    CLASS[index].to_string()
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
