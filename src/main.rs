/*
Thomas Abel
2022-08-09
Machine Learning
*/
mod utility;
mod network;

use std::error::Error;
use ndarray::prelude::*;
use crate::utility::constants::*;
use crate::utility::{read, epoch, visuals, write};
use crate::network::train_test;

type Matrix = Array2<f32>;
type Target = Array1<String>;
type Input = (Matrix, Target);
type ReadResult = Result<Input, Box<dyn Error>>;

/// Main functions.
fn main() {
    // Read the files.
    let index_train: usize = 0;
    let index_test: usize = 0;
    let path = [
        "./data/mnist_test_short.csv",
        "./data/mnist_test.csv",
        "./data/mnist_train.csv",
        "./data/test.csv",
    ];
    print!("Reading data, please be patient... ");
    let input_train = read::read_eval(path[index_train], DIVISOR);
    let input_test = read::read_eval(path[index_test], DIVISOR);
    println!("Done.");
    
    // Test
    test(&input_train, &input_test);

    // Experiment 1: Hidden Nodes

    // Experiment 2: Momentum

    // Experiment 3: Inputs
    

    // End message.
    println!("Ending Session.");
}

// Test
fn test(input_train: &Input, input_test: &Input) {
     // Create the network and data storage.
    let mut network = train_test::create_network(
        INPUT, HIDDEN[2], OUTPUT, LOW, HIGH
    );
    let mut info = epoch::Info {
        epoch: EPOCH,
        state: train_test::EvaluateState::Train, 
        learn_rate: LEARN, 
        momentum: MOMENTUM[0], 
        fraction: TRAIN[0], 
        print: true,
    };

    let name = "Test 30".to_string();
    let result = train_test::epoch_set(&mut network, input_train, input_test, &CLASS, &mut info);
    record(&name, &result);
}

// Creates an graph image and .csv text file recording the results.
fn record(name: &str, result: &(epoch::Results, epoch::Results)) {
    match visuals::accuracy_graph_png(name, &result.1) {
        Ok(_o) => {
            println!("Created image for {}.", name);
        }
        Err(e) => {
            println!("{}", e);
        }
    }
    match write::write_csv(name, &result.1) {
        Ok(_o) => {
            println!("Created text file for {}.", name);
        }
        Err(e) => {
            println!("{}", e);
        }
    }
}
