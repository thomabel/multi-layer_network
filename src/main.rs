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
use crate::utility::{read, epoch, visuals};
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
    
    // Create the network and data storage.
    let hidden = HIDDEN[2];
    let mut network = train_test::create_network(INPUT, hidden, OUTPUT, LOW, HIGH);
    let mut info = epoch::Info {
        epoch: EPOCH,
        state: train_test::EvaluateState::Train, 
        learn_rate: LEARN, 
        momentum: MOMENTUM[0], 
        fraction: TRAIN[0], 
        print: false,
    };

    // Test
    let result = train_test::epoch_set(&mut network, input_train, input_test, &CLASS, &mut info);
    let title = "Test 30".to_string();
    match visuals::accuracy_graph_png(&result.1, &title) {
        Ok(_o) => {
            // Good
            println!("Created image for {}.", title);
        }
        Err(e) => {
            println!("{}", e);
            // Not good but okay
        }
    }

    // Experiment 1: Hidden Nodes

    // Experiment 2: Momentum

    // Experiment 3: Inputs
    

    // End message.
    println!("Ending Session.");
}
