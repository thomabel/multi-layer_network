/*
Thomas Abel
2022-08-09
Machine Learning
*/
mod utility;
mod network;

use std::error::Error;
use chronometer::Chronometer;
use ndarray::prelude::*;
use crate::utility::constants::*;
use crate::utility::print_data::_print_time;
use crate::utility::{read, epoch, visuals, write};
use crate::network::train_test;

type Matrix = Array2<f32>;
type Target = Array1<String>;
type Input = (Matrix, Target);
type ReadResult = Result<Input, Box<dyn Error>>;

/// Main functions.
fn main() {
    let mut watch = Chronometer::new();
    watch.start();
    // Read the files.
    let index_train: usize = 2;
    let index_test: usize = 2;
    let path = [
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
        "./data/mnist_test_short.csv",
        "./data/test.csv",
    ];
    print!("Reading data, please be patient... ");
    let input_train = read::read_eval(path[index_train], DIVISOR);
    let input_test = read::read_eval(path[index_test], DIVISOR);
    println!("Done.");
    experiment_set(&input_train, &input_test);
    
    watch.pause();
    _print_time("Total", &watch);
    println!("Ending Session.");
}

// Performs the set of experiments for the assignment.
fn experiment_set(input_train: &Input, input_test: &Input) {
    // Test
    let mut info = epoch::Info {
        epoch: EPOCH,
        learn_rate: LEARN, 
        momentum: MOMENTUM[0], 
        fraction: TRAIN[0], 
        print: false,
    };
    let hidden = HIDDEN[0];

    /*
    // Experiment 1: Hidden Nodes
    for hidden_ in HIDDEN {
        let name = format!("Hidden Nodes = {}", hidden_);
        experiment(&name, input_train, input_test, &info, hidden_);
    }
    
    // Experiment 2: Momentum
    for momentum in MOMENTUM {
        // Don't retest default value.
        if momentum == MOMENTUM[0] {
            continue;
        }
        let name = format!("Momentum = {:.3}", momentum);
        info.momentum = momentum;
        experiment(&name, input_train, input_test, &info, hidden);
    }
    
    */
    // Experiment 3: Inputs
    info.momentum = MOMENTUM[0];
    for fraction in TRAIN {
        // Don't retest.
        if fraction == TRAIN[0] {
            continue;
        }
        let name = format!("Fraction = {:.3}", fraction);
        info.fraction = fraction;
        experiment(&name, input_train, input_test, &info, hidden);
    }
}

// Performs a single experiment given starting conditions.
fn experiment(name: &str, input_train: &Input, input_test: &Input, info: &epoch::Info, hidden: usize) {
    let mut network = train_test::create_network
        ( INPUT, hidden, OUTPUT, LOW, HIGH );
    let result = train_test::epoch_set(&mut network, input_train, input_test, &CLASS, info);
    record(name, &result);
}

// Creates an graph image and .csv text file recording the results.
fn record(name: &str, result: &(epoch::Results, epoch::Results)) {
    match visuals::accuracy_graph_png(name, result) {
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
