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
use crate::constants::*;
use rand::{self, seq::SliceRandom};

type Input = (Array2<f32>, Array1<String>);

// MAIN
fn main() {
    let temp = read();
    let input;
    match temp {
        Ok(mut o) => {
            println!("Read successful.");
            o.0 /= DIVISOR;
            input = o;
        }
        Err(e) => {
            println!("{}", e);
            return;
        }
    }
    //_print_matrix(&input.0.view(), "INPUT");

    let _network = train(&input);

    println!("End program.\n");
}

fn train(input_raw: &Input) -> Network {
    let size = [
        LayerSize::new(HIDDEN, INPUT, STORAGE),
        LayerSize::new(OUTPUT, HIDDEN, STORAGE),
    ];
    let mut network = Network::new(&size[..], LOW, HIGH);
    let mut confusion = Array2::<u32>::zeros((OUTPUT, OUTPUT));
    let input = &input_raw.0;
    let targets = &input_raw.1;

    // Feed each input into the network and update the weights.
    for i in 0..targets.len() {
        let row = input.row(i);
        
        //println!("Output");
        let output = network.output(&row);
        //println!();
        
        //let target = input.1[i].parse::<f32>().unwrap();
        let target_str = &targets[i];
        let class_str = &classify(&output);
        println!("Prediction: {} from {}", class_str, target_str);
        let target = target_array(target_str);
        confusion[[class_to_index(target_str).unwrap(), class_to_index(class_str).unwrap()]] += 1;
        //print_data::_print_vector(&target.view(), "TARGET");
        //println!();

        //println!("Error");
        network.error(&target.view());
        //println!();

        //println!("Weight");
        network.weight(&row, LEARN, MOMENTUM);
        //println!();
    }
    print_data::_print_matrix(&confusion.view(), "CONFUSION");

    network
}


// Returns the target value needed for error calculation.
fn target_array(target: &str) -> Array1<f32> {
    let mut target_arr = Array1::<f32>::from_elem(OUTPUT, 0.1);
    
    let i = class_to_index(target);
    match i {
        Ok(o) => {
            target_arr[o] = 0.9;
        }
        Err(e) => {
            println!("{}", e);
        }
    }
    target_arr
}

// Finds the class that the model predicted.
fn classify(output: &ArrayView1<f32>) -> String {
    // Find the largest output value.
    let mut index = 0;
    let mut value = 0.;

    for i in 0..output.len() {
        //print!("{}, ", output[i]);
        if output[i] >= value {
            index = i;
            value = output[i];
        }
    }
    //println!();
    CLASS[index].to_string()
}

// Converts a class to an index value.
fn class_to_index(target: &str) -> Result<usize, &str> {
    for i in 0..OUTPUT {
        if target == CLASS[i] {
            return Ok(i);
        }
    }
    Err("No Match.")
}

// Creates a vector of indicies for randomizing the data.
fn _random_index(size: usize) -> Vec<usize> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());
    vec
}

// Reads the data file.
fn read() -> Result<Input, Box<dyn Error>> {
    let path_index = 0;
    let path = [
        "./data/mnist_test_short.csv",
        "./data/mnist_test.csv",
        "./data/mnist_train.csv",
        "./data/test.csv",
    ];
    println!("Reading data, please be patient...");
    read::read(path[path_index])
}
