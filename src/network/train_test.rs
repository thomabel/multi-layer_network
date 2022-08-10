use ndarray::prelude::*;
use rand::{self, seq::SliceRandom};
use crate::print_data;
use crate::constants::*;
use crate::network::layer_size::LayerSize;
use crate::network::multi_layer::MultiLayer;

/// Public
// Trains the network over a single epoch and returns the resulting confusion matrix.
pub fn train_epoch(network: &mut MultiLayer, input: &Array2<f32>, targets: &Array1<String>, ) -> Array2<u32> {
    // Create confusion matrix and random index array.
    let mut confusion = Array2::<u32>::zeros((OUTPUT, OUTPUT));
    let input_random = create_random_index(targets.len());

    // Feed each input into the network and update the weights.
    // Completing this is considered 1 epoch.
    for i in 0..targets.len() {
        // Set-up
        print!("{:>7}: ", i);
        let index = input_random[i];
        let row = input.row(index);
        let output = network.output(&row);
        
        // Printing things
        let target_str = &targets[index];
        let predict_str = &classify(&output);
        println!("[ {} ] => {}", target_str, predict_str);
        // Update the confusion matrix
        confusion[[class_to_index(target_str).unwrap(), class_to_index(predict_str).unwrap()]] += 1;
        
        // Find the error values and update the weights.
        let target = target_array(target_str);
        network.error(&target.view());
        network.weight(&row, LEARN, MOMENTUM);
    }

    confusion
}

// Create the network using some predefined layer sizes.
pub fn create_network() -> MultiLayer {
    let size = [
        LayerSize::new(HIDDEN, INPUT, STORAGE),
        LayerSize::new(OUTPUT, HIDDEN, STORAGE),
    ];
    MultiLayer::new(&size[..], LOW, HIGH)
}

// Print the confusion matrix and accuracy.
pub fn print_confusion(confusion: &Array2<u32>) {
    println!();
    print_data::_print_matrix(&confusion.view(), "CONFUSION");
    print_data::_print_total_error(confusion.diag().sum(), confusion.sum());
}


/// Private
// Creates a vector of indicies for randomizing the data.
fn create_random_index(size: usize) -> Array1<usize> {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());
    Array1::<usize>::from_vec(vec)
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
    //for (i, <item>) in CLASS.iter().enumerate().take(OUTPUT) {}
    for i in 0..OUTPUT {
        if target == CLASS[i] {
            return Ok(i);
        }
    }
    Err("No Match.")
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
