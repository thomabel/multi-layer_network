
use ndarray::prelude::*;
use rand::Rng;
use crate::network::layer_size::LayerSize;

type Vector = Array1<f32>;
type VectorView<'a> = ArrayView1<'a, f32>;
type Matrix = Array2<f32>;

pub struct Layer {
    size:   LayerSize,
    output: Vector,
    error:  Vector,
    weight: Matrix,
    delta:  Matrix,
}
impl Layer {
    pub fn new(size: LayerSize) -> Layer {
        let output = Array1::<f32>::zeros(size.output);
        let error = Array1::<f32>::zeros(size.output);
        let shape = (size.output, size.input + 1);
        let weight = Array2::<f32>::zeros(shape);
        let delta = Array2::<f32>::zeros(shape);
        
        Layer { size, output, error, weight, delta }
    }

    // Weight Setters
    pub fn weights_randomize(&mut self, low: f32, high: f32) {
        let distr = rand::distributions::Uniform::new_inclusive(low, high);
        let mut rng = rand::thread_rng();

        for w in &mut self.weight {
            *w = rng.sample(distr);
        }
    }
    pub fn _weights_set(&mut self, value: f32) {
        for w in &mut self.weight {
            *w = value;
        }
    }

    // Sigmoid function + derivative
    fn sigmoid(input: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-input))
    }
    fn sigmoid_derivative(input: f32, factor: f32) -> f32 {
        input * (1.0 - input) * factor
    }

    // Getters
    pub fn output_vector(&self) -> &Vector {
        &self.output
    }

    // Given some input vector, get output values for each node.
    pub fn output(&mut self, input: &VectorView) {
        // For each output node.
        for j in 0..self.size.output {
            let mut output = 0.;
            let weight = self.weight.row(j);
            // Weights * Inputs
            for i in 0..self.size.input {
                output += weight[i] * input[i];
            }
            // Bias
            output += weight[self.size.input];
            // Sigmoid
            output = Layer::sigmoid(output);
            self.output[j] = output;

            //print!("{}, ", self.output[j]);
        }
        //println!();
    }

    // Error functions
    pub fn error_output(&mut self, target: &VectorView) {
        for k in 0..self.size.output {
            self.error[k] = Layer::sigmoid_derivative(self.output[k], target[k] - self.output[k]);
            //print!("{}, ", self.error[k]);
        }
        //println!();
    }
    pub fn error_layer(&mut self, prev_layer: &Layer) {
        for j in 0..self.size.output {
            let mut factor = 0.;
            for k in 0..prev_layer.size.output {
                factor += prev_layer.weight[[k, j]] * prev_layer.error[k];
            }
            self.error[j] = Layer::sigmoid_derivative(self.output[j], factor);
            //print!("{}, ", self.error[j]);
        }
        //println!();
    }

    // Update weights based on error values.
    pub fn weight(&mut self, input: &VectorView, learn_rate: f32, momentum: f32) {
        // For each output node.
        for k in 0..self.size.output {
            for j in 0..self.size.input {
                let delta = learn_rate * self.error[k] * input[j] + momentum * self.delta[[k, j]];
                self.weight[[k, j]] += delta;
                self.delta[[k, j]] = delta;
                //print!("{}, ", self.weight[[k, j]]);
            }
            // Bias
            let j = self.size.input;
            let delta = learn_rate * self.error[k] * 1.0 + momentum * self.delta[[k, j]];
            self.weight[[k, j]] += delta;
            self.delta[[k, j]] = delta;
            //print!("{} \n", self.weight[[k, j]]);
        }
    }

}
