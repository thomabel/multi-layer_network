
use ndarray::prelude::*;
use crate::layer_size::LayerSize;

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

    // Sigmoid function + derivative
    fn sigmoid(input: f32) -> f32 {
        1.0 / (1.0 + f32::exp(-input))
    }
    fn sigmoid_derivative(input: f32, factor: f32) -> f32 {
        input * (1.0 - input) * factor
    }

    pub fn output_vector(&self) -> &Vector {
        &self.output
    }
    // 
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

            print!("{}, ", self.output[j]);
        }
        println!();
    }


}
