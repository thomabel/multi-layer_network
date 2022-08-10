use crate::layer::Layer;
use crate::layer_size::LayerSize;
use ndarray::prelude::*;

type Vector<'a> = ArrayView1<'a, f32>;

pub struct Network {
    layer: Vec<Layer>,
}
impl Network {
    pub fn new(size: &[LayerSize]) -> Network {
        let mut layer = Vec::with_capacity(size.len());
        for l in size {
            layer.push(Layer::new(*l));
        }
        for l in &mut layer {
            l._set_weights(0.1);
        }
        Network { layer }
    }

    // OUTPUT
    pub fn output(&mut self, input: &Vector) -> Vector {
        let mut iter = self.layer.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        layer_prev.output(input);
        
        for layer in iter {
            layer.output(&layer_prev.output_vector().view());
            layer_prev = layer;
        }

        layer_prev.output_vector().view()
    }

    // ERROR
    pub fn error(&mut self, target: f32) {
        let mut iter = self.layer.iter_mut().rev();
        let mut layer_prev = iter.next().unwrap();
        layer_prev.error_output(target);
        
        for layer in iter {
            layer.error_layer(layer_prev);
            layer_prev = layer;
        }
    }

    // WEIGHT
    pub fn weight(&mut self, input: &Vector, learn_rate: f32, momentum: f32) {
        let mut iter = self.layer.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        layer_prev.weight(input, learn_rate, momentum);
        
        for layer in iter {
            layer.weight(&layer_prev.output_vector().view(), learn_rate, momentum);
            layer_prev = layer;
        }
    }

}
