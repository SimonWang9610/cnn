
use crate::utils;
use utils::_flatten_withno_channel;
use utils::utils::{_softmax, _relu, relu_derivate};

use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use std::cell::{RefCell};

// boundary == out_channel
// boundary == 0 means this FC layer is not connected with convolution layer or pooling layer
pub struct FullLayer {
    pub neurons: usize,
    pub prev_neurons: usize,
    pub alpha: f32,
    pub boundary: usize,
    pub weights: RefCell<Array2<f32>>,
    pub bias: RefCell<Array2<f32>>,
}

impl FullLayer {
    fn initialization(neurons: usize, prev_neurons: usize) -> (Array2<f32>, Array2<f32>) {
        (
            Array::random((neurons, prev_neurons), StandardNormal) * 0.05,
            Array2::zeros((neurons, 1))
        )
    }
}

impl FullLayer {
    pub fn new(neurons: usize, prev_neurons: usize, alpha: f32, boundary: usize) -> FullLayer {
        let (weights, bias) = FullLayer::initialization(neurons, prev_neurons);

        FullLayer {
            neurons,
            prev_neurons,
            alpha,
            boundary,
            weights: RefCell::new(weights),
            bias: RefCell::new(bias)
        }
    }

    pub fn forward(&self, inputs: &Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        // inputs [sample, channel, input_width, input_width]
        // output [1, 1, sample, neurons]
        println!("FullConnected forwarding....");

        let flattened_input = if self.boundary != 0 {
            _flatten_withno_channel(inputs, self.prev_neurons)
        } else {
            inputs[0].to_owned()
        }; // [1, sample, channel * input_width * input_width]
        
        let z = self.weights.borrow().dot(&flattened_input[0].t()) + &*self.bias.borrow(); // [neurons, sample]
        vec![vec![z.reversed_axes()]]
    }


    pub fn backward(&self, inputs: Vec<Vec<Array2<f32>>>, next_deltas: Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        // input [1, 1, sample, prev_neurons] or [sample, channel, input_width, input_width]
        // next_deltas [1, 1, sample, neurons]

        let sample = next_deltas[0][0].shape()[0];

        let flattened_input = if self.boundary != 1 {
            _flatten_withno_channel(&inputs, self.prev_neurons)
        } else {
            inputs[0].to_owned()
        }; // [1, sample, channel * input_width * input_width]

        let derivate_weight = next_deltas[0][0].t().dot(&flattened_input[0]);
        let deltas = self.cal_delta(&next_deltas, sample);
        
        self.update(derivate_weight, &next_deltas[0][0], sample);
        
        deltas
    }
}

impl FullLayer {

    fn cal_delta(&self, next_delta: &Vec<Vec<Array2<f32>>>, sample: usize) -> Vec<Vec<Array2<f32>>> {
        // next delta [1, 1, sample, neurons]
        // delta [sample, prev_neurons]
        // output [sample, out_channel, input_width, input_width]
        let delta = next_delta[0][0].dot(&*self.weights.borrow());

        if self.boundary == 0 {
            vec![vec![delta]]
        } else {
            let width = (self.prev_neurons as f32 / self.boundary as f32).sqrt() as usize;
            let mut data_iter = delta.into_iter();

            (0..sample).map(|_| {
                (0..self.boundary).map(|_| {
                    let v =  (0..width * width).map(|_| *data_iter.next().unwrap()).collect::<Vec<f32>>();
                    Array2::from_shape_vec((width, width), v).unwrap()
                }).collect::<Vec<Array2<f32>>>()
            }).collect::<Vec<Vec<Array2<f32>>>>()
        }

    }

    fn update(&self, derivate_weight: Array2<f32>, derivate_bias: &Array2<f32>, sample: usize) {
        let cloned_weights = self.weights.borrow().clone();
        let cloned_bias = self.bias.borrow().clone();

        *self.weights.borrow_mut() = cloned_weights - self.alpha * derivate_weight /  self.neurons as f32;
        *self.bias.borrow_mut() = cloned_bias - self.alpha * derivate_bias.t().sum_axis(Axis(0)) / sample as f32;
    }
}