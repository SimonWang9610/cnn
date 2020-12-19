use crate::utils;
use utils::utils::{_relu, _softmax, relu_derivate};
use ndarray::Array2;

pub struct Activation {
    pub end: usize
}

impl Activation {
    pub fn new(end: usize) -> Activation {
        Activation {
            end
        }
    }

    pub fn forward(&self, inputs: &Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        println!("output shape [{:?}, {:?}, {:?}]", inputs.len(), inputs[0].len(), inputs[0][0].shape());
        println!("Activation forwarding....");
        
        if self.end == 1 {
            inputs.iter().map(|input| {
                let mut z = input[0].to_owned();
                _softmax(&mut z)
            }).collect::<Vec<Vec<Array2<f32>>>>()
        } else {
            inputs.iter().map(|input| {
                input.iter().map(|arr| _relu(arr)).collect::<Vec<Array2<f32>>>()
            }).collect::<Vec<Vec<Array2<f32>>>>()
        }
    }

    pub fn backward(&self, deltas: Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        if self.end == 1 {
            deltas
        } else {
            deltas.into_iter().map(|delta| {
                delta.into_iter().map(|arr| relu_derivate(arr)).collect::<Vec<Array2<f32>>>()
            }).collect::<Vec<Vec<Array2<f32>>>>()
        }
    }
}