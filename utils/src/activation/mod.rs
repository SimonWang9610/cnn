use crate::utils;
use utils::utils::{_relu, _softmax};
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
}