use crate::convolution::Conv3D;
use crate::pooling::Pool;
use crate::full_connected::FullLayer;
use crate::activation::Activation;

use ndarray::Array2;
use std::cell::{RefCell, Ref, RefMut};

pub enum nn {
    Conv(Conv3D),
    Pool(Pool),
    Activation(Activation),
    Full(FullLayer)
}

impl nn {
    pub fn new(name: String, config: Vec<usize>, alpha: f32) -> nn {
        
        if name == "Conv" {
            nn::Conv(Conv3D::new(config[0], config[1], config[2], config[3], config[4], config[5], alpha, config[6]))
        } else if name == "Pool" {
            nn::Pool(Pool::new(config[0], config[1], config[2], config[3], config[4], config[5]))
        } else if name == "Full" {
            nn::Full(FullLayer::new(config[0], config[1], alpha, config[2]))
        } else {
            nn::Activation(Activation::new(config[0]))
        }
    }

    pub fn forward(&self, input: &Vec<Vec<Array2<f32>>>) -> Vec<Vec<Array2<f32>>> {
        // inputs [sample, in_channel, width, width]
        // output [sample, out_channel, out_width, out_width]
        
        match self {
            Self::Conv(conv) => conv.forward(input),
            Self::Pool(p) => p.forward(input),
            Self::Activation(a) => a.forward(input),
            Self::Full(f) => f.forward(input),
        }
    }
}

pub fn forward(network: Vec<nn>, input: Vec<Vec<Array2<f32>>>) -> Vec<Vec<Vec<Array2<f32>>>> {
    let mut outputs = vec![input];

    for layer in network.iter() {
        let output = layer.forward(outputs.iter().last().unwrap());
        outputs.push(output);
    }
    outputs
}
