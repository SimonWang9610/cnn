use crate::convolution::Conv3D;
use crate::pooling::Pool;
use crate::full_connected::FullLayer;
use crate::activation::Activation;

use crate::trained::{convolution, pooling, activation, full_connected};

use ndarray::Array2;
use std::fmt::{self, Formatter};

use serde::Deserializer;
use serde_json::Value;

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

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

    /* pub fn backward(&self, outputs: &Vec<Vec<Array2<f32>>>, delta: Vec<Vec<Array2<f32>>>, layer: usize) -> Vec<Vec<Array2<f32>>> {
        
        match self {
            Self::Conv(conv) => conv.backward(outputs, delta),
            Self::Pool(p) => p.backward(delta),
            Self::Activation(a) => a.backward(delta),
            Self::Full(f) => f.backward(outputs, delta),
        }
    } */
}

impl fmt::Display for nn {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {

        let empty = Conv3D::new(1,1,1,1,28,2, 0.1,0);
        let text = match self {
            nn::Conv(conv) => conv,
            _ => &empty,
        };

        write!(
            f,
            "layer {}",
            text
        )
    }
}

pub fn forward(network: &Vec<nn>, input: Vec<Vec<Array2<f32>>>) -> Vec<Vec<Vec<Array2<f32>>>> {
    let mut outputs = vec![input];

    for layer in network.iter() {
        let output = layer.forward(outputs.iter().last().unwrap());
        outputs.push(output);
    }
    outputs
}

pub fn backward(network:&mut Vec<nn>, inputs: Vec<Vec<Vec<Array2<f32>>>>, output: Vec<Vec<Array2<f32>>>) {
    // inputs = outputs [0:-1]
    let mut deltas: Vec<Vec<Array2<f32>>> = output;
    for (layer, input) in network.iter_mut().zip(inputs.into_iter()).rev() {
        
        deltas = match layer {
            nn::Conv(conv) => conv.backward(input, deltas),
            nn::Pool(p) => p.backward(deltas),
            nn::Activation(a) => a.backward(deltas),
            nn::Full(f) => f.backward(input, deltas),
        };
        
    }
}

pub fn save(network: Vec<nn>, path: &str) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(Path::new(path))
        .unwrap();

    for layer in network.into_iter() {
        match layer {
            nn::Conv(conv) => {
                let layer_json = convolution::Conv3DJson::new(conv);
                file.write_all(&layer_json.to_json().to_string().as_bytes()).expect("Failed to save [Conv] layer");
            },
            nn::Pool(p) => {
                let layer_json = pooling::PoolJson::new(p);
                file.write_all(&layer_json.to_json().to_string().as_bytes()).expect("Failed to save [Pool] layer");
            },
            nn::Activation(a) => {
                let layer_json = activation::ActivationJson::new(a);
                file.write_all(&layer_json.to_json().to_string().as_bytes()).expect("Failed to save [Activation] layer");
            },
            nn::Full(f) => {
                let layer_json = full_connected::FullJson::new(f);
                file.write_all(&layer_json.to_json().to_string().as_bytes()).expect("Failed to save [Full] layer");
            }
        }
    }
}

// pub fn load(path: &str) -> Vec<nn> {
//     let mut network: Vec<nn> = vec![];

//     let mut data = String::new();
//     let mut file = File::open(Path::new(path)).unwrap();
//     file.read_to_string(&mut data).unwrap();

//     let stream = Deserializer::from_str(&data).into_iter::<Value>();
// }