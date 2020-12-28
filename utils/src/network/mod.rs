use crate::propagation::Propagation;
use crate::convolution::Conv3D;
use crate::pooling::Pool;
use crate::full_connected::FullLayer;
use crate::activation::Activation;
use crate::utils::utils::{compute_loss, evaluate};

use crate::trained::{convolution, pooling, activation, full_connected, Convert};

use ndarray::Array2;
use std::fmt::{self, Formatter};

use serde::{Serialize, Deserialize, Deserializer};

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::fmt::Debug;
use std::string::ToString;
use std::time::Instant;

macro_rules! timing {
    ($x: expr) => {
        {
            let start = Instant::now();
            let result = $x;
            let end = start.elapsed();
            println!("{}. {:03} sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
            result
        }
    };
}

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

    pub fn to_string(self) -> String {

        match self {
            Self::Conv(conv) => convolution::Conv3DJson::new(conv).to_string(),
            Self::Pool(p) => pooling::PoolJson::new(p).to_string(),
            Self::Activation(a) => activation::ActivationJson::new(a).to_string(),
            Self::Full(f) => full_connected::FullJson::new(f).to_string(),
        }
    }

    // pub fn convert(&self) -> Box<dyn Propagation> {

    //     match self {
    //         Self::Conv(conv) => Box::new(*conv),
    //         Self::Activation(a) => Box::new(*a),
    //         Self::Full(f) => Box::new(*f),
    //         Self::Pool(p) => Box::new(*p),
    //     }

    // }
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

#[derive(Serialize, Deserialize, Debug)]
pub struct NetworkGraph {
    pub conv1: convolution::Conv3DJson,
    pub activation1: activation::ActivationJson,
    pub pool1: pooling::PoolJson,

    pub conv2: convolution::Conv3DJson,
    pub activation2: activation::ActivationJson,
    pub pool2: pooling::PoolJson,
    
    pub full1: full_connected::FullJson,
    pub activation3: activation::ActivationJson,
    pub full2: full_connected::FullJson,
    pub activation: activation::ActivationJson
}

impl NetworkGraph {
    pub fn to_string(self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn to_layer(self) -> Vec<nn> {
        let mut network: Vec<nn> = vec![];

        network.push(nn::Conv(self.conv1.to_layer()));
        network.push(nn::Activation(self.activation1.to_layer()));
        network.push(nn::Pool(self.pool1.to_layer()));
        
        network.push(nn::Conv(self.conv2.to_layer()));
        network.push(nn::Activation(self.activation2.to_layer()));
        network.push(nn::Pool(self.pool2.to_layer()));

        network.push(nn::Full(self.full1.to_layer()));
        network.push(nn::Activation(self.activation3.to_layer()));
        network.push(nn::Full(self.full2.to_layer()));
        network.push(nn::Activation(self.activation.to_layer()));

        network
    }
}

pub fn forward(network: &Vec<nn>, input: &Vec<Vec<Array2<f32>>>) -> Vec<Vec<Vec<Array2<f32>>>> {
    let mut outputs = vec![input.clone()];

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
            nn::Pool(p) => p.backward(input, deltas),
            nn::Activation(a) => a.backward(input, deltas),
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

    for (i, layer) in network.into_iter().enumerate() {
        println!("saving [{:?}] layer...", i);
        file.write_all(&layer.to_string().as_bytes()).expect("failed to save current layer");
    }
}

pub fn train(
    network: &mut Vec<nn>, 
    epochs: usize, 
    inputs: Vec<Vec<Array2<f32>>>,
    test_inputs: Vec<Vec<Array2<f32>>>,
    train_target: Array2<f32>,
    test_target: Array2<f32>
) {
    //target [sample, 1 * 10]
    //inputs [sample, 1, 28 * 28]
    let samples = inputs.len() as f32;

    for epoch in 0..epochs {
        println!("******************************************");
        println!("Starting #{:?}# Epoch...", epoch);

        let mut outputs = forward(network, &inputs);
        let final_output = outputs.pop().unwrap();

        let loss = compute_loss(&final_output[0][0], &train_target);
        let deltas = vec![vec![final_output[0][0].clone() - &train_target]];
        let accuracy = evaluate(&final_output[0][0], &train_target) / samples;

        // let test_accuracy = predict(network, &test_inputs, &test_target);

        println!("Starting Backward...");
        backward(network, outputs, deltas);
        println!("Epoch#{:?}# loss: {:?}, Train-Acc: {:?}", epoch, loss, accuracy);

    }
}

pub fn predict(network: &mut Vec<nn>, test_inputs: &Vec<Vec<Array2<f32>>>, target: &Array2<f32>) -> f32 {

    let samples = test_inputs.len();

    let output = network.iter().fold(test_inputs.clone(), |out, layer| {
        layer.forward(&out)
    }); // [1, 1, sample * 10]

    evaluate(&output[0][0], &target) / samples as f32

}


pub fn train_one_by_one(
    network: &mut Vec<nn>, 
    epochs: usize, 
    inputs: Vec<Vec<Array2<f32>>>,
    target: Array2<f32>
) {
    let samples = inputs.len() as f32;

    for epoch in 0..epochs {
        println!("******************************************");
        println!("Starting #{:?}# Epoch...", epoch);

        let mut correct = 0.;
        let mut loss = 0.;

        timing!({
            for (i, input) in inputs.iter().enumerate() {
                let mut outputs = forward(network, &vec![input.clone()]);
                let final_output = outputs.pop().unwrap();
                
                let label = target.row(i).to_owned().into_shape((1, 10)).unwrap();
                loss += compute_loss(&final_output[0][0], &label);
                let deltas = vec![vec![final_output[0][0].clone() - &label]];
    
                correct += evaluate(&final_output[0][0], &label);
                backward(network, outputs, deltas);
            }
        });
        
        let train_accuracy = correct / samples;
        println!("Epoch#{:?}# Train-Acc: {:?} loss: {:?}", epoch, train_accuracy, loss);
    }
}