use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use std::cell::RefCell;
use ndarray::Array2;

use crate::full_connected::FullLayer;

#[derive(Deserialize, Serialize)]
pub struct FullJson {
    pub neurons: usize,
    pub prev_neurons: usize,
    pub alpha: f32,
    pub boundary: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>
}

impl FullJson {
    pub fn new(full: FullLayer) -> FullJson {

        FullJson {
            neurons: full.neurons,
            prev_neurons: full.prev_neurons,
            alpha: full.alpha,
            boundary: full.boundary,
            weights: full.weights.into_inner()
                .into_iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>(),
            bias: full.bias.into_inner()
                .into_iter()
                .map(|ele| *ele)
                .collect::<Vec<f32>>()
        }
    }

    pub fn to_json(self) -> Value {
        
        json!({
            "Full": {
                "neurons": self.neurons,
                "prev_neurons": self.prev_neurons,
                "alpha": self.alpha,
                "boundary": self.boundary,
                "weights": self.weights,
                "bias": self.bias
            }
        })
    }

    pub fn to_layer(self) -> FullLayer {
        let weights = Array2::from_shape_vec((self.neurons, self.prev_neurons), self.weights).unwrap();
        let bias = Array2::from_shape_vec((self.neurons, 1), self.bias).unwrap();

        FullLayer {
            neurons: self.neurons,
            prev_neurons: self.prev_neurons,
            alpha: self.alpha,
            boundary: self.boundary,
            weights: RefCell::new(weights),
            bias: RefCell::new(bias)
        }

    }
}