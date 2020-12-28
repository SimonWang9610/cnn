use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt::Debug;

use crate::activation::Activation;
use crate::trained::Convert;

#[derive(Deserialize, Serialize, Debug)]
pub struct ActivationJson {
    pub end: usize
}

impl Convert<Activation, ActivationJson> for ActivationJson {
    fn new (activation: Activation) -> ActivationJson {
        ActivationJson {
            end: activation.end
        }
    }

    fn to_layer(self) -> Activation {
        
        Activation {
            end: self.end
        }
    }
}

impl ToString for ActivationJson {
    
    fn to_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}