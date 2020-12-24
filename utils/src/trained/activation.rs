use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::activation::Activation;

#[derive(Deserialize, Serialize)]
pub struct ActivationJson {
    pub end: usize
}

impl ActivationJson {
    pub fn new (activation: Activation) -> ActivationJson {
        ActivationJson {
            end: activation.end
        }
    }

    pub fn to_json(self) -> Value {
        
        json!({
            "Activation": {
                "end": self.end
            }
        })
    }

    pub fn to_layer(self) -> Activation {
        
        Activation {
            end: self.end
        }
    }
}