use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use std::cell::RefCell;

use crate::pooling::Pool;

#[derive(Deserialize, Serialize)]
pub struct PoolJson {
    pub width: usize,
    pub stride: usize,
    pub padding: usize,
    pub boundary: usize,
    pub out_channel: usize,
    pub input_width: usize
}

impl PoolJson {
    pub fn new(pool: Pool) -> PoolJson {
        PoolJson {
            width: pool.width,
            stride: pool.stride,
            padding: pool.padding,
            boundary: pool.boundary,
            out_channel: pool.out_channel,
            input_width: pool.input_width
        }
    }

    pub fn to_json(self) -> Value {
        json!({
            "Pool": {
                "width": self.width,
                "stride": self.stride,
                "padding": self.padding,
                "boundary": self.boundary,
                "out_channel": self.out_channel,
                "input_width": self.input_width
            }
        })
    }

    pub fn to_layer(self) -> Pool {
        Pool {
            width: self.width,
            stride: self.stride,
            padding: self.padding,
            boundary: self.boundary,
            out_channel: self.out_channel,
            input_width: self.input_width,
            positions: RefCell::new(vec![])
        }
    }
}