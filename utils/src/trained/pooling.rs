use serde::{Deserialize, Serialize};
use serde_json;
use std::fmt::Debug;

use std::cell::RefCell;

use crate::pooling::Pool;
use crate::trained::Convert;

#[derive(Deserialize, Serialize, Debug)]
pub struct PoolJson {
    pub width: usize,
    pub stride: usize,
    pub padding: usize,
    pub boundary: usize,
    pub out_channel: usize,
    pub input_width: usize
}

impl Convert<Pool, PoolJson> for PoolJson {
    fn new(pool: Pool) -> PoolJson {
        PoolJson {
            width: pool.width,
            stride: pool.stride,
            padding: pool.padding,
            boundary: pool.boundary,
            out_channel: pool.out_channel,
            input_width: pool.input_width
        }
    }

    fn to_layer(self) -> Pool {
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

impl ToString for PoolJson {
    
    fn to_string(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}