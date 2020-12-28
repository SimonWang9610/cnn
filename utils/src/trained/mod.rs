use serde_json;

pub mod convolution;
pub mod pooling;
pub mod full_connected;
pub mod activation;

pub trait Convert<T, U> {
    fn new(p: T) -> U;
    fn to_layer(self) -> T;
}
