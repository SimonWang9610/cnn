
pub fn DATA_CHECK(input_width: usize, filter_width: usize, pad: usize, stride: usize) -> bool {
    if (input_width - filter_width + 2 * pad) % stride == 0 {
        true
    } else {
        false
    }
}