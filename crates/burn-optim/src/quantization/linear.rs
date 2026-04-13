//! Linear (Uniform) Quantization
//!
//! Maps values to 8-bit unsigned integers [0, 255] using a simple
//! linear mapping: quantized = round(value / scale) + 128
//!
//! This provides uniform absolute precision across the entire range,
//! making it well-suited for the second moment (v_t) in Adam which
//! stores non-negative squared gradients with a relatively narrow
//! dynamic range.

use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_core as burn;

/// Linearly quantize normalized [-1, 1] values to integer codes [0, 255].
///
/// mapping: code = round(value * 127) + 128
///   -1.0 -> 1
///    0.0 -> 128
///    1.0 -> 255
pub(crate) fn encode<B: Backend>(normalized: Tensor<B, 2>) -> Tensor<B, 2, Int> {
    normalized
        .mul_scalar(127.0)
        .round()
        .int()
        .add_scalar(128)
        .clamp(0, 255)
}

/// Dequantize integer codes [0, 255] back to normalized [-1, 1] values.
///
/// mapping: value = (code - 128) / 127
pub(crate) fn decode<B: Backend>(encoded: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    encoded.sub_scalar(128).float().div_scalar(127.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray;

    #[test]
    fn test_linear_roundtrip() {
        let device = Default::default();

        let values: Vec<f32> = vec![
            0.0, 0.5, -0.5, 1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 0.001, 0.0001, 0.00001, 0.95, 0.05,
            0.005, 0.123, -0.456, 0.789,
        ];
        let n = values.len();

        let input = Tensor::<B, 1>::from_floats(values.as_slice(), &device).reshape([1, n]);

        let encoded = encode::<B>(input.clone());
        let decoded = decode::<B>(encoded.clone());

        let input_data: Vec<f32> = input.to_data().to_vec().unwrap();
        let output_data: Vec<f32> = decoded.to_data().to_vec().unwrap();
        let encoded_data: Vec<i64> = encoded.to_data().to_vec().unwrap();

        println!();
        for i in 0..n {
            let orig = input_data[i];
            let reconstructed = output_data[i];
            let code = encoded_data[i];
            let err = if orig.abs() > 1e-7 {
                ((reconstructed - orig) / orig).abs() * 100.0
            } else {
                (reconstructed - orig).abs() * 100.0
            };
            println!(
                "  {:.7} -> {:.7}  (code: {:3}, err: {:.2}%)",
                orig, reconstructed, code, err
            );
        }
    }
}
