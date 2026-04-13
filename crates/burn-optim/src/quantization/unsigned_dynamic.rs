//! [indicator (variable)] [fraction (variable)]

use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_core as burn;

/// Logical AND of two bool tensors.
fn bool_and<B: Backend>(
    a: Tensor<B, 2, burn::tensor::Bool>,
    b: Tensor<B, 2, burn::tensor::Bool>,
) -> Tensor<B, 2, burn::tensor::Bool> {
    a.int().mul(b.int()).greater_elem(0)
}

pub(crate) fn encode<B: Backend>(normalized: Tensor<B, 2>) -> Tensor<B, 2, Int> {
    let device = normalized.device();
    let [rows, cols] = normalized.shape().dims();

    // Second moment is always >= 0; no sign bit needed.
    // All 8 bits go to indicator + fraction, giving us one extra depth level.
    let abs_val = normalized; // no abs() needed, but harmless to leave if caller wants safety

    let is_zero = abs_val.clone().lower_elem(1e-7);

    let ge_d0 = abs_val.clone().greater_equal_elem(0.1);
    let ge_d1 = abs_val.clone().greater_equal_elem(0.01);
    let ge_d2 = abs_val.clone().greater_equal_elem(0.001);
    let ge_d3 = abs_val.clone().greater_equal_elem(0.0001);
    let ge_d4 = abs_val.clone().greater_equal_elem(0.00001);
    let ge_d5 = abs_val.clone().greater_equal_elem(0.000001);
    let ge_d6 = abs_val.clone().greater_equal_elem(0.0000001);

    let at_d0 = ge_d0.clone();
    let at_d1 = bool_and(ge_d1.clone(), ge_d0.clone().bool_not());
    let at_d2 = bool_and(ge_d2.clone(), ge_d1.clone().bool_not());
    let at_d3 = bool_and(ge_d3.clone(), ge_d2.clone().bool_not());
    let at_d4 = bool_and(ge_d4.clone(), ge_d3.clone().bool_not());
    let at_d5 = bool_and(ge_d5.clone(), ge_d4.clone().bool_not());
    let at_d6 = bool_and(ge_d6.clone(), ge_d5.clone().bool_not());
    let at_d7 = bool_and(is_zero.clone().bool_not(), ge_d6.bool_not());

    let zeros_f = Tensor::<B, 2>::zeros([rows, cols], &device);
    let zeros_i = Tensor::<B, 2, Int>::zeros([rows, cols], &device);

    // Upper bound per element: 10^(-depth)
    let upper = zeros_f
        .clone()
        .mask_fill(at_d0.clone(), 1.0)
        .mask_fill(at_d1.clone(), 0.1)
        .mask_fill(at_d2.clone(), 0.01)
        .mask_fill(at_d3.clone(), 0.001)
        .mask_fill(at_d4.clone(), 0.0001)
        .mask_fill(at_d5.clone(), 0.00001)
        .mask_fill(at_d6.clone(), 0.000001)
        .mask_fill(at_d7.clone(), 0.0000001);

    // Lower bound per element: 10^(-(depth+1))
    let lower = zeros_f
        .clone()
        .mask_fill(at_d0.clone(), 0.1)
        .mask_fill(at_d1.clone(), 0.01)
        .mask_fill(at_d2.clone(), 0.001)
        .mask_fill(at_d3.clone(), 0.0001)
        .mask_fill(at_d4.clone(), 0.00001)
        .mask_fill(at_d5.clone(), 0.000001)
        .mask_fill(at_d6.clone(), 0.0000001)
        .mask_fill(at_d7.clone(), 0.00000001);

    // Indicator bit value per depth — uses full 8 bits now
    let indicator = zeros_i
        .clone()
        .mask_fill(at_d0.clone(), 128)
        .mask_fill(at_d1.clone(), 64)
        .mask_fill(at_d2.clone(), 32)
        .mask_fill(at_d3.clone(), 16)
        .mask_fill(at_d4.clone(), 8)
        .mask_fill(at_d5.clone(), 4)
        .mask_fill(at_d6.clone(), 2)
        .mask_fill(at_d7.clone(), 1);

    // Max fraction value per depth = 2^fraction_bits - 1
    let max_frac = zeros_i
        .clone()
        .mask_fill(at_d0.clone(), 127)
        .mask_fill(at_d1.clone(), 63)
        .mask_fill(at_d2.clone(), 31)
        .mask_fill(at_d3.clone(), 15)
        .mask_fill(at_d4.clone(), 7)
        .mask_fill(at_d5.clone(), 3)
        .mask_fill(at_d6.clone(), 1)
        .mask_fill(at_d7.clone(), 0);

    let max_frac_f = zeros_f
        .clone()
        .mask_fill(at_d0, 127.0)
        .mask_fill(at_d1, 63.0)
        .mask_fill(at_d2, 31.0)
        .mask_fill(at_d3, 15.0)
        .mask_fill(at_d4, 7.0)
        .mask_fill(at_d5, 3.0)
        .mask_fill(at_d6, 1.0)
        .mask_fill(at_d7.clone(), 0.0);

    let range = upper - lower.clone();
    let is_tiny_range = range.clone().lower_elem(1e-10);
    let safe_range = range.mask_fill(is_tiny_range.clone(), 1.0);
    let t = (abs_val - lower)
        .div(safe_range)
        .clamp(0.0, 1.0)
        .mask_fill(is_tiny_range, 0.0);

    let fraction_f = t.mul(max_frac_f).round().clamp(0.0, 127.0);
    let fraction_raw = fraction_f.int();

    let over_max = fraction_raw.clone().greater(max_frac.clone());
    let fraction = fraction_raw.mask_where(over_max, max_frac);

    // Assemble: indicator + fraction (no sign contribution)
    let encoded = indicator + fraction;

    encoded.mask_fill(is_zero, 0)
}

pub(crate) fn decode<B: Backend>(encoded: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let device = encoded.device();
    let [rows, cols] = encoded.shape().dims();

    // True zero: encoded value is 0
    let is_zero = encoded.clone().equal_elem(0);

    // No sign bit — all 8 bits are indicator + fraction
    let ge_128 = encoded.clone().greater_equal_elem(128);
    let ge_64 = encoded.clone().greater_equal_elem(64);
    let ge_32 = encoded.clone().greater_equal_elem(32);
    let ge_16 = encoded.clone().greater_equal_elem(16);
    let ge_8 = encoded.clone().greater_equal_elem(8);
    let ge_4 = encoded.clone().greater_equal_elem(4);
    let ge_2 = encoded.clone().greater_equal_elem(2);

    let at_d0 = ge_128.clone();
    let at_d1 = bool_and(ge_64.clone(), ge_128.bool_not());
    let at_d2 = bool_and(ge_32.clone(), ge_64.bool_not());
    let at_d3 = bool_and(ge_16.clone(), ge_32.bool_not());
    let at_d4 = bool_and(ge_8.clone(), ge_16.bool_not());
    let at_d5 = bool_and(ge_4.clone(), ge_8.bool_not());
    let at_d6 = bool_and(ge_2.clone(), ge_4.bool_not());
    // at_d7: encoded == 1 (indicator=1, fraction=0)

    let zeros_f = Tensor::<B, 2>::zeros([rows, cols], &device);

    let upper = zeros_f
        .clone()
        .mask_fill(at_d0.clone(), 1.0)
        .mask_fill(at_d1.clone(), 0.1)
        .mask_fill(at_d2.clone(), 0.01)
        .mask_fill(at_d3.clone(), 0.001)
        .mask_fill(at_d4.clone(), 0.0001)
        .mask_fill(at_d5.clone(), 0.00001)
        .mask_fill(at_d6.clone(), 0.000001);

    let lower_f = zeros_f
        .clone()
        .mask_fill(at_d0.clone(), 0.1)
        .mask_fill(at_d1.clone(), 0.01)
        .mask_fill(at_d2.clone(), 0.001)
        .mask_fill(at_d3.clone(), 0.0001)
        .mask_fill(at_d4.clone(), 0.00001)
        .mask_fill(at_d5.clone(), 0.000001)
        .mask_fill(at_d6.clone(), 0.0000001);

    let indicator_val = Tensor::<B, 2, Int>::zeros([rows, cols], &device)
        .mask_fill(at_d0.clone(), 128)
        .mask_fill(at_d1.clone(), 64)
        .mask_fill(at_d2.clone(), 32)
        .mask_fill(at_d3.clone(), 16)
        .mask_fill(at_d4.clone(), 8)
        .mask_fill(at_d5.clone(), 4)
        .mask_fill(at_d6.clone(), 2);

    let fraction = encoded - indicator_val;

    let max_frac_f = zeros_f
        .clone()
        .mask_fill(at_d0, 127.0)
        .mask_fill(at_d1, 63.0)
        .mask_fill(at_d2, 31.0)
        .mask_fill(at_d3, 15.0)
        .mask_fill(at_d4, 7.0)
        .mask_fill(at_d5, 3.0)
        .mask_fill(at_d6, 1.0);

    let safe_denom = max_frac_f.clamp_min(1.0);
    let t = fraction.float().div(safe_denom);

    let range = upper - lower_f.clone();
    let abs_val = lower_f + t.mul(range);

    // No sign to apply — always non-negative
    abs_val.mask_fill(is_zero, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray;
    #[test]
    fn test_unsigned_dynamic_quantization_roundtrip() {
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
