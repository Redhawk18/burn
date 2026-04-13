use burn_core as burn;

use burn::record::Record;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Shape};

const PACKING_AMOUNT: usize = 2;
const PACK_SHIFT: i32 = 256_i32.pow(1); // 256

#[derive(Record, Clone)]
pub(crate) struct QuantizeBlockwise<B: Backend, const D: usize> {
    pub quantized: Tensor<B, 1, Int>,
    pub scales: Tensor<B, 1>,
    pub shape: [usize; D],
}

impl<B: Backend, const D: usize> QuantizeBlockwise<B, D> {
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            quantized: self.quantized.to_device(device),
            scales: self.scales.to_device(device),
            shape: self.shape,
        }
    }
}

pub fn quantize_blockwise<B: Backend, const D: usize, F>(
    tensor: Tensor<B, D>,
    block_size: usize,
    encode: F,
) -> QuantizeBlockwise<B, D>
where
    F: FnOnce(Tensor<B, 2>) -> Tensor<B, 2, Int>,
{
    debug_assert!(
        block_size % PACKING_AMOUNT == 0,
        "block_size ({}) must be divisible by PACKING_AMOUNT ({})",
        block_size,
        PACKING_AMOUNT
    );

    let device = tensor.device();
    let shape = tensor.shape();
    let total_elements = shape.num_elements();
    let padding = (block_size - (total_elements % block_size)) % block_size;

    let flattened = tensor.reshape([total_elements]);
    let input = if padding > 0 {
        let pad_tensor = Tensor::<B, 1>::zeros([padding], &device);
        Tensor::cat(vec![flattened, pad_tensor], 0)
    } else {
        flattened
    };

    let padded_total = total_elements + padding;
    let num_blocks = padded_total / block_size;

    let blocked = input.reshape([num_blocks, block_size]);

    // Per-block absolute maximum = DTQ scaling factor
    let abs_max = blocked.clone().abs().max_dim(1).squeeze_dims(&[1]);
    let scales = abs_max.clone().mask_fill(abs_max.equal_elem(0.0), 1.0);

    // Normalize to [-1, 1]
    let scales_expanded = scales
        .clone()
        .reshape([num_blocks, 1])
        .expand([num_blocks, block_size]);
    let normalized = blocked.div(scales_expanded);

    let quantized = encode(normalized);

    // Pack
    let flat = quantized.reshape([padded_total / PACKING_AMOUNT, PACKING_AMOUNT]);
    let v0 = flat
        .clone()
        .slice([0..(padded_total / PACKING_AMOUNT), 0..1])
        .squeeze_dims(&[1]);
    let v1 = flat
        .slice([0..(padded_total / PACKING_AMOUNT), 1..2])
        .squeeze_dims(&[1]);
    let packed = v0.mul_scalar(PACK_SHIFT).add(v1);

    QuantizeBlockwise {
        quantized: packed,
        scales,
        shape: shape.dims(),
    }
}

// pub(crate) fn dequantize_blockwise<B: Backend, const D: usize>(
pub fn dequantize_blockwise<B: Backend, const D: usize, F>(
    quantized_blockwise: QuantizeBlockwise<B, D>,
    block_size: usize,
    decode: F,
) -> Tensor<B, D>
where
    F: FnOnce(Tensor<B, 2, Int>) -> Tensor<B, 2>,
{
    let shape = Shape::from(quantized_blockwise.shape);
    let total_elements = shape.num_elements();
    let packed_len = quantized_blockwise.quantized.shape().num_elements();
    let padded_total = packed_len * PACKING_AMOUNT;
    let num_blocks = padded_total / block_size;

    // Unpack
    let packed = quantized_blockwise.quantized;
    let v0 = packed.clone().div_scalar(256);
    let v1 = packed - v0.clone().mul_scalar(256);
    let unpacked = Tensor::stack::<2>(vec![v0, v1], 1).reshape([num_blocks, block_size]);

    let decoded = decode(unpacked);

    // Apply per-block scales
    let scales_expanded = quantized_blockwise
        .scales
        .reshape([num_blocks, 1])
        .expand([num_blocks, block_size]);
    let dequantized = decoded.mul(scales_expanded);

    dequantized
        .reshape([padded_total])
        .slice(0..total_elements)
        .reshape(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type B = NdArray;

    use crate::quantization::signed_dynamic;
    use crate::quantization::unsigned_dynamic;

    #[test]
    fn test_blockwise_roundtrip_uniform() {
        let device = Default::default();
        let block_size = 256;

        // All same value — trivial case
        let input = Tensor::<B, 1>::full([512], 0.5, &device).reshape([2, 256]);
        let quantized = quantize_blockwise(input.clone(), block_size, signed_dynamic::encode);
        let recovered = dequantize_blockwise(quantized, block_size, signed_dynamic::decode);

        let input_data = input.clone().into_data();
        let recovered_data = recovered.clone().into_data();

        let input_vec: Vec<f32> = input_data.to_vec().unwrap();
        let recovered_vec: Vec<f32> = recovered_data.to_vec().unwrap();

        let max_err = input_vec
            .iter()
            .zip(recovered_vec.iter())
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Uniform 0.5 roundtrip max relative error: {:.4}%",
            max_err * 100.0
        );
        assert!(max_err < 0.02, "max relative error too high: {}", max_err);
    }

    #[test]
    fn test_blockwise_roundtrip_mixed() {
        let device = Default::default();
        let block_size = 256;

        // Mixed positive and negative values
        let data: Vec<f32> = (0..512)
            .map(|i| {
                let t = i as f32 / 511.0; // 0..1
                (t * 2.0 - 1.0) * 0.9 // -0.9..0.9
            })
            .collect();

        let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([2, 256]);
        let quantized = quantize_blockwise(input.clone(), block_size, signed_dynamic::encode);
        let recovered = dequantize_blockwise(quantized, block_size, signed_dynamic::decode);

        let input_data = input.into_data();
        let recovered_data = recovered.into_data();

        let input_vec: Vec<f32> = input_data.to_vec().unwrap();
        let recovered_vec: Vec<f32> = recovered_data.to_vec().unwrap();

        input_vec
            .iter()
            .zip(recovered_vec.iter())
            .enumerate()
            .filter(|(_, (a, _))| a.abs() > 0.01) // skip near-zero
            .for_each(|(i, (a, b))| {
                let rel_err = ((a - b) / a).abs() * 100.0;
                if rel_err > 5.0 {
                    println!("  [{i}] {a:.7} -> {b:.7}  (err: {rel_err:.2}%)");
                }
            });

        let max_err = input_vec
            .iter()
            .zip(recovered_vec.iter())
            .filter(|(a, _)| a.abs() > 0.01)
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Mixed roundtrip max relative error: {:.4}%",
            max_err * 100.0
        );
        assert!(max_err < 0.10, "max relative error too high: {}", max_err);
    }

    #[test]
    fn test_blockwise_roundtrip_unsigned() {
        let device = Default::default();
        let block_size = 256;

        // Unsigned — all non-negative, simulating moment 2
        let data: Vec<f32> = (0..512).map(|i| (i as f32 / 511.0) * 0.9 + 0.001).collect();

        let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device).reshape([2, 256]);
        let quantized = quantize_blockwise(input.clone(), block_size, unsigned_dynamic::encode);
        let recovered = dequantize_blockwise(quantized, block_size, unsigned_dynamic::decode);

        let input_data = input.into_data();
        let recovered_data = recovered.into_data();

        let input_vec: Vec<f32> = input_data.to_vec().unwrap();
        let recovered_vec: Vec<f32> = recovered_data.to_vec().unwrap();

        let max_err = input_vec
            .iter()
            .zip(recovered_vec.iter())
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f32, f32::max);

        println!(
            "Unsigned roundtrip max relative error: {:.4}%",
            max_err * 100.0
        );
        assert!(max_err < 0.05, "max relative error too high: {}", max_err);
    }
}
