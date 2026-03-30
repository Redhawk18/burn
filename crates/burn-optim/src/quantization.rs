use burn_core as burn;

use burn::record::Record;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Shape};

/// The amount of values we pack into one index of a ensor.
const PACKING_AMOUNT: usize = 2;

/// Holds required quantized blockwise infomation.
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
            shape: self.shape, // Arrays are already on the stack/copyable
        }
    }
}

const CHUNK_BLOCKS: usize = 1024;

pub(crate) fn quantize_blockwise<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    block_size: usize,
) -> QuantizeBlockwise<B, D> {
    let device = tensor.device();
    let shape = tensor.shape();
    let total_elements = shape.num_elements();

    // Ensure we can pack by 2; if block_size is odd, this might need adjustment,
    // but usually block_size is a power of 2.
    let padding = (block_size - (total_elements % block_size)) % block_size;
    let padded_total = total_elements + padding;
    let num_blocks = padded_total / block_size;

    let flattened = tensor.reshape([total_elements]);
    let input = if padding > 0 {
        let pad_tensor = Tensor::<B, 1>::zeros([padding], &device);
        Tensor::cat(vec![flattened, pad_tensor], 0)
    } else {
        flattened
    };

    let mut packed_chunks = Vec::new();
    let mut scale_chunks = Vec::new();

    for start_block in (0..num_blocks).step_by(CHUNK_BLOCKS) {
        let end_block = core::cmp::min(start_block + CHUNK_BLOCKS, num_blocks);
        let current_chunk_blocks = end_block - start_block;
        let elements_in_chunk = current_chunk_blocks * block_size;

        let chunk_input = input
            .clone()
            .slice(start_block * block_size..end_block * block_size);
        let blocked = chunk_input.reshape([current_chunk_blocks, block_size]);

        let abs_max = blocked.clone().abs().max_dim(1).squeeze_dims(&[1]);
        let chunk_scales = abs_max
            .clone()
            .div_scalar(127.0)
            .mask_fill(abs_max.clone().equal_elem(0.0), 1.0);

        let scales_expanded = chunk_scales
            .clone()
            .reshape([current_chunk_blocks, 1])
            .expand([current_chunk_blocks, block_size]);

        // Quantize to 0-255
        let quantized = blocked.div(scales_expanded).round().int().add_scalar(128);

        // --- 2-to-1 Packing Logic ---
        // We pack two 8-bit values into the lower 16 bits of an i32.
        let to_pack = quantized.reshape([elements_in_chunk / PACKING_AMOUNT, PACKING_AMOUNT]);

        let v0 = to_pack
            .clone()
            .slice([0..elements_in_chunk / PACKING_AMOUNT, 0..1])
            .squeeze_dims(&[1])
            .mul_scalar(256); // Shift left by 8 bits
        let v1 = to_pack
            .slice([0..elements_in_chunk / PACKING_AMOUNT, 1..2])
            .squeeze_dims(&[1]);

        packed_chunks.push(v0.add(v1));
        scale_chunks.push(chunk_scales);
    }

    QuantizeBlockwise {
        quantized: Tensor::cat(packed_chunks, 0),
        scales: Tensor::cat(scale_chunks, 0),
        shape: shape.dims(),
    }
}

pub(crate) fn dequantize_blockwise<B: Backend, const D: usize>(
    quantized_blockwise: QuantizeBlockwise<B, D>,
    block_size: usize,
) -> Tensor<B, D> {
    let shape = Shape::from(quantized_blockwise.shape);
    let total_elements = shape.num_elements();
    let num_blocks = quantized_blockwise.scales.shape().num_elements();
    let packed_per_block = block_size / PACKING_AMOUNT;

    let mut dequantized_chunks = Vec::new();

    for start_block in (0..num_blocks).step_by(CHUNK_BLOCKS) {
        let end_block = core::cmp::min(start_block + CHUNK_BLOCKS, num_blocks);
        let current_chunk_blocks = end_block - start_block;

        let packed_chunk = quantized_blockwise
            .quantized
            .clone()
            .slice(start_block * packed_per_block..end_block * packed_per_block);
        let scales_chunk = quantized_blockwise
            .scales
            .clone()
            .slice(start_block..end_block);

        // --- 2-to-1 Unpacking Logic ---
        let v0 = packed_chunk.clone().div_scalar(256);
        let v1 = packed_chunk.sub(v0.clone().mul_scalar(256));

        let unpacked = Tensor::stack::<2>(vec![v0, v1], 1)
            .reshape([current_chunk_blocks * block_size])
            .sub_scalar(128);

        let scales_expanded = scales_chunk
            .reshape([current_chunk_blocks, 1])
            .expand([current_chunk_blocks, block_size])
            .reshape([current_chunk_blocks * block_size]);

        dequantized_chunks.push(unpacked.float().mul(scales_expanded));
    }

    Tensor::cat(dequantized_chunks, 0)
        .slice(0..total_elements)
        .reshape(shape)
}

#[cfg(test)]
mod tests {

    use super::*;
    use burn::tensor::{Distribution, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_quantization_roundtrip_accuracy() {
        let device = Default::default();
        let block_size = 128;
        // 200 elements. Padded total should be 256 (2 blocks).
        let tensor = Tensor::<TestBackend, 1>::random([200], Distribution::Default, &device);

        let quantize = quantize_blockwise(tensor.clone(), block_size);

        // Verify internal dimensions
        assert_eq!(
            quantize.quantized.shape().num_elements(),
            256 / PACKING_AMOUNT
        ); // Packed by 4s.
        assert_eq!(quantize.scales.shape().num_elements(), 2);

        let dequantized: Tensor<TestBackend, 1> = dequantize_blockwise(quantize, block_size);

        assert_eq!(dequantized.shape().dims::<1>(), [200]);
        let diff = (tensor - dequantized)
            .abs()
            .max()
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(diff < 0.05);
    }

    #[test]
    fn test_zero_scale_nan_protection() {
        let device = Default::default();
        let block_size = 64;

        // Block 1: Random data, Block 2: All zeros
        let data_part = Tensor::<TestBackend, 1>::random([64], Distribution::Default, &device);
        let zero_part = Tensor::<TestBackend, 1>::zeros([64], &device);
        let tensor = Tensor::cat(vec![data_part, zero_part], 0);

        let quantize = quantize_blockwise(tensor.clone(), block_size);

        // Check scales: The second block should have been masked to 1.0 to prevent div by zero
        let scale_values = quantize
            .scales
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        assert!(
            scale_values[1] == 1.0,
            "Zero-scale mask failed: {}",
            scale_values[1]
        );

        let dequantized: Tensor<TestBackend, 1> = dequantize_blockwise(quantize, block_size);

        // Ensure no NaNs were produced in the dequantized output
        let has_nan = dequantized
            .is_nan()
            .any()
            .into_data()
            .as_slice::<bool>()
            .unwrap()[0];
        assert!(!has_nan, "Quantization produced NaNs in zero-valued blocks");
    }

    #[test]
    fn test_exact_multiple_no_padding() {
        let device = Default::default();
        let block_size = 64;
        // 128 elements. No padding needed.
        let tensor = Tensor::<TestBackend, 1>::random([128], Distribution::Default, &device);

        let quantize = quantize_blockwise(tensor.clone(), block_size);

        assert_eq!(
            quantize.quantized.shape().num_elements(),
            128 / PACKING_AMOUNT
        ); // Packed by 4s.
        assert_eq!(quantize.scales.shape().num_elements(), 2);

        let dequantized: Tensor<TestBackend, 1> = dequantize_blockwise(quantize, block_size);
        assert_eq!(dequantized.shape().dims::<1>(), [128]);
    }

    #[test]
    fn test_multi_dimensional_reshape() {
        let device = Default::default();
        let block_size = 32;
        // 10x10 = 100 elements. Padded total should be 128 (4 blocks).
        let tensor = Tensor::<TestBackend, 2>::random([10, 10], Distribution::Default, &device);

        let quantize = quantize_blockwise(tensor.clone(), block_size);

        assert_eq!(
            quantize.quantized.shape().num_elements(),
            128 / PACKING_AMOUNT
        ); // Packed by 4s
        assert_eq!(quantize.scales.shape().num_elements(), 4);

        let dequantized: Tensor<TestBackend, 2> = dequantize_blockwise(quantize, block_size);

        assert_eq!(dequantized.shape().dims::<2>(), [10, 10]);
    }

    #[test]
    fn test_block_isolation_outliers() {
        let device = Default::default();
        let block_size = 64;

        // Create a tensor where block 1 has a huge outlier (1000.0)
        // and block 2 has tiny values (0.01).
        // Block-wise quantization should preserve the precision of the tiny values.
        let mut data = vec![0.01f32; 128];
        data[0] = 1000.0;

        let tensor = Tensor::<TestBackend, 1>::from_floats(data.as_slice(), &device);
        let quantize = quantize_blockwise(tensor.clone(), block_size);
        let dequantized: Tensor<TestBackend, 1> = dequantize_blockwise(quantize, block_size);

        let low_precision_block = dequantized
            .clone()
            .slice([1..2])
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];

        let high_precision_block = dequantized
            .clone()
            .slice([65..66])
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];

        assert!(
            (high_precision_block - 0.01).abs() < 0.0001,
            "Block isolation failed: Block 2 precision lost"
        );
        assert!(
            (low_precision_block - 0.01).abs() > 0.0001,
            "Block 1 should have lower precision due to outlier"
        );
    }

    use burn::prelude::ToElement;

    #[test]
    fn test_packing_overflow_logic() {
        let device = Default::default();

        // We provide values that will be maxed out to 255
        let data = TensorData::from([127.0, 127.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &device);

        let quantized = quantize_blockwise(tensor, 2);

        let packed_values = quantized.quantized.into_data();

        // USE THE BACKEND'S TYPE: This solves the I64 vs I32 mismatch forever
        let slice = packed_values
            .as_slice::<<TestBackend as Backend>::IntElem>()
            .expect("Should match backend's integer type");

        for &val in slice {
            // We cast to i64 here just for the assertion comparison
            let val_i64 = val.to_i64();

            assert!(
                val_i64 >= 0,
                "Packed value is negative (overflowed sign bit): {}",
                val_i64
            );
            assert!(
                val_i64 <= 65535,
                "Value exceeds 16-bit packing range: {}",
                val_i64
            );
        }
    }
}
