//! An 8-bit optimizer of AdamW.

use burn_core as burn;
use burn_core::tensor::DType;

use burn::config::Config;
use burn::tensor::{Int, Shape};
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};
use burn::{module::AutodiffModule, record::Record};

use super::{AdaptiveMomentumState, SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// The amount of values we pack into one index of a ensor.
const PACKING_AMOUNT: usize = 2;

/// [`AdamW8Bit`] Configuration.
#[derive(Config, Debug)]
pub struct AdamWConfig8Bit {
    /// Parameter for AdamW.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for AdamW.
    #[config(default = 0.999)]
    beta_2: f32,
    /// The amount of quantization applied to the optimizer. Always use a power of 2, or have
    /// highly degraded performance.
    #[config(default = 256)]
    block_size: usize,
    /// A value required for numerical stability.
    #[config(default = 1e-3)]
    epsilon: f32,
    /// Weight decay config.
    #[config(default = 1e-4)]
    weight_decay: f32,

    /// Cautious weight decay config.
    ///
    /// See: <https://arxiv.org/abs/2510.12402>
    #[config(default = false)]
    cautious_weight_decay: bool,

    /// Whether to use AMSGrad algorithm
    #[config(default = false)]
    amsgrad: bool,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// AdamW optimizer.
///
/// See:
/// - [Decoupled Weight Decay Regularization, Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101).
/// - [Cautious Weight Decay, 2025](https://arxiv.org/abs/2510.12402)
/// - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
///
/// Configured by [`AdamWConfig`].
#[derive(Clone)]
pub struct AdamW8Bit {
    pub momentum: AdaptiveMomentumW8Bit,
    pub weight_decay: f32,
    pub cautious_weight_decay: bool,
}

/// AdamW state.
#[derive(Record, Clone)]
pub struct AdamWState8Bit<B: Backend, const D: usize> {
    pub time: usize,
    pub moment_1: QuantizeBlockwise<B, D>,
    pub moment_2: QuantizeBlockwise<B, D>,
    pub max_moment_2: Option<QuantizeBlockwise<B, D>>,
}

impl<B: Backend> SimpleOptimizer<B> for AdamW8Bit {
    type State<const D: usize> = AdamWState8Bit<B, D>;

    /// A single optimization step for any tensor that represents the parameters of a model.
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        // 1. Get the update delta, the new quantized state, and the dequantized m1
        let (raw_delta, new_state, m1) = self.momentum.transform(grad, state);

        let decay_rate = lr * (self.weight_decay as f64);

        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else if self.cautious_weight_decay {
            // C-Adam / Cautious weight decay logic
            let tensor_pos = tensor.clone().greater_equal_elem(0.0);

            // Use the dequantized m1 we just got from transform
            let grad_pos = m1.greater_equal_elem(0.0);
            let differ = tensor_pos.not_equal(grad_pos);

            // Apply decay only where it doesn't counter the update direction
            let decay = tensor.clone().mul_scalar(decay_rate).mask_fill(differ, 0.0);
            tensor.clone() - decay
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };

        let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);

        (tensor_updated, Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.moment_1 = state.moment_1.to_device(device);
        state.moment_2 = state.moment_2.to_device(device);
        state
    }
}

impl AdamWConfig8Bit {
    /// Initialize AdamW optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdamW8Bit, M, B> {
        let optim = AdamW8Bit {
            momentum: AdaptiveMomentumW8Bit {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
                amsgrad: self.amsgrad,
                block_size: self.block_size,
            },
            weight_decay: self.weight_decay,
            cautious_weight_decay: self.cautious_weight_decay,
        };

        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

#[derive(Clone)]
pub struct AdaptiveMomentumW8Bit {
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    pub amsgrad: bool,
    pub block_size: usize,
}

impl AdaptiveMomentumW8Bit {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdamWState8Bit<B, D>>, // Using 8-bit state
    ) -> (Tensor<B, D>, AdamWState8Bit<B, D>, Tensor<B, D>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;

        let (mut m1, mut m2, mut max_v, mut time) = if let Some(s) = state {
            (
                dequantize_blockwise::<B, D>(s.moment_1, self.block_size),
                dequantize_blockwise::<B, D>(s.moment_2, self.block_size),
                s.max_moment_2
                    .map(|m| dequantize_blockwise::<B, D>(m, self.block_size)),
                s.time + 1,
            )
        } else {
            (
                Tensor::zeros(grad.shape(), &grad.device()),
                Tensor::zeros(grad.shape(), &grad.device()),
                None,
                1,
            )
        };

        // --- Standard AdamW Logic in Full Precision ---
        m1 = m1
            .mul_scalar(self.beta_1)
            .add(grad.clone().mul_scalar(factor_1));
        m2 = m2
            .mul_scalar(self.beta_2)
            .add(grad.square().mul_scalar(factor_2));

        let v_to_use = if self.amsgrad {
            let current_max = max_v.unwrap_or_else(|| m2.clone());
            let new_max = current_max.max_pair(m2.clone());
            max_v = Some(new_max.clone());
            new_max
        } else {
            m2.clone()
        };

        // Compute update delta
        let m1_corrected = m1.clone().div_scalar(1f32 - self.beta_1.powi(time as i32));
        let m2_corrected = v_to_use.div_scalar(1f32 - self.beta_2.powi(time as i32));
        let update_delta = m1_corrected.div(m2_corrected.sqrt().add_scalar(self.epsilon));

        // --- Re-quantize for Storage ---
        let state_8bit = AdamWState8Bit {
            time,
            moment_1: quantize_blockwise(m1.clone(), self.block_size),
            moment_2: quantize_blockwise(m2, self.block_size),
            max_moment_2: max_v.map(|m| quantize_blockwise(m, self.block_size)),
        };

        (update_delta, state_8bit, m1)
    }
}

/// Holds required quantized blockwise infomation.
#[derive(Record, Clone)]
struct QuantizeBlockwise<B: Backend, const D: usize> {
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

fn quantize_blockwise<B: Backend, const D: usize>(
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
            .slice([start_block * block_size..end_block * block_size]);
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

fn dequantize_blockwise<B: Backend, const D: usize>(
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
            .slice([start_block * packed_per_block..end_block * packed_per_block]);
        let scales_chunk = quantized_blockwise
            .scales
            .clone()
            .slice([start_block..end_block]);

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
        .slice([0..total_elements])
        .reshape(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use crate::{GradientsParams, Optimizer};
    use burn::module::{Module, Param};
    use burn::tensor::{Distribution, Tensor, TensorData};
    use burn::tensor::{Tolerance, ops::FloatElem};
    use burn_nn::{Linear, LinearConfig, LinearRecord};

    type FT = FloatElem<TestAutodiffBackend>;

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    #[ignore] // Failed every 1 out of 5 times.
    fn test_adamw_8bit_optimizer_save_load_state() {
        let device = Default::default();
        let linear = LinearConfig::new(6, 6).init(&device);
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let mut optimizer = create_adamw_8bit();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(LEARNING_RATE, linear, grads);

        #[cfg(feature = "std")]
        {
            use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

            BinFileRecorder::<FullPrecisionSettings>::default()
                .record(
                    optimizer.to_record(),
                    std::env::temp_dir().as_path().join("test_optim_adamw"),
                )
                .unwrap();
        }
        #[cfg(not(feature = "std"))]
        {
            use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

            let result = BinBytesRecorder::<FullPrecisionSettings>::default()
                .record(optimizer.to_record(), ())
                .unwrap();
            assert!(!result.is_empty());
        }

        let state_optim_before = optimizer.to_record();
        let state_optim_before_copy = optimizer.to_record();
        let optimizer = create_adamw_8bit();
        let optimizer = optimizer.load_record(state_optim_before_copy);
        let state_optim_after = optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }
    #[test]
    fn test_adamw_8bit_optimizer_with_amsgrad_50_steps() {
        let device = Default::default();
        let mut linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );

        let mut optimizer = AdamWConfig8Bit::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_amsgrad(true)
            .with_weight_decay(0.5)
            .init();

        for i in 1..=50 {
            let x = Tensor::<TestAutodiffBackend, 2>::ones([2, 6], &device)
                .mul_scalar(i as f32 * 0.1)
                .require_grad();

            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);
        }

        let state_updated = linear.into_record();
        let weight_updated = state_updated.weight.to_data();
        let bias_updated = state_updated.bias.unwrap().to_data();

        let weights_expected = TensorData::from([
            [
                -0.7822558283805847,
                -0.42578864097595215,
                -0.21805696189403534,
                -0.28366872668266296,
                -0.46587175130844116,
                -0.4805040955543518,
            ],
            [
                -0.4722539782524109,
                -0.5471276640892029,
                -0.8181359767913818,
                -0.33425918221473694,
                -0.3805687427520752,
                -0.7601516842842102,
            ],
            [
                -0.5475167632102966,
                -0.5057991743087769,
                -0.763265073299408,
                -0.3393959403038025,
                -0.7490996718406677,
                -0.28911691904067993,
            ],
            [
                -0.7646660208702087,
                -0.7050473093986511,
                -0.8218720555305481,
                -0.7647438049316406,
                -0.5919585227966309,
                -0.40617525577545166,
            ],
            [
                -0.27588561177253723,
                -0.7025567889213562,
                -0.24343004822731018,
                -0.6672990918159485,
                -0.23728127777576447,
                -0.556389570236206,
            ],
            [
                -0.5451040267944336,
                -0.5420684814453125,
                -0.4348171353340149,
                -0.3832150399684906,
                -0.5099242925643921,
                -0.23440153896808624,
            ],
        ]);
        let bias_expected = TensorData::from([
            -0.7473056316375732,
            -0.3745720386505127,
            -0.5188710689544678,
            -0.35184532403945923,
            -0.33705732226371765,
            -0.4332566559314728,
        ]);

        type FT = FloatElem<TestAutodiffBackend>;
        let tolerance = Tolerance::absolute(1e-5);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
    }
    #[test]
    fn test_adamw_8bit_optimizer_with_numbers() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );
        let device = Default::default();
        let x_1 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig8Bit::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = TensorData::from([
            [-0.337295, 0.117827, 0.380358, 0.296868, 0.065232, 0.046534],
            [
                0.057032, -0.036518, -0.382951, 0.232516, 0.173738, -0.309182,
            ],
            [
                -0.038703, 0.016052, -0.313155, 0.225982, -0.295039, 0.289981,
            ],
            [
                -0.314920, -0.237394, -0.387704, -0.315067, -0.095153, 0.141081,
            ],
            [
                0.306815, -0.234226, 0.348083, -0.191115, 0.356002, -0.049993,
            ],
            [-0.035634, -0.030083, 0.104636, 0.170244, 0.009196, 0.359580],
        ]);
        let bias_expected = TensorData::from([
            -0.406555, 0.067568, -0.115982, 0.096477, 0.115287, -0.007080,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        let tolerance = Tolerance::absolute(1e-2);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    #[test]
    fn test_adamw_8bit_optimizer_with_numbers_cautious() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );
        let device = Default::default();
        let x_1 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, -0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig8Bit::new()
            .with_cautious_weight_decay(true)
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = TensorData::from([
            [-0.337295, 0.117827, 0.380358, 0.296868, 0.065232, 0.046534],
            [
                0.057032, -0.036518, -0.382951, 0.232516, 0.173738, -0.309182,
            ],
            [
                -0.038703, 0.016052, -0.313155, 0.225982, -0.295039, 0.289981,
            ],
            [
                -0.314920, -0.237394, -0.387704, -0.315067, -0.095153, 0.141081,
            ],
            [
                0.306815, -0.234226, 0.348083, -0.191115, 0.356002, -0.049993,
            ],
            [
                -0.035634, -0.030083, 0.104636, 0.170244, 0.009196, 0.37061332,
            ],
        ]);
        let bias_expected = TensorData::from([
            -0.406555, 0.067568, -0.115982, 0.096477, 0.115287, -0.007080,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        let tolerance = Tolerance::absolute(1e-2);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    #[test]
    fn test_adamw_8bit_optimizer_no_nan() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );

        let x = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &Default::default(),
        )
        .require_grad();

        let mut optimizer = AdamWConfig8Bit::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x.clone()).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        assert!(!state_updated.weight.to_data().as_slice::<f32>().unwrap()[0].is_nan());
    }

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        LinearConfig::new(6, 6).init(&device).load_record(record)
    }

    fn create_adamw_8bit()
    -> OptimizerAdaptor<AdamW8Bit, Linear<TestAutodiffBackend>, TestAutodiffBackend> {
        let config = AdamWConfig8Bit::new();
        AdamW8Bit {
            momentum: AdaptiveMomentumW8Bit {
                beta_1: config.beta_1,
                beta_2: config.beta_2,
                epsilon: config.epsilon,
                amsgrad: config.amsgrad,
                block_size: config.block_size,
            },
            weight_decay: config.weight_decay,
            cautious_weight_decay: false,
        }
        .into()
    }

    mod quantization {
        use crate::{GradientsParams, optim::adamw_8bit::tests::given_linear_layer};

        use super::super::*;
        use crate::TestAutodiffBackend;
        use burn::tensor::{Distribution, Tensor};
        use burn_core::tensor::TensorData;
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

        use crate::optim::adamw_8bit::tests::LEARNING_RATE;
        use crate::optim::base::Optimizer;

        #[test]
        fn test_adamw_8bit_distribution_shift_nan_check() {
            let device = Default::default();

            // Corrected TensorData initialization
            let weight_data = TensorData::ones::<f32, _>([12, 12]);
            let bias_data = TensorData::ones::<f32, _>([12]);

            let mut linear = given_linear_layer(weight_data, bias_data);

            let mut optimizer = AdamWConfig8Bit::new()
                .with_epsilon(1e-8)
                .with_block_size(32)
                .init();

            // --- PHASE 1: The "Hot" Start (High Energy) ---
            for _ in 1..=20 {
                let x = Tensor::<TestAutodiffBackend, 2>::random(
                    [4, 12],
                    Distribution::Default,
                    &device,
                )
                .mul_scalar(50.0);
                let grads = linear.forward(x).backward();
                let grads = GradientsParams::from_grads(grads, &linear);
                linear = optimizer.step(LEARNING_RATE, linear, grads);
            }

            // --- PHASE 2: The "Cold" Convergence (Tiny Gradients) ---
            for step in 1..=100 {
                let x = Tensor::<TestAutodiffBackend, 2>::random(
                    [4, 12],
                    Distribution::Default,
                    &device,
                )
                .mul_scalar(0.001);
                let grads = linear.forward(x).backward();
                let grads = GradientsParams::from_grads(grads, &linear);
                linear = optimizer.step(LEARNING_RATE, linear, grads);

                // Check finite
                let weights = linear.weight.val().to_data();
                for (i, val) in weights.as_slice::<f32>().unwrap().iter().enumerate() {
                    if !val.is_finite() {
                        panic!(
                            "NaN/Inf detected at step {} (index {}). Value: {}",
                            step, i, val
                        );
                    }
                }
            }

            println!("Passed stability test without NaNs.");
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
}
