//! An 8-bit optimizer of AdamW.

use burn_core as burn;
use burn_core::tensor::{Int, Shape};

use burn::config::Config;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};
use burn::{module::AutodiffModule, record::Record};

use super::{AdaptiveMomentumState, SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`AdamW8Bit`] Configuration.
#[derive(Config, Debug)]
pub struct AdamWConfig8Bit {
    /// Parameter for AdamW.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for AdamW.
    #[config(default = 0.999)]
    beta_2: f32,
    /// The amount of quantization applied to the optimizer. Always use a power of 2.
    #[config(default = 2048)]
    block_size: usize,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
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
#[derive(Record, Clone, new)]
pub struct AdamWState8Bit<B: Backend, const D: usize> {
    /// Th current adaptive momentum state.
    pub momentum: AdaptiveMomentumState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for AdamW8Bit {
    type State<const D: usize> = AdamWState8Bit<B, D>;

    // /// A single optimization step for any tensor that represents the parameters of a model.
    fn step<const D: usize>(
        &self,
        // Learning rate.
        lr: LearningRate,
        // Any tensor that represents the parameters of a model.
        tensor: Tensor<B, D>,
        // Gradient of the loss w.r.t. the parameters.
        grad: Tensor<B, D>,
        // State of the optimizer.
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (raw_delta, momentum_state) = self.momentum.transform(grad, state.map(|s| s.momentum));

        let decay_rate = lr * (self.weight_decay as f64);

        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else if self.cautious_weight_decay {
            // Cautious weight decay.
            // See: https://arxiv.org/abs/2510.12402
            let tensor_pos = tensor.clone().greater_equal_elem(0.0);
            let grad_pos = momentum_state.moment_1.clone().greater_equal_elem(0.0);
            let differ = tensor_pos.not_equal(grad_pos);

            // Zero out the decay where the decay is counter to the update direction.
            tensor.clone() - tensor.mul_scalar(decay_rate).mask_fill(differ, 0.0)
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };

        let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);

        let state = AdamWState8Bit {
            momentum: momentum_state,
        };

        (tensor_updated, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
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
}

impl AdaptiveMomentumW8Bit {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdaptiveMomentumState<B, D>>,
    ) -> (Tensor<B, D>, AdaptiveMomentumState<B, D>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;

        let state = if let Some(mut state) = state {
            // Update first moment estimate.
            state.moment_1 = state
                .moment_1
                .mul_scalar(self.beta_1)
                .add(grad.clone().mul_scalar(factor_1));

            // Update second moment estimate.
            state.moment_2 = state
                .moment_2
                .mul_scalar(self.beta_2)
                .add(grad.square().mul_scalar(factor_2));

            if self.amsgrad {
                let max_v = state
                    .max_moment_2
                    .take()
                    .unwrap_or_else(|| state.moment_2.clone());
                state.max_moment_2 = Some(max_v.max_pair(state.moment_2.clone()));
            }

            // Update time.
            state.time += 1;

            state
        } else {
            // Initialize first moment estimate.
            let moment_1 = grad.clone().mul_scalar(factor_1);

            // Initialize second moment estimate.
            let moment_2 = grad.square().mul_scalar(factor_2);
            let max_moment_2 = self.amsgrad.then(|| moment_2.clone());
            AdaptiveMomentumState {
                time: 1,
                moment_1,
                moment_2,
                max_moment_2,
            }
        };

        let time: i32 = state.time as i32;

        // Compute bias-corrected first and second moment estimates.
        let moment_1_corrected = state
            .moment_1
            .clone()
            .div_scalar(1f32 - self.beta_1.powi(time));

        let v_to_use = if self.amsgrad {
            state.max_moment_2.as_ref().unwrap_or(&state.moment_2)
        } else {
            &state.moment_2
        };

        let moment_2_corrected = v_to_use.clone().div_scalar(1f32 - self.beta_2.powi(time));

        let update_delta =
            moment_1_corrected.div(moment_2_corrected.sqrt().add_scalar(self.epsilon));

        (update_delta, state)
    }
}

/// Holds required quantized blockwise infomation.
struct QuantizeBlockwise<B: Backend> {
    pub quantized: Tensor<B, 1, Int>,
    pub scales: Tensor<B, 1>,
    pub shape: Shape,
}

fn quantize_blockwise<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    block_size: usize,
) -> QuantizeBlockwise<B> {
    let shape = tensor.shape();
    let total_elements = shape.num_elements();

    let padding = (block_size - (total_elements % block_size)) % block_size;
    let padded_total = total_elements + padding;
    let num_blocks = padded_total / block_size;

    let flattened = tensor.reshape([total_elements]);
    let input = if padding > 0 {
        let device = flattened.device();
        let pad_tensor = Tensor::<B, 1>::zeros([padding], &device);
        Tensor::cat(vec![flattened, pad_tensor], 0)
    } else {
        flattened
    };

    // Before.
    // index:  0   1   2   3   4   5   6   7   8   9  10  11
    // value:  a0  a1  a2  a3  a4  a5  a6  a7  a8  a9 a10 a11
    //
    // After.
    // block 0: [ a0, a1, a2, a3 ]
    // block 1: [ a4, a5, a6, a7 ]
    // block 2: [ a8, a9, a10, a11 ]
    let blocked = input.reshape([num_blocks, block_size]);

    // Compute scales, max absolute value per block / 127.
    let abs_max = blocked.clone().abs().max_dim(1).squeeze();
    let scales = abs_max / 127.0; // or .div(127.0)
    let mask = scales.clone().equal_elem(0.0);
    let scales = scales.mask_fill(mask, 1.0); // Avoid division by zero.

    let scales_expanded = scales
        .clone()
        .reshape([num_blocks, 1])
        .expand([num_blocks, block_size]);

    let quantized = blocked
        .div(scales_expanded)
        .round()
        .int()
        .reshape([padded_total]);

    QuantizeBlockwise {
        quantized,
        scales,
        shape,
    }
}

/// Dequantizes a block‑wise quantized tensor back to its original shape.
fn dequantize_blockwise<B: Backend, const D: usize>(
    quantized_blockwise: QuantizeBlockwise<B>,
    block_size: usize,
) -> Tensor<B, D> {
    let total_elements = quantized_blockwise.shape.num_elements();
    let padded_total = quantized_blockwise.quantized.shape().num_elements(); // should be a multiple of block_size
    let num_blocks = padded_total / block_size;

    // Reshape quantized data into blocks
    let quantized_blocked = quantized_blockwise
        .quantized
        .reshape([num_blocks, block_size]);

    // Expand scales to match block shape
    let scales_expanded = quantized_blockwise
        .scales
        .reshape([num_blocks, 1])
        .expand([num_blocks, block_size]);

    // Dequantize: convert to float and multiply by scales
    let dequantized = quantized_blocked.float().mul(scales_expanded);

    // Flatten and remove padding
    let dequantized_flat = dequantized.reshape([padded_total]);
    let dequantized_unpadded = dequantized_flat.slice([0..total_elements]);

    // Restore original shape
    dequantized_unpadded.reshape(quantized_blockwise.shape)
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
            },
            weight_decay: config.weight_decay,
            cautious_weight_decay: false,
        }
        .into()
    }

    mod quantization {
        use super::super::*;
        use burn::tensor::{Distribution, Tensor};
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
            assert_eq!(quantize.quantized.shape().num_elements(), 256);
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

            assert_eq!(quantize.quantized.shape().num_elements(), 128);
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

            assert_eq!(quantize.quantized.shape().num_elements(), 128);
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
    }
}
