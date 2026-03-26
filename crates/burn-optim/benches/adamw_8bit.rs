use burn_core::{
    prelude::Backend as _,
    tensor::{Distribution, Tensor},
};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_optim::{AdamW8Bit, AdaptiveMomentumW8Bit, SimpleOptimizer};
use std::hint::black_box;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

type Backend = NdArray<f32>;

fn main() {
    let _profiler = dhat::Profiler::new_heap();

    let device = NdArrayDevice::default();
    Backend::seed(&device, 42);
    let shape = [1024, 1024];
    let lr = 0.001;

    let tensor = Tensor::<Backend, 2>::random(shape, Distribution::Default, &device);
    let grad = Tensor::<Backend, 2>::random(shape, Distribution::Default, &device);

    let momentum_config = AdaptiveMomentumW8Bit {
        beta_1: 0.9,
        beta_2: 0.999,
        epsilon: 1e-8,
        amsgrad: false,
        block_size: 2048,
    };

    let optim_std = AdamW8Bit {
        momentum: momentum_config.clone(),
        weight_decay: 0.01,
        cautious_weight_decay: false,
    };

    let optim_cautious = AdamW8Bit {
        momentum: momentum_config,
        weight_decay: 0.01,
        cautious_weight_decay: true,
    };

    let (updated, _) = optim_std.step(
        black_box(lr),
        black_box(tensor.clone()),
        black_box(grad.clone()),
        None,
    );
    let _ = black_box(updated.into_data());

    let (updated, _) = optim_cautious.step(
        black_box(lr),
        black_box(tensor.clone()),
        black_box(grad.clone()),
        None,
    );
    let _ = black_box(updated.into_data());
}
