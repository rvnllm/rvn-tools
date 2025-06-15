/// Compute RMSNorm on a raw f32 slice with corresponding weights
pub fn rmsnorm_arr_f32(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    eps: f32,
) -> anyhow::Result<()> {
    if input.len() != output.len() || input.len() != weight.len() {
        anyhow::bail!("RMSNorm: size mismatch");
    }

    let input_len = input.len() as f32;

    let mean_square = input.iter().map(|&v| v * v).sum::<f32>() / input_len;

    let rms = (mean_square + eps).sqrt();

    for (out, (&x, &w)) in output.iter_mut().zip(input.iter().zip(weight.iter())) {
        *out = (x / rms) * w;
    }

    Ok(())
}
