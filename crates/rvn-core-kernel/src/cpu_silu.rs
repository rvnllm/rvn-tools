pub fn silu_f32(input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
    if input.len() != output.len() {
        anyhow::bail!("SiLU: input/output length mismatch");
    }

    for (i, &x) in input.iter().enumerate() {
        output[i] = x / (1.0 + (-x).exp());
    }

    Ok(())
}

#[test]
fn test_silu_values() {
    let x = [1.0, 0.0, -1.0];
    let mut y = [0.0; 3];
    silu_f32(&x, &mut y).unwrap();

    assert!((y[0] - 0.731).abs() < 1e-3);
    assert!((y[1] - 0.0).abs() < 1e-3);
    assert!((y[2] + 0.268).abs() < 1e-3);
}
