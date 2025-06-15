/// GELU activation (approximate) over a raw slice of f32
pub fn gelu_arr_f32(input: &[f32], output: &mut [f32]) -> anyhow::Result<()> {
    if input.len() != output.len() {
        anyhow::bail!("[kernel][gelu] length mismatch");
    }

    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = 0.5
            * x
            * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
    }

    Ok(())
}

#[test]
fn test_gelu_arr_f32_basic() {
    let input = vec![-1.0f32, 0.0, 1.0];
    let mut output = vec![0.0f32; input.len()];
    let expected = input
        .iter()
        .map(|&x| {
            0.5 * x
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        })
        .collect::<Vec<_>>();

    gelu_arr_f32(&input, &mut output).unwrap();

    for (i, (o, e)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (o - e).abs() < 1e-5,
            "Mismatch at index {}: got {}, expected {}",
            i,
            o,
            e
        );
    }
}
