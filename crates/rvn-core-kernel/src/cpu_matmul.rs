/// Naive CPU matmul on raw slices (f32), assumes row-major layout.
/// A: m x k, B: k x n, Output: m x n
pub fn matmul_arr_f32(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> anyhow::Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        anyhow::bail!("Input/output length mismatch");
    }

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    Ok(())
}
