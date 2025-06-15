use anyhow;

// Takes two tensors a and b (both f32)
// Fills out with out[i] = a[i] + b[i]
// No allocations inside
// CPU version
// Clean memory

pub fn add_arr_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> anyhow::Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        anyhow::bail!("add_arr_f32: input/output length mismatch");
    }

    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }

    Ok(())
}
