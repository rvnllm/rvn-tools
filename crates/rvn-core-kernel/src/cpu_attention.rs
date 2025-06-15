use crate::cpu_inspect::{cosine_sim, entropy, l2_norm};

pub fn attention_scores_f32(
    q: &[f32],       // shape: [d_k]
    k: &[f32],       // shape: [n_tokens * d_k]
    out: &mut [f32], // shape: [n_tokens]
    d_k: usize,
    n_tokens: usize,
) -> anyhow::Result<()> {
    for t in 0..n_tokens {
        let mut sum = 0.0;
        for i in 0..d_k {
            sum += q[i] * k[t * d_k + i]; // row of k
        }
        out[t] = sum;
    }
    Ok(())
}

pub fn scale_scores(scores: &mut [f32], factor: f32) {
    for s in scores.iter_mut() {
        *s *= factor;
    }
}

pub fn attention_weighted_sum_f32(
    weights: &[f32], // shape: [n_tokens]
    v: &[f32],       // shape: [n_tokens * d_v]
    out: &mut [f32], // shape: [d_v]
    n_tokens: usize,
    d_v: usize,
) -> anyhow::Result<()> {
    for j in 0..d_v {
        let mut sum = 0.0;
        for t in 0..n_tokens {
            sum += weights[t] * v[t * d_v + j];
        }
        out[j] = sum;
    }
    Ok(())
}

/// Compute attention: q ⋅ kᵀ → softmax → weighted sum of v
// 1. q ⋅ kᵀ → scores
// 2. scale scores by 1 / sqrt(d_k)
// 3. softmax(scores)
// 4. scores ⋅ v → output
///
pub fn attention_forward(
    q: &[f32],       // shape: [d_k]
    k: &[f32],       // shape: [n_tokens * d_k]
    v: &[f32],       // shape: [n_tokens * d_v]
    out: &mut [f32], // shape: [d_v]
    d_k: usize,
    d_v: usize,
    n_tokens: usize,
) -> anyhow::Result<()> {
    // Step 1: Compute dot(q, k[t]) for all t
    let mut scores = vec![0.0f32; n_tokens];
    for t in 0..n_tokens {
        let mut sum = 0.0;
        for i in 0..d_k {
            sum += q[i] * k[t * d_k + i];
        }
        scores[t] = sum;
    }

    // Step 2: scale by 1/sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    for s in scores.iter_mut() {
        *s *= scale;
    }

    // Step 3: softmax
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }

    // Step 4: weighted sum: Σ softmax[t] * v[t]
    for j in 0..d_v {
        let mut acc = 0.0;
        for t in 0..n_tokens {
            acc += scores[t] * v[t * d_v + j];
        }
        out[j] = acc;
    }

    Ok(())
}

/*
 * [TRACE] Q norm = 0.9876, K[0] norm = 0.9023, cos(Q, K[0]) = 0.012
 * [TRACE] Softmax entropy = 0.0013
 * [TRACE] Attention output norm = 0.0021
 *
 * q_norm ≈ 0 → you're feeding junk to attention
 * cos(Q, K) ≈ 0 for all K → model isn't focusing on anything
 * entropy ≈ 0 → attention collapsed (bad Q/K or scale)
 * output_norm ≈ 0 → V vectors are garbage or canceled out
 **/
pub fn attention_forward_diagnostics(
    q: &[f32],       // shape: [d_k]
    k: &[f32],       // shape: [n_tokens * d_k]
    v: &[f32],       // shape: [n_tokens * d_v]
    out: &mut [f32], // shape: [d_v]
    d_k: usize,
    d_v: usize,
    n_tokens: usize,
) -> anyhow::Result<()> {
    // ----------------- 🔍 DIAGNOSTIC POINT #1 -------------------
    let q_norm = l2_norm(q);
    let k0_norm = l2_norm(&k[..d_k]);
    let cos = cosine_sim(q, &k[..d_k]);

    println!(
        "[TRACE] Q norm = {:.4}, K[0] norm = {:.4}, cos(Q, K[0]) = {:.4}",
        q_norm, k0_norm, cos
    );

    // ----------------- Step 1: q · kᵀ -------------------
    let mut scores = vec![0.0f32; n_tokens];
    for t in 0..n_tokens {
        let mut sum = 0.0;
        for i in 0..d_k {
            sum += q[i] * k[t * d_k + i];
        }
        scores[t] = sum;
    }

    // ----------------- Step 2: Scale -------------------
    let scale = 1.0 / (d_k as f32).sqrt();
    for s in scores.iter_mut() {
        *s *= scale;
    }

    // ----------------- Step 3: Softmax -------------------
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }

    // ----------------- 🔍 DIAGNOSTIC POINT #2 -------------------
    let attn_entropy = entropy(&scores);
    println!("[TRACE] Softmax entropy = {:.4}", attn_entropy);

    // ----------------- Step 4: Attention output -------------------
    for j in 0..d_v {
        let mut acc = 0.0;
        for t in 0..n_tokens {
            acc += scores[t] * v[t * d_v + j];
        }
        out[j] = acc;
    }

    // ----------------- 🔍 DIAGNOSTIC POINT #3 -------------------
    let output_norm = l2_norm(out);
    println!("[TRACE] Attention output norm = {:.4}", output_norm);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_forward() {
        let q = vec![1.0, 0.0]; // [d_k]
        let k = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            1.0, 1.0, // token 2
        ];
        let v = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ];

        let mut out = vec![0.0; 2];
        attention_forward(&q, &k, &v, &mut out, 2, 2, 3).unwrap();

        println!("out = {:?}", out);
        assert!(out.iter().any(|x| *x != 0.0));
    }
}
