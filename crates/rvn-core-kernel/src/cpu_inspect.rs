// 1. Softmax Entropy
// If softmax is too peaky → model is overconfident
// If too flat → model lost focus (junk Q/K)
// If entropy ≈ 0 → token output will loop or collapse
// If entropy ≈ log(n_tokens) → token doesn’t attend to anything meaningfully
pub fn entropy(p: &[f32]) -> f32 {
    p.iter()
        .cloned()
        .filter(|&x| x > 0.0)
        .map(|x| -x * x.ln())
        .sum()
}

// 2. Q/K/V Norms
// Vanishing norms? Input hidden state died.
// Exploding norms? Model is unstable post-conversion.
// Use on q, each row of k, and each row of v
pub fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// 3. Cosine Similarity Between Q and K
// Flat dot products? Maybe no directional info in attention
// diagnosis: Low cosine = query doesn't match any key = attention misses
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    dot / (norm_a * norm_b + 1e-9)
}

// 4. Value Vector Corruption Detection
// If v rows are all very close to each other → attention output is insensitive to weighting
// Use variance/stddev across v rows

/*
🧠 token: "sky" (ID: 4023)
Q norm = 0.001 ❗ (too low)
K[0] norm = 0.012
K[1] norm = 0.014
cos(Q, K[0]) = 0.002 ❗
softmax entropy = 0.001 ❗
attention output norm = 0.002
→ [FAIL] Token failed to focus
*/
