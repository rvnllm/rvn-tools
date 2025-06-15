use anyhow;

use anyhow::Result;

/// In-place softmax
pub fn softmax_inplace_f32(arr: &mut [f32]) -> Result<()> {
    if arr.is_empty() {
        return Ok(());
    }

    let max_val = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0;
    for x in arr.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }

    for x in arr.iter_mut() {
        *x /= sum;
    }

    Ok(())
}

/// KL-style entropy — detects flat or collapsed softmax
pub fn compute_entropy(p: &[f32]) -> f32 {
    p.iter()
        .copied()
        .filter(|&x| x > 0.0)
        .map(|x| -x * x.ln())
        .sum()
}

/**
| Improvement                      | Why                                    |
| -------------------------------- | -------------------------------------- |
| `#[cfg(test)]` scoped module     | Keeps tests cleanly separated          |
| `assert!((sum - 1.0).abs() < ε)` | Confirms probability distribution      |
| Upper bound: `ln(N)`             | Makes entropy bound more precise       |
| Empty case                       | Prevents future panics                 |
| Prints for debug                 | Helps test trace if things go sideways |
 */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_entropy_range() {
        let mut probs = vec![1.0, 2.0, 3.0];
        softmax_inplace_f32(&mut probs).unwrap();

        let entropy = compute_entropy(&probs);

        println!("softmax probs = {:?}", probs);
        println!("entropy = {}", entropy);

        // Sanity checks
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Softmax output does not sum to 1");

        // Entropy bounds: [0.0, log(N)]
        assert!(
            entropy > 0.0 && entropy <= (3.0f32.ln()),
            "Entropy out of expected range"
        );
    }

    #[test]
    fn test_softmax_edge_case_empty() {
        let mut empty: Vec<f32> = vec![];
        let result = softmax_inplace_f32(&mut empty);
        assert!(result.is_ok());
    }

    #[test]
    fn test_softmax_entropy_collapse() {
        let mut logits = vec![100.0, 0.0, -100.0];
        softmax_inplace_f32(&mut logits).unwrap();

        let entropy = compute_entropy(&logits);
        println!("collapsed entropy = {}", entropy);

        assert!(entropy < 1e-3, "Softmax did not collapse as expected");
    }
}
