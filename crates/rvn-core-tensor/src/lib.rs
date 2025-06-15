use anyhow::{Result, anyhow, bail};
use once_cell::sync::{Lazy, OnceCell};
use rvn_core_kernel::cpu_add::add_arr_f32;
use rvn_core_kernel::cpu_attention::attention_scores_f32;
use rvn_core_kernel::cpu_attention::attention_weighted_sum_f32;
use rvn_core_kernel::cpu_attention::scale_scores;
use rvn_core_kernel::cpu_matmul::matmul_arr_f32;
use rvn_core_kernel::cpu_rmsnorm::rmsnorm_arr_f32;
use rvn_core_kernel::cpu_softmax::softmax_inplace_f32;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::HashMap;

/// Tensor registry - initialized once, accessed many
static TENSOR_REGISTRY: Lazy<OnceCell<HashMap<TensorKind, Box<dyn TensorDecoder>>>> =
    Lazy::new(OnceCell::new);

pub fn register_tensor_registry_formats() {
    TENSOR_REGISTRY.get_or_init(|| {
        let mut registry = HashMap::with_capacity(18); // Pre-size for known variants
        registry.insert(
            TensorKind::F32,
            Box::new(F32Format) as Box<dyn TensorDecoder>,
        );
        // TODO: Add other quantized formats here
        // registry.insert(TensorKind::Q4_0, Box::new(Q4_0Format));
        // registry.insert(TensorKind::Q2_K, Box::new(Q2_KFormat));
        registry
    });
}

/// A lightweight descriptor for one tensor blob inside a model file.
/// All fields are read straight from the model's index table; no heap
/// allocations other than `name`/`shape`.
#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    /// Human-readable identifier (e.g. `"blk.0.attn_q.weight"`).
    pub name: Cow<'a, str>,
    /// Quantisation / storage kind (enum-backed, not a bare `u32`).
    pub kind: TensorKind,
    /// Byte offset **from the beginning of the data section**.
    pub offset: u64,
    /// Size in *bytes* (already includes alignment padding).
    pub size: u64,
    /// Logical dimensions, outermost first.
    pub shape: Cow<'a, [u64]>,
}

/// Tensor View -> a view into the mmap
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: &'a [u8],        // Slice into mmap
    pub shape: Cow<'a, [u64]>, // Tensor dimensions
    pub dtype: DType,          // How to interpret bytes
}
// Macro-generated quantized slice accessors
macro_rules! impl_quant_view {
    ($fn_name:ident, $variant:ident) => {
        pub fn $fn_name(&self) -> Result<&[u8]> {
            if self.dtype != DType::$variant {
                bail!("TensorView is not {}", stringify!($variant));
            }
            if self.expected_byte_len() != self.data.len() {
                bail!("TensorView length mismatch for {}", stringify!($variant));
            }
            Ok(self.data)
        }
    };
}
impl<'a> TensorView<'a> {
    #[inline(always)]
    pub const fn element_size(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
            // Packed quant types—all addressable in 1 byte.
            DType::Q4_0
            | DType::Q4_1
            | DType::Q5_0
            | DType::Q5_1
            | DType::Q8_0
            | DType::Q8_1
            | DType::Q2_K
            | DType::Q3_K_S
            | DType::Q3_K_M
            | DType::Q3_K_L
            | DType::Q4_K_S
            | DType::Q4_K_M
            | DType::Q5_K_S
            | DType::Q5_K_M
            | DType::Q6_K => 1,
        }
    }

    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&x| x as usize).product()
    }

    #[inline(always)]
    pub fn expected_byte_len(&self) -> usize {
        self.num_elements() * self.element_size()
    }

    /// F32 slice - zero-copy, alignment checked
    pub fn as_f32_slice(&self) -> Result<&'a [f32]> {
        if self.dtype != DType::F32 {
            bail!("Tensor is not f32");
        }

        if self.data.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0 {
            bail!("Tensor data is not aligned for f32");
        }

        let expected_len = self.expected_byte_len();
        if self.data.len() != expected_len {
            bail!(
                "Tensor data length mismatch: got {}, expected {}",
                self.data.len(),
                expected_len
            );
        }

        let ptr = self.data.as_ptr().cast::<f32>();
        // SAFETY: We've checked alignment, length, and data type
        unsafe { Ok(std::slice::from_raw_parts(ptr, self.num_elements())) }
    }

    pub fn as_i8_slice(&self) -> Result<&'a [i8]> {
        if self.dtype != DType::I8 {
            bail!("Tensor is not i8");
        }

        if self.expected_byte_len() != self.data.len() {
            bail!("Tensor data length mismatch");
        }

        let ptr = self.data.as_ptr().cast::<i8>();
        // SAFETY: i8 has no alignment requirements, length checked above
        unsafe { Ok(std::slice::from_raw_parts(ptr, self.num_elements())) }
    }

    impl_quant_view!(as_q4_0_slice, Q4_0);
    impl_quant_view!(as_q4_1_slice, Q4_1);
    impl_quant_view!(as_q5_0_slice, Q5_0);
    impl_quant_view!(as_q5_1_slice, Q5_1);
    impl_quant_view!(as_q8_0_slice, Q8_0);
    impl_quant_view!(as_q8_1_slice, Q8_1);
    impl_quant_view!(as_q2_k_slice, Q2_K);
    impl_quant_view!(as_q3_k_s_slice, Q3_K_S);
    impl_quant_view!(as_q3_k_m_slice, Q3_K_M);
    impl_quant_view!(as_q3_k_l_slice, Q3_K_L);
    impl_quant_view!(as_q4_k_s_slice, Q4_K_S);
    impl_quant_view!(as_q4_k_m_slice, Q4_K_M);
    impl_quant_view!(as_q5_k_s_slice, Q5_K_S);
    impl_quant_view!(as_q5_k_m_slice, Q5_K_M);
    impl_quant_view!(as_q6_k_slice, Q6_K);
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum DType {
    F32,
    F16,
    I8,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum TensorKind {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    Q4_0 = 3,
    Q4_1 = 4,
    Q5_0 = 5,
    Q5_1 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q2_K = 9,
    Q3_K_S = 10,
    Q3_K_M = 11,
    Q3_K_L = 12,
    Q4_K_S = 13,
    Q4_K_M = 14,
    Q5_K_S = 15,
    Q5_K_M = 16,
    Q6_K = 17,
    Unknown = 999, // fallback for exotic future kinds
}

impl From<u32> for TensorKind {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::Q4_0,
            4 => Self::Q4_1,
            5 => Self::Q5_0,
            6 => Self::Q5_1,
            7 => Self::Q8_0,
            8 => Self::Q8_1,
            9 => Self::Q2_K,
            10 => Self::Q3_K_S,
            11 => Self::Q3_K_M,
            12 => Self::Q3_K_L,
            13 => Self::Q4_K_S,
            14 => Self::Q4_K_M,
            15 => Self::Q5_K_S,
            16 => Self::Q5_K_M,
            17 => Self::Q6_K,
            _ => Self::Unknown,
        }
    }
}

pub type ShapeBuf = SmallVec<[u64; 6]>;

#[derive(Debug, Default)]
pub struct Header {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// decoder trait - zero allocation paths only
pub trait TensorDecoder: Send + Sync {
    fn id(&self) -> u32;
    fn name(&self) -> &'static str;
    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>>;
}

#[allow(non_camel_case_types)]
struct F32Format;

impl TensorDecoder for F32Format {
    #[inline(always)]
    fn id(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn name(&self) -> &'static str {
        "F32"
    }

    fn decode<'a>(&self, bytes: &'a [u8], shape: &[usize]) -> Result<TensorView<'a>> {
        // Zero-copy view construction - Memory Principle #1
        let shape_u64: ShapeBuf = shape.iter().map(|&d| d as u64).collect();
        Ok(TensorView {
            data: bytes,
            shape: Cow::Owned(shape_u64.into_vec()),
            dtype: DType::F32,
        })
    }
}

#[inline(always)]
fn get_format(kind: TensorKind) -> Option<&'static dyn TensorDecoder> {
    TENSOR_REGISTRY
        .get()
        .and_then(|m| m.get(&kind))
        .map(|boxed| &**boxed)
}

impl Tensor<'_> {
    /// View creation - bounds checked, zero-copy
    pub fn view<'a>(&self, blob: &'a [u8]) -> Result<TensorView<'a>> {
        let end = self.offset.saturating_add(self.size);

        if end as usize > blob.len() {
            bail!(
                "tensor '{}' slice out of bounds: end={} > blob_len={}",
                self.name,
                end,
                blob.len()
            );
        }

        let fmt =
            get_format(self.kind).ok_or_else(|| anyhow!("unknown tensor kind {:?}", self.kind))?;

        let shape_usize: Vec<usize> = self.shape.iter().map(|&d| d as usize).collect();
        let slice = &blob[self.offset as usize..end as usize];

        fmt.decode(slice, &shape_usize)
    }

    /// Debug view with corruption detection - only for embeddings
    #[cfg(debug_assertions)]
    pub fn view_debug<'a>(&self, blob: &'a [u8]) -> Result<TensorView<'a>> {
        if self.name.contains("token_embd") {
            self.validate_embedding_integrity(blob)?;
        }
        self.view(blob)
    }

   // #[cfg(not(debug_assertions))]
  //  #[inline(always)]
   // pub fn view<'a>(&self, blob: &'a [u8]) -> Result<TensorView<'a>> {
     //   self.view(blob)
   // }

    #[cfg(debug_assertions)]
    fn validate_embedding_integrity(&self, blob: &[u8]) -> Result<()> {
        // Embedding integrity validator - debug builds only
        if self.offset < 1024 {
            bail!(
                "token_embd offset {} too small! Likely reading header!",
                self.offset
            );
        }

        let end = self.offset.saturating_add(self.size);
        if end as usize > blob.len() {
            bail!(
                "token_embd extends beyond blob! offset={}, size={}, blob_len={}",
                self.offset,
                self.size,
                blob.len()
            );
        }

        // Additional validation could go here
        Ok(())
    }
}

// COMPUTE OPERATIONS - Zero allocation hot paths

/// Softmax - operates in-place, no allocations
pub fn softmax(input: &TensorView<'_>, output: &mut [f32]) -> Result<()> {
    if input.dtype != DType::F32 {
        bail!("softmax: only f32 supported");
    }

    if input.shape.len() != 1 {
        bail!("softmax: only 1D tensors supported");
    }

    if input.num_elements() != output.len() {
        bail!("softmax: output length mismatch");
    }

    let input_slice = input.as_f32_slice()?;

    // Find max for numerical stability
    let max_val = input_slice
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for (out_val, &in_val) in output.iter_mut().zip(input_slice) {
        *out_val = (in_val - max_val).exp();
        sum += *out_val;
    }

    // Normalize
    let inv_sum = 1.0 / sum;
    for out_val in output {
        *out_val *= inv_sum;
    }

    Ok(())
}

/// matmul - delegates to optimized kernel
pub fn matmul(a: &TensorView<'_>, b: &TensorView<'_>, output: &mut [f32]) -> Result<()> {
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        bail!("matmul: only f32 supported");
    }

    if a.shape.len() != 2 || b.shape.len() != 2 {
        bail!("matmul: only 2D tensors supported");
    }

    let (m, k1) = (a.shape[0] as usize, a.shape[1] as usize);
    let (k2, n) = (b.shape[0] as usize, b.shape[1] as usize);

    if k1 != k2 {
        bail!(
            "matmul: shape mismatch: a.shape[1]={}, b.shape[0]={}",
            k1,
            k2
        );
    }

    let expected_output_len = m * n;
    if output.len() != expected_output_len {
        bail!(
            "matmul: output length mismatch: got {}, expected {}",
            output.len(),
            expected_output_len
        );
    }

    let a_slice = a.as_f32_slice()?;
    let b_slice = b.as_f32_slice()?;

    // Delegate to optimized kernel - now with native usize
    _ = matmul_arr_f32(a_slice, b_slice, output, m, k1, n);

    Ok(())
}

/// Attention forward pass
pub fn attention_forward(
    q: &TensorView<'_>, // [1, d_k]
    k: &TensorView<'_>, // [n_tokens, d_k]
    v: &TensorView<'_>, // [n_tokens, d_v]
    output: &mut [f32], // [1, d_v]
) -> Result<()> {
    let d_k = q.shape[1] as usize;
    let n_tokens = k.shape[0] as usize;
    let d_v = v.shape[1] as usize;

    let q_slice = q.as_f32_slice()?;
    let k_slice = k.as_f32_slice()?;
    let v_slice = v.as_f32_slice()?;

    // Step 1: q • k.T → attn_scores
    let mut attn_scores = vec![0.0f32; n_tokens];
    attention_scores_f32(q_slice, k_slice, &mut attn_scores, d_k, n_tokens)?;

    // Step 2: scale by sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    scale_scores(&mut attn_scores, scale);

    // Step 3: softmax in-place
    softmax_inplace_f32(&mut attn_scores)?;

    // Step 4: weighted sum with values
    attention_weighted_sum_f32(&attn_scores, v_slice, output, n_tokens, d_v)?;

    Ok(())
}

/// RMSNorm
pub fn rmsnorm(
    input: &TensorView<'_>,
    weight: &TensorView<'_>,
    output: &mut [f32],
    eps: f32,
) -> Result<()> {
    if input.dtype != DType::F32 || weight.dtype != DType::F32 {
        bail!("rmsnorm: only f32 supported");
    }

    let input_elements = input.num_elements();
    let weight_elements = weight.num_elements();

    if weight_elements != input_elements || input_elements != output.len() {
        bail!(
            "rmsnorm: size mismatch - input: {}, weight: {}, output: {}",
            input_elements,
            weight_elements,
            output.len()
        );
    }

    let input_slice = input.as_f32_slice()?;
    let weight_slice = weight.as_f32_slice()?;

    // Aliasing check in debug builds only
    debug_assert_ne!(
        input_slice.as_ptr(),
        weight_slice.as_ptr(),
        "Aliased input and weight buffers in rmsnorm"
    );

    rmsnorm_arr_f32(input_slice, weight_slice, output, eps)
}

/// element-wise addition
pub fn add(a: &TensorView<'_>, b: &TensorView<'_>, output: &mut [f32]) -> Result<()> {
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        bail!("add: only f32 supported");
    }

    if a.shape != b.shape {
        bail!("add: shape mismatch");
    }

    if output.len() != a.num_elements() {
        bail!("add: output buffer size mismatch");
    }

    let a_slice = a.as_f32_slice()?;
    let b_slice = b.as_f32_slice()?;

    add_arr_f32(a_slice, b_slice, output)
}

#[cfg(test)]
mod tests {
    use crate::*;
    use crate::{DType, Tensor, TensorKind, TensorView, register_tensor_registry_formats};
    use std::borrow::Cow;
    // TENSOR TESTS - Zero-Copy Views

    #[test]
    fn test_tensor_view_f32_alignment() {
        // Create cache-aligned F32 data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };

        let tensor_view = TensorView {
            data: bytes,
            shape: Cow::Owned(vec![2, 3]), // 2x3 matrix
            dtype: DType::F32,
        };

        // validation
        assert_eq!(tensor_view.num_elements(), 6);
        assert_eq!(tensor_view.element_size(), 4);
        assert_eq!(tensor_view.expected_byte_len(), 24);

        // Zero-copy slice access
        let f32_slice = tensor_view.as_f32_slice().expect("F32 slice failed");
        assert_eq!(f32_slice.len(), 6);
        assert_eq!(f32_slice[0], 1.0);
        assert_eq!(f32_slice[5], 6.0);

        // Verify zero-copy (same memory address)
        assert_eq!(f32_slice.as_ptr(), data.as_ptr());
    }

    #[test]
    fn test_tensor_view_misaligned_f32_fails() {
        // Create guaranteed misaligned data by allocating aligned data and offsetting
        let aligned_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // Extra element for offset
        let aligned_bytes = unsafe {
            std::slice::from_raw_parts(aligned_data.as_ptr() as *const u8, aligned_data.len() * 4)
        };

        // Create misaligned slice by taking bytes 1..25 (guaranteed misaligned for f32)
        let misaligned_slice = &aligned_bytes[1..25]; // 24 bytes starting at offset 1

        // Verify it's actually misaligned
        assert_ne!(
            misaligned_slice
                .as_ptr()
                .align_offset(std::mem::align_of::<f32>()),
            0,
            "Test setup failed: slice should be misaligned"
        );

        let tensor_view = TensorView {
            data: misaligned_slice,
            shape: Cow::Owned(vec![2, 3]), // 6 elements
            dtype: DType::F32,
        };

        // memory protection - should reject misaligned data
        let result = tensor_view.as_f32_slice();
        assert!(
            result.is_err(),
            "Alignment check should reject misaligned data"
        );

        // Verify we get the expected error message
        let error_msg = result.unwrap_err().to_string();
        assert!(
            error_msg.contains("not aligned"),
            "Expected alignment error message"
        );
    }

    #[test]
    fn test_tensor_view_i8_access() {
        let data: Vec<i8> = vec![-1, 0, 1, 127, -128];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };

        let tensor_view = TensorView {
            data: bytes,
            shape: Cow::Owned(vec![5]),
            dtype: DType::I8,
        };

        let i8_slice = tensor_view.as_i8_slice().expect("I8 slice failed");
        assert_eq!(i8_slice.len(), 5);
        assert_eq!(i8_slice[0], -1);
        assert_eq!(i8_slice[4], -128);
    }

    #[test]
    fn test_tensor_kind_conversions() {
        // Test all variants convert correctly
        assert_eq!(TensorKind::from(0), TensorKind::F32);
        assert_eq!(TensorKind::from(1), TensorKind::F16);
        assert_eq!(TensorKind::from(17), TensorKind::Q6_K);
        assert_eq!(TensorKind::from(999), TensorKind::Unknown);
        assert_eq!(TensorKind::from(12345), TensorKind::Unknown); // Fallback
    }

    #[test]
    fn test_tensor_registry_initialization() {
        register_tensor_registry_formats();

        // Verify F32 decoder is registered
        let tensor = Tensor {
            name: Cow::Borrowed("test.weight"),
            kind: TensorKind::F32,
            offset: 0,
            size: 16,
            shape: Cow::Owned(vec![4]),
        };

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };

        let view = tensor.view(bytes).expect("View creation failed");
        assert_eq!(view.dtype, DType::F32);
        assert_eq!(view.data.len(), 16);
    }

    // COMPUTE TESTS - Zero Allocation Validation

    #[test]
    fn test_softmax_in_place() {
        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let input_bytes = unsafe {
            std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * 4)
        };

        let input_view = TensorView {
            data: input_bytes,
            shape: Cow::Owned(vec![4]),
            dtype: DType::F32,
        };

        let mut output = vec![0.0f32; 4];
        softmax(&input_view, &mut output).expect("Softmax failed");

        // Verify softmax properties
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax sum should be 1.0, got {}",
            sum
        );

        // Verify monotonic increase (since input was monotonic)
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1], "Softmax should preserve order");
        }
    }

    #[test]
    fn test_matmul_correctness() {
        // Test 2x3 * 3x2 = 2x2 matmul
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

        let a_bytes =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        let b_bytes =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        let a_view = TensorView {
            data: a_bytes,
            shape: Cow::Owned(vec![2, 3]),
            dtype: DType::F32,
        };
        let b_view = TensorView {
            data: b_bytes,
            shape: Cow::Owned(vec![3, 2]),
            dtype: DType::F32,
        };

        let mut output = vec![0.0f32; 4]; // 2x2
        crate::matmul(&a_view, &b_view, &mut output).expect("Matmul failed");

        // Expected: [22, 28, 49, 64]
        // A[0,:] * B[:,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // A[0,:] * B[:,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // A[1,:] * B[:,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // A[1,:] * B[:,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert_eq!(output, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_add_element_wise() {
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let b_data = [5.0f32, 6.0, 7.0, 8.0];

        let a_bytes =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        let b_bytes =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        let a_view = TensorView {
            data: a_bytes,
            shape: Cow::Owned(vec![4]),
            dtype: DType::F32,
        };
        let b_view = TensorView {
            data: b_bytes,
            shape: Cow::Owned(vec![4]),
            dtype: DType::F32,
        };

        let mut output = vec![0.0f32; 4];
        add(&a_view, &b_view, &mut output).expect("Add failed");

        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }
}
