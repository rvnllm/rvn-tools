use log::debug;
use rvn_core_parser::Model;
use rvn_core_parser::types::Value;

pub fn render_header(model: &Model) -> String {
    let h = model.header();
    format!(
        "Header\n  vendor: {}\n  version: {}\n  tensor_count: {}",
        h.vendor, h.version, h.tensor_count
    )
}

pub fn render_tensor(model: &Model, name: &str) -> String {
    match model.tensor(name) {
        Some(t) => format!("{}: shape={:?}, dtype={:?}", t.name, t.shape, t.kind),
        None => format!("Tensor '{}' not found", name),
    }
}

// this shit maybe -> or should go in debug inspect etc decide later
//pub fn render_vocab(model: &Model) {
/*debug!("render_vocab");
if let Some(Value::Array(tokens)) = model.metadata().get("tokenizer.ggml.tokens") {
    debug!("[DEBUG] Vocabulary ({} tokens):", tokens.len());
    for (i, token) in tokens.iter().enumerate() {
        match token {
            Value::String(s) => println!("{:>5}: {:?}", i, s),
            _ => println!("{:>5}: <invalid or non-string>", i),
        }
    }
} else {
    warn!("[WARN  No vocabulary found in tokenizer.ggml.tokens");
}*/
//}

pub fn render_metadata(model: &Model) {
    let mut total_keys = 0;
    let mut printed_keys = 0;
    let mut tokenizer_keys = 0;

    for (k, v) in model.metadata() {
        total_keys += 1;

        if let Some(key) = k.strip_prefix("tokenizer") {
            tokenizer_keys += 1;
            // Only show essential tokenizer metadata, not the huge vocab
            match key {
                ".ggml.bos_token_id" | ".ggml.eos_token_id" | ".ggml.pad_token_id" => {
                    println!("  {}: {}", k, v);
                    printed_keys += 1;
                }
                ".ggml.model" => {
                    match v {
                        Value::Bytes(bytes) => {
                            println!("  {}: <{} bytes>", k, bytes.len());
                        }
                        Value::String(s) => {
                            println!("  {}: \"{}\"", k, s);
                        }
                        _ => {
                            println!("  {}: {:?}", k, v);
                        }
                    }
                    printed_keys += 1;
                }
                _ => {
                    debug!("[VERBOSE] tokenizer metadata: {} => (filtered)", key);
                }
            }
            continue;
        }

        match v {
            Value::String(s) => {
                println!("  {}: \"{}\"", k, s);
                printed_keys += 1;
            }
            Value::Uint32(n) => {
                println!("  {}: {}", k, n);
                printed_keys += 1;
            }
            Value::Uint64(n) => {
                // <-- Add this missing case!
                println!("  {}: {}", k, n);
                printed_keys += 1;
            }
            Value::Float32(f) => {
                println!("  {}: {:.6}", k, f);
                printed_keys += 1;
            }
            Value::Float64(f) => {
                // <-- Add this missing case!
                println!("  {}: {:.6}", k, f);
                printed_keys += 1;
            }
            Value::Bool(b) => {
                println!("  {}: {}", k, b);
                printed_keys += 1;
            }
            Value::Int32(i) => {
                println!("  {}: {}", k, i);
                printed_keys += 1;
            }
            Value::Int64(i) => {
                // <-- Add this missing case!
                println!("  {}: {}", k, i);
                printed_keys += 1;
            }
            Value::Uint8(n) => {
                // <-- Add this missing case!
                println!("  {}: {}", k, n);
                printed_keys += 1;
            }
            Value::Int8(i) => {
                // <-- Add this missing case!
                println!("  {}: {}", k, i);
                printed_keys += 1;
            }
            Value::Uint16(n) => {
                // <-- Add this missing case!
                println!("  {}: {}", k, n);
                printed_keys += 1;
            }
            Value::Int16(i) => {
                // <-- Add this missing case!
                println!("  {}: {}", k, i);
                printed_keys += 1;
            }
            Value::Bytes(bytes) => {
                // <-- Add this missing case!
                println!("  {}: <{} bytes>", k, bytes.len());
                printed_keys += 1;
            }
            Value::Array(arr) => {
                println!("  {}: [array with {} elements]", k, arr.len());
                debug!("[filtered] {}: complex structure", k);
                printed_keys += 1;
            }
        }
    }

    println!();
    println!("📊 Metadata summary:");
    println!("  Total keys: {}", total_keys);
    println!("  Printed keys: {}", printed_keys);
    println!("  Tokenizer keys: {}", tokenizer_keys);
    println!("  Filtered keys: {}", total_keys - printed_keys);
}

pub fn render_tensor_summary(model: &Model, name: &str) -> Option<String> {
    model
        .tensor(name)
        .map(|t| format!("{}: shape={:?}, dtype={:?}", t.name, t.shape, t.kind))
}

pub fn render_all_tensor_summaries(model: &Model) -> impl Iterator<Item = String> + '_ {
    model
        .tensors()
        .iter()
        .map(|t| format!("{}: shape={:?}, dtype={:?}", t.name, t.shape, t.kind))

    //}

    //    println!("[render]");
    //  let mut entries: Vec<_> = model.iter_named().collect();
    //entries.sort_by_key(|(_, t)| t.offset); // optional sort

    //  let mut writer: Box<dyn Write + Send> = match output {
    //   Some(path) => Box::new(BufWriter::new(File::create(path)?)),
    // None => Box::new(std::io::stdout()),
    //};
    //
    //    rayon::ThreadPoolBuilder::new()
    //      .num_threads(8)
    //    .build_global()
    //  .expect("Rayon thread pool init failed");

    //  println!("Rayon threads: {}", rayon::current_num_threads());

    //  let results: Vec<String> = entries
    //    .par_iter()
    //  .map(|(name, tensor)| {
    //            println!("Running on thread: {:?}", std::thread::current().id());
    // let mut buf = String::new();
    // use std::fmt::Write;

    //      writeln!(buf, "  [{}]:", name).ok();
    //writeln!(buf, "    kind: {}", tensor.kind.into()).ok();
    //    writeln!(buf, "    offset: {}", tensor.offset).ok();
    //  writeln!(buf, "    size: {}", tensor.size).ok();
    //writeln!(buf, "    shape: {:?}", tensor.shape).ok();

    //           if tensor.size < 1024 * 4 {
    //             if let Ok(view) = tensor.view(model.raw()) {
    //               if view.dtype == DType::F32 {
    //                 if let Ok(slice) = view.as_f32_slice() {
    //                   let preview: Vec<_> = slice.iter().take(10).cloned().collect();
    //                 writeln!(buf, "    preview: {:?}", preview).ok();
    //           }
    //     }
    //  }
    //}

    //    buf
    //  })
    //.collect();

    //   for line in results {
    //        writer.write_all(line.as_bytes())?;
    //     println!("{line}");
    // }

    //    Ok(())
    //    model
    //      .tensors()
    //    .iter()
    //  .map(|t| format!("{}: shape={:?}, dtype={:?}", t.name, t.shape, t.kind))

    //    let results: Vec<String> = entries.par_iter().map(|(name, tensor)| {
    //      let mut buf = String::new();
    //      use std::fmt::Write as _;
    //           writeln!(buf, "  [{}]:", name).ok();
    //           writeln!(buf, "    kind: {}", tensor.kind).ok();
    //           writeln!(buf, "    offset: {}", tensor.offset).ok();
    //           writeln!(buf, "    size: {}", tensor.size).ok();
    //           writeln!(buf, "    shape: {:?}", tensor.shape).ok();

    //           if tensor.size < 1024 * 4 {
    //             if let Ok(view) = tensor.view(gguf.raw_bytes()) {
    //               view.dtype == DType::F32 {
    //                 if let Ok(slice) = view.as_f32_slice() {
    //                   let preview: Vec<_> = slice.iter().take(10).cloned().collect();
    //                writeln!(buf, "    preview: {:?}", preview).ok();
    //                       }
    //                 }
    //              }
    //        }
    //      buf
    //      }).collect();

    //    for line in results {
    //      writer.write_all(line.as_bytes())?;
    //    }
    //  writer.flush()?;
    // }
}
