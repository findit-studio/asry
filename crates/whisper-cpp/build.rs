//! Build script for the in-house whisper.cpp bindings.
//!
//! Two stages:
//!
//! 1. Compile the vendored `whisper.cpp/` (git submodule) via
//!    cmake-rs. Feature flags translate to `-DGGML_METAL=ON` etc.
//!    Output is a static `libwhisper.a` plus the ggml satellite
//!    libraries that whisper.cpp's CMakeLists produces.
//! 2. Run bindgen against `whisper.cpp/include/whisper.h`,
//!    allowlist-filtered to the `whisper_*` and `ggml_*` symbols
//!    we actually consume. Output goes to `OUT_DIR/bindings.rs`.
//!
//! Bootstrap behaviour: when `whisper.cpp/` is missing (e.g. the
//! consumer cloned without `--recurse-submodules`), this script
//! emits a clear instruction string and `cargo:warning=`s — it does
//! not panic — so `cargo check` still resolves the crate's API.
//! The actual link step will fail downstream, by design.

use std::{env, path::PathBuf};

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=wrapper.h");
  println!("cargo:rerun-if-env-changed=WHISPER_CPP_DIR");

  let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  // `WHISPER_CPP_DIR` lets a CI runner point at a pre-built tree
  // (skip cmake on every cargo run). When unset, default to the
  // submodule path under the crate.
  let whisper_src = env::var("WHISPER_CPP_DIR")
    .map(PathBuf::from)
    .unwrap_or_else(|_| crate_dir.join("whisper.cpp"));

  if !whisper_src.join("CMakeLists.txt").is_file() {
    println!(
      "cargo:warning=whisper.cpp source not found at {:?}.",
      whisper_src
    );
    println!("cargo:warning=Run `git submodule update --init --recursive` from the repo root,");
    println!(
      "cargo:warning=or set WHISPER_CPP_DIR to a checkout. Skipping cmake + bindgen for now;"
    );
    println!("cargo:warning=link step will fail until the source is available.");
    return;
  }

  let dst = build_whisper_cpp(&whisper_src);
  emit_link_directives(&dst);
  generate_bindings(&whisper_src);
}

/// Drive the cmake build. Returns the install root cmake-rs
/// produced (typically `OUT_DIR/`).
fn build_whisper_cpp(whisper_src: &PathBuf) -> PathBuf {
  let mut cfg = cmake::Config::new(whisper_src);
  cfg
    .define("BUILD_SHARED_LIBS", "OFF")
    .define("WHISPER_BUILD_EXAMPLES", "OFF")
    .define("WHISPER_BUILD_TESTS", "OFF")
    .define("WHISPER_BUILD_SERVER", "OFF")
    // ggml fast-math + Apple Accelerate / OpenBLAS are decided
    // per-feature below.
    .profile("Release");

  if cfg!(feature = "metal") {
    cfg.define("GGML_METAL", "ON");
    cfg.define("GGML_METAL_NDEBUG", "ON");
    // Embed the metal shader library bytes into libggml-metal.a
    // so the runtime doesn't need a sibling `default.metallib`.
    cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
  } else {
    cfg.define("GGML_METAL", "OFF");
  }

  if cfg!(feature = "coreml") {
    cfg.define("WHISPER_COREML", "ON");
    // Enable the post-init fallback: if the `.mlmodelc` companion
    // is missing at runtime, fall back to the GGML encoder rather
    // than aborting. This is what whisper-cli does by default.
    cfg.define("WHISPER_COREML_ALLOW_FALLBACK", "ON");
  }

  if cfg!(feature = "openblas") {
    cfg.define("GGML_BLAS", "ON");
    cfg.define("GGML_BLAS_VENDOR", "OpenBLAS");
  } else if cfg!(target_vendor = "apple") && !cfg!(feature = "metal") {
    // Apple CPU build: prefer the system Accelerate framework.
    cfg.define("GGML_BLAS", "ON");
    cfg.define("GGML_BLAS_VENDOR", "Apple");
  }

  if cfg!(feature = "cuda") {
    cfg.define("GGML_CUDA", "ON");
  }

  cfg.build()
}

/// Tell cargo which static libraries to link, in the right order
/// for the GNU/macos/MSVC linkers. cmake-rs's `build()` returns
/// `<OUT_DIR>/`, with libs under `lib/`.
fn emit_link_directives(install_root: &PathBuf) {
  let lib_dir = install_root.join("lib");
  println!("cargo:rustc-link-search=native={}", lib_dir.display());

  // Order matters for GNU ld: depending libs first, low-level
  // last. whisper depends on ggml; ggml's metal/blas/coreml
  // sub-libs are leaves.
  println!("cargo:rustc-link-lib=static=whisper");
  println!("cargo:rustc-link-lib=static=ggml");
  println!("cargo:rustc-link-lib=static=ggml-base");
  println!("cargo:rustc-link-lib=static=ggml-cpu");

  // On Apple Silicon, whisper.cpp's CMake also builds the
  // ggml-blas backend automatically (the BLAS-via-Accelerate
  // path), even when Metal is the primary backend. We link it
  // unconditionally on Apple targets so the resulting binary
  // resolves `ggml_backend_blas_reg`.
  if cfg!(target_vendor = "apple") {
    println!("cargo:rustc-link-lib=static=ggml-blas");
    println!("cargo:rustc-link-lib=framework=Accelerate");
  }
  if cfg!(feature = "metal") {
    println!("cargo:rustc-link-lib=static=ggml-metal");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
  }
  if cfg!(feature = "coreml") {
    println!("cargo:rustc-link-lib=static=whisper.coreml");
    println!("cargo:rustc-link-lib=framework=CoreML");
  }
  if cfg!(feature = "openblas") {
    println!("cargo:rustc-link-lib=dylib=openblas");
  }

  // C++ stdlib — whisper.cpp / ggml are C++.
  if cfg!(target_os = "macos") {
    println!("cargo:rustc-link-lib=dylib=c++");
  } else if cfg!(target_os = "linux") {
    println!("cargo:rustc-link-lib=dylib=stdc++");
  }
}

/// Run bindgen against a curated `wrapper.h` and write the result
/// to `src/generated.rs`.
///
/// We deliberately put the file IN-TREE rather than under
/// `OUT_DIR`:
///
/// - the generated FFI is visible in `git grep` / IDE jumps,
///   which matters for an FFI surface we own end-to-end (every
///   `unsafe` block downstream lands on a definition you can
///   read in this repo).
/// - the diff of a whisper.cpp upgrade is reviewable in normal
///   PR tooling — no "look in the build cache" indirection.
///
/// The trade-off: `src/generated.rs` is a build artefact, so
/// build.rs must NOT add `cargo:rerun-if-changed=src/generated.rs`
/// (would rebuild forever) and the file must be in `.gitignore`.
/// We add a header comment marking the file machine-generated so
/// nobody hand-edits it.
fn generate_bindings(whisper_src: &PathBuf) {
  let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  let header = crate_dir.join("wrapper.h");
  let include = whisper_src.join("include");
  let ggml_include = whisper_src.join("ggml").join("include");

  let bindings = bindgen::Builder::default()
    .header(header.to_string_lossy().to_string())
    .clang_arg(format!("-I{}", include.display()))
    .clang_arg(format!("-I{}", ggml_include.display()))
    // Only the symbols we consume. New surface needs an explicit
    // allowlist add — keeps the generated file small AND keeps
    // unintended dependencies out of the public API.
    .allowlist_function("whisper_.*")
    .allowlist_function("ggml_log_.*")
    .allowlist_type("whisper_.*")
    .allowlist_type("ggml_log_.*")
    .allowlist_var("WHISPER_.*")
    // CargoCallbacks calls println!("cargo:rerun-if-changed=...")
    // for every header bindgen pulled. Those land under
    // `whisper.cpp/...` so we DO want them — a header change in
    // the submodule should re-bindgen.
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .layout_tests(false)
    .derive_default(true)
    .derive_debug(true)
    .generate()
    .expect("bindgen failed");

  let dest = crate_dir.join("src").join("generated.rs");
  let body = bindings.to_string();
  let header_comment = format!(
    "// @generated\n\
     //\n\
     // whisper.cpp FFI surface — produced by bindgen against the\n\
     // submodule pinned at `crates/whisper-cpp/whisper.cpp/`. Do\n\
     // not edit by hand: every `cargo build` regenerates this\n\
     // file from `wrapper.h` and the allowlist in `build.rs`.\n\
     //\n\
     // Adding a new symbol means extending BOTH `wrapper.h`'s\n\
     // `#include` set AND the `allowlist_*` directives in\n\
     // `build.rs::generate_bindings()`. Do not relax the\n\
     // allowlists casually — they are the boundary between\n\
     // \"FFI surface we own\" and \"every header bindgen could\n\
     // possibly pull in transitively.\"\n\
     //\n\
     // Source crate: {pkg} {ver}\n\
     // Source header: wrapper.h -> whisper.cpp/include/whisper.h\n\
     //\n\n",
    pkg = env!("CARGO_PKG_NAME"),
    ver = env!("CARGO_PKG_VERSION"),
  );

  // Skip the write if the content is byte-identical to what's on
  // disk — keeps `cargo build` from touching mtimes and triggering
  // downstream rebuilds when nothing actually changed.
  let new_contents = format!("{header_comment}{body}");
  if let Ok(existing) = std::fs::read_to_string(&dest) {
    if existing == new_contents {
      return;
    }
  }
  std::fs::write(&dest, new_contents).expect("failed to write src/generated.rs");
}
