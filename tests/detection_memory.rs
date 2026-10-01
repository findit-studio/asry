//! **A detection holds its text once, however many events it finds.**
//!
//! A detection is bound to the text it read, and that binding must not
//! cost a copy of the text per event: a transcript whose characters are
//! all out of vocabulary would then allocate its length squared before
//! anything was aligned. This test counts the bytes `detect_oov`
//! allocates, on this thread, for a text of n unspellable characters (one
//! event each), and requires the count to grow linearly in n.
#![cfg(feature = "emissions")]

use std::{
  alloc::{GlobalAlloc, Layout, System},
  cell::Cell,
};

use asry::{Lang, emissions::EmissionsAligner};

/// Counts the bytes allocated on the current thread while `COUNTING` is
/// set. Thread-local, const-initialized and drop-free, so reading it never
/// allocates, and allocations by other test threads never count.
struct CountingAllocator;

thread_local! {
  static COUNTING: Cell<bool> = const { Cell::new(false) };
  static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn count(bytes: usize) {
  let _ = COUNTING.try_with(|counting| {
    if counting.get() {
      let _ = BYTES.try_with(|total| total.set(total.get() + bytes));
    }
  });
}

// SAFETY: every method forwards to `System` with the caller's layout and
// pointer unchanged; counting touches only drop-free thread-locals.
unsafe impl GlobalAlloc for CountingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    count(layout.size());
    // SAFETY: forwarded unchanged.
    unsafe { System.alloc(layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    count(layout.size());
    // SAFETY: forwarded unchanged.
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    count(new_size);
    // SAFETY: forwarded unchanged.
    unsafe { System.realloc(ptr, layout, new_size) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: forwarded unchanged.
    unsafe { System.dealloc(ptr, layout) }
  }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// A wav2vec2-base-960h-shape vocabulary: uppercase letters, the
/// apostrophe, a `|` delimiter, and no digit.
const TOKENIZER_JSON: &str = r#"{
 "version": "1.0",
 "truncation": null,
 "padding": null,
 "added_tokens": [],
 "normalizer": null,
 "pre_tokenizer": {"type": "Split", "pattern": {"Regex": ""}, "behavior": "Isolated", "invert": false},
 "post_processor": null,
 "decoder": null,
 "model": {
 "type": "WordLevel",
 "vocab": {
 "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
 "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12,
 "R": 13, "D": 14, "L": 15, "U": 16, "M": 17, "W": 18, "C": 19, "F": 20,
 "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26, "'": 27, "X": 28,
 "J": 29, "Q": 30, "Z": 31
 },
 "unk_token": "<unk>"
 }
 }"#;

/// Bytes `detect_oov` allocates for `n` digits, none of which the
/// vocabulary spells: `n` events.
fn bytes_to_detect(aligner: &EmissionsAligner, n: usize) -> usize {
  let text = "4".repeat(n);
  BYTES.with(|total| total.set(0));
  COUNTING.with(|counting| counting.set(true));
  let detection = aligner.detect_oov(&text);
  COUNTING.with(|counting| counting.set(false));
  let detection = detection.expect("detect_oov");
  assert_eq!(detection.events().len(), n, "one event per digit");
  drop(detection);
  BYTES.with(Cell::get)
}

#[test]
fn detection_memory_grows_linearly_with_the_text() {
  let aligner = EmissionsAligner::builder(Lang::En, TOKENIZER_JSON.as_bytes())
    .build()
    .expect("build");
  // Warm any lazily initialized state before measuring.
  bytes_to_detect(&aligner, 64);

  let sizes = [2_000_usize, 4_000, 8_000, 16_000];
  let bytes: Vec<usize> = sizes
    .iter()
    .map(|&n| bytes_to_detect(&aligner, n))
    .collect();
  for (pair, sizes) in bytes.windows(2).zip(sizes.windows(2)) {
    // Twice the text allocates about twice the bytes. A copy of the
    // text per event would allocate four times as much.
    assert!(
      pair[1] * 10 <= pair[0] * 25,
      "doubling the text from {} to {} characters took {} to {} bytes: more than \
       linear",
      sizes[0],
      sizes[1],
      pair[0],
      pair[1],
    );
  }
  let (&last_bytes, &last_n) = (bytes.last().expect("sizes"), sizes.last().expect("sizes"));
  assert!(
    last_bytes <= last_n * 512,
    "{last_n} characters took {last_bytes} bytes: more than 512 per character"
  );
}
