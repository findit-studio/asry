//! Punctuation marks nobody reads aloud.
//!
//! A punctuation mark has no acoustic realization a CTC lattice could place, so it is never an
//! alignment target. Tokenization drops every mark [`is_silent_mark`] names that the vocabulary
//! does not spell, wherever it stands and under every OOV policy: no token, no wildcard, and no
//! OOV event. The policies decide the spoken characters: letters, digits, symbols (`$`, `<`, `©`)
//! and the [marks read aloud](READ_ALOUD).

use core::cmp::Ordering;

/// Whether `c` is a punctuation mark nobody reads aloud: of Unicode general category P*
/// (connector, dash, open, close, initial quote, final quote and other punctuation) and not one of
/// the [marks read aloud](READ_ALOUD).
///
/// Says nothing about the vocabulary. A mark the vocabulary spells, such as the apostrophe of
/// wav2vec2-base-960h, is a token wherever normalization leaves it.
pub(crate) fn is_silent_mark(c: char) -> bool {
  is_punctuation(c) && !READ_ALOUD.contains(&c)
}

/// The punctuation marks a reader says as a word: the number sign, the percent sign and its
/// per-mille kin, the ampersand, the commercial at, the section sign and the pilcrow, with the
/// Arabic percent sign and the fullwidth forms of the first four.
///
/// A mark read aloud only in context stays silent: the `.` of `3.5`, the `,` of the German `4,9`,
/// the `/` of `km/h`. That is sound at asry's output granularity, the word, because a normalizer
/// never splits a word at a mark inside it: the Latin normalizers keep `km/h` and `well-known` one
/// word each, their surfaces as written, and their one segmentation rule splits a recognised
/// French or Italian clitic off the word it begins (`l'eau`), which splits no spoken mark. So a mark inside a word is
/// spoken, when it is, inside that word's span: the "per" of `km/h` lies between the `m` and the
/// `h`, whose tokens bound the word, and dropping the mark cannot move a word boundary. Surfacing
/// it as an OOV event instead would make a fail-closed policy refuse words it can align, such as
/// `3.5` on a vocabulary that spells digits. A word made only of such marks (a standalone `/`) has
/// no token to bound it: it is accounted like any word that aligns nothing, and a unit holding
/// nothing else is `UnalignedCause::NoAlignableText`.
pub(crate) const READ_ALOUD: [char; 13] = [
  '#', '%', '&', '@', '\u{A7}', '\u{B6}', '\u{66A}', '\u{2030}', '\u{2031}', '\u{FF03}',
  '\u{FF05}', '\u{FF06}', '\u{FF20}',
];

/// Whether `c` is of Unicode general category P*.
fn is_punctuation(c: char) -> bool {
  PUNCTUATION
    .binary_search_by(|&(first, last)| {
      if last < c {
        Ordering::Less
      } else if c < first {
        Ordering::Greater
      } else {
        Ordering::Equal
      }
    })
    .is_ok()
}

/// Unicode 16.0.0's general category P* as inclusive ranges, ascending and apart from one
/// another: 198 ranges, 855 code points.
///
/// Generated from the Unicode Character Database: every code point whose `General_Category` is
/// `Pc`, `Pd`, `Ps`, `Pe`, `Pi`, `Pf` or `Po`, merged into ranges. Python's `unicodedata` gives
/// the same set when its `unidata_version` is `16.0.0`.
const PUNCTUATION: [(char, char); 198] = [
  ('\u{21}', '\u{23}'),
  ('\u{25}', '\u{2A}'),
  ('\u{2C}', '\u{2F}'),
  ('\u{3A}', '\u{3B}'),
  ('\u{3F}', '\u{40}'),
  ('\u{5B}', '\u{5D}'),
  ('\u{5F}', '\u{5F}'),
  ('\u{7B}', '\u{7B}'),
  ('\u{7D}', '\u{7D}'),
  ('\u{A1}', '\u{A1}'),
  ('\u{A7}', '\u{A7}'),
  ('\u{AB}', '\u{AB}'),
  ('\u{B6}', '\u{B7}'),
  ('\u{BB}', '\u{BB}'),
  ('\u{BF}', '\u{BF}'),
  ('\u{37E}', '\u{37E}'),
  ('\u{387}', '\u{387}'),
  ('\u{55A}', '\u{55F}'),
  ('\u{589}', '\u{58A}'),
  ('\u{5BE}', '\u{5BE}'),
  ('\u{5C0}', '\u{5C0}'),
  ('\u{5C3}', '\u{5C3}'),
  ('\u{5C6}', '\u{5C6}'),
  ('\u{5F3}', '\u{5F4}'),
  ('\u{609}', '\u{60A}'),
  ('\u{60C}', '\u{60D}'),
  ('\u{61B}', '\u{61B}'),
  ('\u{61D}', '\u{61F}'),
  ('\u{66A}', '\u{66D}'),
  ('\u{6D4}', '\u{6D4}'),
  ('\u{700}', '\u{70D}'),
  ('\u{7F7}', '\u{7F9}'),
  ('\u{830}', '\u{83E}'),
  ('\u{85E}', '\u{85E}'),
  ('\u{964}', '\u{965}'),
  ('\u{970}', '\u{970}'),
  ('\u{9FD}', '\u{9FD}'),
  ('\u{A76}', '\u{A76}'),
  ('\u{AF0}', '\u{AF0}'),
  ('\u{C77}', '\u{C77}'),
  ('\u{C84}', '\u{C84}'),
  ('\u{DF4}', '\u{DF4}'),
  ('\u{E4F}', '\u{E4F}'),
  ('\u{E5A}', '\u{E5B}'),
  ('\u{F04}', '\u{F12}'),
  ('\u{F14}', '\u{F14}'),
  ('\u{F3A}', '\u{F3D}'),
  ('\u{F85}', '\u{F85}'),
  ('\u{FD0}', '\u{FD4}'),
  ('\u{FD9}', '\u{FDA}'),
  ('\u{104A}', '\u{104F}'),
  ('\u{10FB}', '\u{10FB}'),
  ('\u{1360}', '\u{1368}'),
  ('\u{1400}', '\u{1400}'),
  ('\u{166E}', '\u{166E}'),
  ('\u{169B}', '\u{169C}'),
  ('\u{16EB}', '\u{16ED}'),
  ('\u{1735}', '\u{1736}'),
  ('\u{17D4}', '\u{17D6}'),
  ('\u{17D8}', '\u{17DA}'),
  ('\u{1800}', '\u{180A}'),
  ('\u{1944}', '\u{1945}'),
  ('\u{1A1E}', '\u{1A1F}'),
  ('\u{1AA0}', '\u{1AA6}'),
  ('\u{1AA8}', '\u{1AAD}'),
  ('\u{1B4E}', '\u{1B4F}'),
  ('\u{1B5A}', '\u{1B60}'),
  ('\u{1B7D}', '\u{1B7F}'),
  ('\u{1BFC}', '\u{1BFF}'),
  ('\u{1C3B}', '\u{1C3F}'),
  ('\u{1C7E}', '\u{1C7F}'),
  ('\u{1CC0}', '\u{1CC7}'),
  ('\u{1CD3}', '\u{1CD3}'),
  ('\u{2010}', '\u{2027}'),
  ('\u{2030}', '\u{2043}'),
  ('\u{2045}', '\u{2051}'),
  ('\u{2053}', '\u{205E}'),
  ('\u{207D}', '\u{207E}'),
  ('\u{208D}', '\u{208E}'),
  ('\u{2308}', '\u{230B}'),
  ('\u{2329}', '\u{232A}'),
  ('\u{2768}', '\u{2775}'),
  ('\u{27C5}', '\u{27C6}'),
  ('\u{27E6}', '\u{27EF}'),
  ('\u{2983}', '\u{2998}'),
  ('\u{29D8}', '\u{29DB}'),
  ('\u{29FC}', '\u{29FD}'),
  ('\u{2CF9}', '\u{2CFC}'),
  ('\u{2CFE}', '\u{2CFF}'),
  ('\u{2D70}', '\u{2D70}'),
  ('\u{2E00}', '\u{2E2E}'),
  ('\u{2E30}', '\u{2E4F}'),
  ('\u{2E52}', '\u{2E5D}'),
  ('\u{3001}', '\u{3003}'),
  ('\u{3008}', '\u{3011}'),
  ('\u{3014}', '\u{301F}'),
  ('\u{3030}', '\u{3030}'),
  ('\u{303D}', '\u{303D}'),
  ('\u{30A0}', '\u{30A0}'),
  ('\u{30FB}', '\u{30FB}'),
  ('\u{A4FE}', '\u{A4FF}'),
  ('\u{A60D}', '\u{A60F}'),
  ('\u{A673}', '\u{A673}'),
  ('\u{A67E}', '\u{A67E}'),
  ('\u{A6F2}', '\u{A6F7}'),
  ('\u{A874}', '\u{A877}'),
  ('\u{A8CE}', '\u{A8CF}'),
  ('\u{A8F8}', '\u{A8FA}'),
  ('\u{A8FC}', '\u{A8FC}'),
  ('\u{A92E}', '\u{A92F}'),
  ('\u{A95F}', '\u{A95F}'),
  ('\u{A9C1}', '\u{A9CD}'),
  ('\u{A9DE}', '\u{A9DF}'),
  ('\u{AA5C}', '\u{AA5F}'),
  ('\u{AADE}', '\u{AADF}'),
  ('\u{AAF0}', '\u{AAF1}'),
  ('\u{ABEB}', '\u{ABEB}'),
  ('\u{FD3E}', '\u{FD3F}'),
  ('\u{FE10}', '\u{FE19}'),
  ('\u{FE30}', '\u{FE52}'),
  ('\u{FE54}', '\u{FE61}'),
  ('\u{FE63}', '\u{FE63}'),
  ('\u{FE68}', '\u{FE68}'),
  ('\u{FE6A}', '\u{FE6B}'),
  ('\u{FF01}', '\u{FF03}'),
  ('\u{FF05}', '\u{FF0A}'),
  ('\u{FF0C}', '\u{FF0F}'),
  ('\u{FF1A}', '\u{FF1B}'),
  ('\u{FF1F}', '\u{FF20}'),
  ('\u{FF3B}', '\u{FF3D}'),
  ('\u{FF3F}', '\u{FF3F}'),
  ('\u{FF5B}', '\u{FF5B}'),
  ('\u{FF5D}', '\u{FF5D}'),
  ('\u{FF5F}', '\u{FF65}'),
  ('\u{10100}', '\u{10102}'),
  ('\u{1039F}', '\u{1039F}'),
  ('\u{103D0}', '\u{103D0}'),
  ('\u{1056F}', '\u{1056F}'),
  ('\u{10857}', '\u{10857}'),
  ('\u{1091F}', '\u{1091F}'),
  ('\u{1093F}', '\u{1093F}'),
  ('\u{10A50}', '\u{10A58}'),
  ('\u{10A7F}', '\u{10A7F}'),
  ('\u{10AF0}', '\u{10AF6}'),
  ('\u{10B39}', '\u{10B3F}'),
  ('\u{10B99}', '\u{10B9C}'),
  ('\u{10D6E}', '\u{10D6E}'),
  ('\u{10EAD}', '\u{10EAD}'),
  ('\u{10F55}', '\u{10F59}'),
  ('\u{10F86}', '\u{10F89}'),
  ('\u{11047}', '\u{1104D}'),
  ('\u{110BB}', '\u{110BC}'),
  ('\u{110BE}', '\u{110C1}'),
  ('\u{11140}', '\u{11143}'),
  ('\u{11174}', '\u{11175}'),
  ('\u{111C5}', '\u{111C8}'),
  ('\u{111CD}', '\u{111CD}'),
  ('\u{111DB}', '\u{111DB}'),
  ('\u{111DD}', '\u{111DF}'),
  ('\u{11238}', '\u{1123D}'),
  ('\u{112A9}', '\u{112A9}'),
  ('\u{113D4}', '\u{113D5}'),
  ('\u{113D7}', '\u{113D8}'),
  ('\u{1144B}', '\u{1144F}'),
  ('\u{1145A}', '\u{1145B}'),
  ('\u{1145D}', '\u{1145D}'),
  ('\u{114C6}', '\u{114C6}'),
  ('\u{115C1}', '\u{115D7}'),
  ('\u{11641}', '\u{11643}'),
  ('\u{11660}', '\u{1166C}'),
  ('\u{116B9}', '\u{116B9}'),
  ('\u{1173C}', '\u{1173E}'),
  ('\u{1183B}', '\u{1183B}'),
  ('\u{11944}', '\u{11946}'),
  ('\u{119E2}', '\u{119E2}'),
  ('\u{11A3F}', '\u{11A46}'),
  ('\u{11A9A}', '\u{11A9C}'),
  ('\u{11A9E}', '\u{11AA2}'),
  ('\u{11B00}', '\u{11B09}'),
  ('\u{11BE1}', '\u{11BE1}'),
  ('\u{11C41}', '\u{11C45}'),
  ('\u{11C70}', '\u{11C71}'),
  ('\u{11EF7}', '\u{11EF8}'),
  ('\u{11F43}', '\u{11F4F}'),
  ('\u{11FFF}', '\u{11FFF}'),
  ('\u{12470}', '\u{12474}'),
  ('\u{12FF1}', '\u{12FF2}'),
  ('\u{16A6E}', '\u{16A6F}'),
  ('\u{16AF5}', '\u{16AF5}'),
  ('\u{16B37}', '\u{16B3B}'),
  ('\u{16B44}', '\u{16B44}'),
  ('\u{16D6D}', '\u{16D6F}'),
  ('\u{16E97}', '\u{16E9A}'),
  ('\u{16FE2}', '\u{16FE2}'),
  ('\u{1BC9F}', '\u{1BC9F}'),
  ('\u{1DA87}', '\u{1DA8B}'),
  ('\u{1E5FF}', '\u{1E5FF}'),
  ('\u{1E95E}', '\u{1E95F}'),
];

#[cfg(test)]
mod tests {
  use super::*;

  /// Every ASCII character is classified as Unicode files it: the ASCII punctuation, less the
  /// nine characters Unicode files as symbols.
  #[test]
  fn ascii_punctuation_is_unicodes_p_categories() {
    for code in 0..=0x7F_u8 {
      let c = char::from(code);
      assert_eq!(
        is_punctuation(c),
        c.is_ascii_punctuation() && !"$+<=>^`|~".contains(c),
        "{c:?}"
      );
    }
  }

  /// The table is well formed: every range runs forward and stands apart from the next, so the
  /// binary search sees one answer per code point. It holds 855 code points, Unicode 16.0.0's
  /// count, and the lookup answers at both edges of every range and just outside them.
  #[test]
  fn the_table_is_unicode_16s_p_categories() {
    for pair in PUNCTUATION.windows(2) {
      let ((first, last), (next, _)) = (pair[0], pair[1]);
      assert!(first <= last, "{pair:?}");
      assert!(u32::from(last) + 1 < u32::from(next), "{pair:?}");
    }
    let code_points: u32 = PUNCTUATION
      .iter()
      .map(|&(first, last)| u32::from(last) - u32::from(first) + 1)
      .sum();
    assert_eq!(code_points, 855);

    for &(first, last) in &PUNCTUATION {
      assert!(
        is_punctuation(first) && is_punctuation(last),
        "{first:?}..={last:?}"
      );
      let before = u32::from(first).checked_sub(1).and_then(char::from_u32);
      let after = char::from_u32(u32::from(last) + 1);
      for outside in before.into_iter().chain(after) {
        assert!(!is_punctuation(outside), "{outside:?}");
      }
    }

    for mark in [
      '\u{A1}',
      '\u{B7}',
      '\u{2019}',
      '\u{2026}',
      '\u{3001}',
      '\u{300A}',
      '\u{FF08}',
      '\u{1E95F}',
    ] {
      assert!(is_punctuation(mark), "{mark:?}");
    }
    for other in [
      '\u{A0}',
      '\u{A9}',
      '\u{B0}',
      '\u{E9}',
      '\u{2BC}',
      '\u{20AC}',
      '\u{65E5}',
      '\u{1E960}',
    ] {
      assert!(!is_punctuation(other), "{other:?}");
    }
  }

  /// A punctuation mark nobody reads aloud is silent; every other character is spoken: a letter,
  /// a digit, a symbol, a mark read aloud.
  #[test]
  fn a_mark_nobody_reads_aloud_is_silent_and_every_other_character_is_spoken() {
    for mark in [
      '.', ',', '?', '!', ';', ':', '"', '\'', '(', ')', '-', '/', '*', '_', '\\', '\u{2014}',
      '\u{201C}', '\u{2019}', '\u{2026}', '\u{AB}', '\u{BF}', '\u{3001}', '\u{3002}', '\u{300A}',
      '\u{301C}', '\u{FF0C}',
    ] {
      assert!(is_silent_mark(mark), "{mark:?} is silent");
    }
    for spoken in [
      'a', 'Z', '\u{E9}', '5', '\u{65E5}', '$', '<', '+', '|', '\u{A9}', '\u{20AC}', '\u{2BC}',
    ] {
      assert!(!is_silent_mark(spoken), "{spoken:?} is spoken");
    }
    for aloud in READ_ALOUD {
      assert!(
        is_punctuation(aloud),
        "{aloud:?} is punctuation, or listing it changes nothing"
      );
      assert!(!is_silent_mark(aloud), "{aloud:?} is read aloud");
    }
  }
}
