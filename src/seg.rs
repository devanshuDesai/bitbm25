//! One immutable segment: what it stores, how it is built, and how a term's postings are read.
//!
//! Rows are numbered by position and grouped into 64-row blocks. By default a segment orders its
//! rows by length (`Order::Length`), so the rows of a block are about equally long and a term's
//! best possible score in a block follows from the largest tf it has there and the block's shortest
//! row: nothing about bounds is stored per posting. `ids` maps a row back to the host's id.
//!
//! Terms present in at least a quarter of the blocks do keep a bound, one byte per block: the
//! largest length-dependent score part there, rounded up to a 255th of its ceiling (k1 + 1). Queries
//! sum those bytes sixteen blocks at a time, and they are tight in either row order. Below that, an
//! Elias-Fano term keeps one such byte per stretch of 64 of its buckets (one per Elias-Fano sample):
//! about a byte per 64 to 128 postings, which bounds it in the blocks it was not decoded for.
//!
//! A term's rows are stored in one of four forms, by how many it has:
//!   - one or two rows: inline in the lexicon;
//!   - a bitmap of all rows (one word per block), for terms in more than about a sixth of them;
//!   - two levels of bytes, for terms with two or more rows a block: per block, which of its eight
//!     8-row groups hold the term, then one byte of rows per such group. A block's mask is one table
//!     lookup away, which is what counts and ORs of such terms spend their time on;
//!   - Elias-Fano over the row numbers with at least six low bits, so a bucket is a block (or a
//!     power-of-two run of blocks for rare terms) and a block's rows are found in O(1).
//!
//! Term frequencies are nothing (all ones), two bits with a nibble for 4..18, or a byte; larger
//! values sit in a short sorted table.
//!
//! Positions (optional) are a stream per term, in posting order. A posting with one occurrence
//! stores its position in as many bits as the longest row of its 64-posting chunk needs; one with
//! more stores them as Elias-Fano over the same range. Both sizes follow from the tf and the chunk's
//! width, so there are no per-posting headers, only a bit offset and a width per chunk.
//!
//! The lexicon is sorted and front-coded in groups of 16 terms, each entry followed by where its
//! postings are.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;

use crate::bits::{self, Ef, Packer};
use crate::{as_bytes, bad_input, invalid, replace_file, Analyzer, Arr, Buf, Params, Stats};

pub(crate) const SHIFT: u32 = 6; // 64 rows per block
const LEX_GROUP: usize = 16;
const NONE: u32 = u32::MAX;

/// How a segment numbers its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Order {
    /// Shortest rows first. Bounds for ranked search are tight; a host maps rows to its ids with
    /// `Index::ids`.
    #[default]
    Length,
    /// As added (or, after a merge, segment by segment). Row masks then line up with the host's own
    /// row order, at the price of looser bounds for ranked search.
    Added,
}

/// Lower edges of the 256 length classes: every length up to 32, then steps of a sixteenth. A
/// block's class is that of its shortest row, so a bound computed from the edge is never too low.
pub(crate) const EDGES: [u32; 256] = {
    let mut e = [0u32; 256];
    let mut c = 1;
    while c < 256 {
        let step = e[c - 1] >> 4;
        e[c] = e[c - 1] + if step == 0 { 1 } else { step };
        c += 1;
    }
    e
};

/// The length-dependent part of BM25 for one term: tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)).
fn tf_part(tf: f64, dl: f64, avgdl: f64, p: Params) -> f64 {
    tf * (p.k1 + 1.0) / (tf + p.k1 * (1.0 - p.b + p.b * dl / avgdl))
}

/// tf_part of one row as a bound byte: units of (k1 + 1) / 255, rounded up.
fn bound_byte(tf: u32, dl: u32, avgdl: f64, p: Params) -> u8 {
    let units = tf_part(tf as f64, dl as f64, avgdl, p) / ((p.k1 + 1.0) / 255.0);
    (units.ceil() as u32).clamp(1, 255) as u8 // tf_part < k1 + 1, so at most 255
}

fn len_class(dl: u32) -> u8 {
    (EDGES.partition_point(|&e| e <= dl) - 1) as u8
}

const FORM_BITMAP: u8 = 1; // a word of rows per block
const FORM_BYTES: u8 = 2; // two levels of bytes
const FORM_MASK: u8 = 3; // neither: Elias-Fano
const TF_SMALL: u8 = 1 << 2; // two bits, a nibble for 4..18, a table beyond
const TF_BYTES: u8 = 2 << 2; // a byte, a table for 255 and more
const TF_MASK: u8 = 3 << 2; // neither: every tf is 1
const BOUNDS: u8 = 1 << 4; // a bound byte per block follows
const DENSE: u8 = 1 << 5; // in at least half the blocks: Elias-Fano samples every 8th bucket

/// Where a term's postings are, as the lexicon records it.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub(crate) struct Meta {
    pub(crate) df: u32,
    kind: u8,
    l: u8,                  // Elias-Fano low bits
    top: u8,                // the largest bound byte any of its rows has (see `Postings::top`)
    off: u64,               // first word in `post`
    n_esc: u32,             // TF_SMALL: postings with tf >= 4
    n_big: u32,             // postings whose tf is in the table
    inline: [(u32, u32); 2], // (row, tf) when df <= 2
    pos: u64,               // first word of its positions in `pos`; when df <= 2, where each row's are in the lexicon
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(bytes: &[u8], pos: &mut usize) -> u64 {
    let (mut v, mut shift) = (0u64, 0);
    loop {
        let b = bytes[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 || shift >= 56 {
            return v;
        }
        shift += 7;
    }
}

impl Meta {
    /// `positions`: with them stored, each posting's (in posting order, back to back) for an inline
    /// term; otherwise None.
    fn write(&self, out: &mut Vec<u8>, positions: Option<&[u32]>) {
        put_varint(out, self.df as u64);
        if self.df <= 2 {
            let mut at = 0;
            for &(row, tf) in &self.inline[..self.df as usize] {
                put_varint(out, row as u64);
                put_varint(out, tf as u64);
                if let Some(p) = positions {
                    let mut last = 0;
                    for &x in &p[at..at + tf as usize] {
                        put_varint(out, (x - last) as u64); // ascending: gaps
                        last = x;
                    }
                    at += tf as usize;
                }
            }
            return;
        }
        out.extend_from_slice(&[self.kind, self.l, self.top]);
        put_varint(out, self.off);
        if self.kind & TF_MASK != 0 {
            put_varint(out, self.n_esc as u64);
            put_varint(out, self.n_big as u64);
        }
        if positions.is_some() {
            put_varint(out, self.pos);
        }
    }

    fn read(bytes: &[u8], pos: &mut usize, positions: bool) -> Meta {
        let mut m = Meta { df: get_varint(bytes, pos) as u32, ..Default::default() };
        if m.df <= 2 {
            for i in 0..m.df as usize {
                m.inline[i] = (get_varint(bytes, pos) as u32, get_varint(bytes, pos) as u32);
                if positions {
                    m.pos |= (*pos as u64) << (32 * i); // where this row's positions start
                    for _ in 0..m.inline[i].1 {
                        get_varint(bytes, pos);
                    }
                }
            }
            return m;
        }
        (m.kind, m.l, m.top) = (bytes[*pos], bytes[*pos + 1], bytes[*pos + 2]);
        *pos += 3;
        m.off = get_varint(bytes, pos);
        if m.kind & TF_MASK != 0 {
            m.n_esc = get_varint(bytes, pos) as u32;
            m.n_big = get_varint(bytes, pos) as u32;
        }
        if positions {
            m.pos = get_varint(bytes, pos);
        }
        m
    }
}

/// Bits the positions of a posting take: `tf` of them, each below `2^width`.
fn pos_bits(tf: u32, width: u32) -> usize {
    match tf {
        1 => width as usize,
        // Elias-Fano with the top log2(tf) bits of each position in unary
        tf => (tf * (width - tf.ilog2() + 1)) as usize + (1 << tf.ilog2()),
    }
}

/// Appends one posting's positions (ascending, each below `2^width`).
fn write_pos(p: &mut Packer, positions: &[u32], width: u32) {
    let tf = positions.len() as u32;
    if tf == 1 {
        return p.push(positions[0], width);
    }
    let l = width - tf.ilog2();
    if l > 0 {
        positions.iter().for_each(|&x| p.push(x & ((1 << l) - 1), l));
    }
    let mut bucket = 0;
    for &x in positions {
        while bucket < x >> l {
            p.push(0, 1);
            bucket += 1;
        }
        p.push(1, 1);
    }
    (bucket..1 << (width - l)).for_each(|_| p.push(0, 1)); // every bucket ends with a zero
}

/// Reads a posting's `tf` positions, written by `write_pos` at bit `at` of `data`, onto `out`.
fn read_pos(data: &[u64], at: usize, tf: u32, width: u32, out: &mut Vec<u32>) {
    if tf == 1 {
        return out.push(bits::get(data, at, width));
    }
    let l = width - tf.ilog2();
    let high_at = at + (tf * l) as usize;
    let (mut k, mut from) = (0, high_at);
    while k < tf {
        let mut w = bits::window(data, from);
        while w != 0 && k < tf {
            // the k-th one sits at (its high part) + k
            let high = (from + w.trailing_zeros() as usize - high_at) as u32 - k;
            let low = if l > 0 { bits::get(data, at + (k * l) as usize, l) } else { 0 };
            out.push(high << l | low);
            k += 1;
            w &= w - 1;
        }
        from += 64;
    }
}

/// A posting of a term and the bit where its positions start: see `Postings::positions`.
#[derive(Clone, Copy)]
pub(crate) struct PosCursor {
    posting: usize,
    bit: usize,
}

impl Default for PosCursor {
    fn default() -> Self {
        PosCursor { posting: usize::MAX, bit: 0 } // in no chunk
    }
}

/// Where a term's positions are.
#[derive(Clone, Copy)]
enum Positions<'a> {
    None,
    /// Gap-coded varints in the lexicon; each row's start at `at`.
    Inline { lex: &'a [u8], at: [usize; 2] },
    /// `chunks`: for every 64 postings, the bit in `data` where their positions start (low 56 bits)
    /// and their width.
    Packed { chunks: &'a [u64], data: &'a [u64] },
}

/// A term's rows.
#[derive(Clone, Copy)]
enum Rows<'a> {
    Inline([u32; 2]),
    Ef(Ef<'a>),
    Bitmap { words: &'a [u64], rank: &'a [u32] }, // rank: rows before every 8th block
    /// `present`: a byte per block, a bit per 8-row group holding the term. `groups`: a byte of rows
    /// per such group (and eight spare bytes, so eight can be read from any of them). `at`: for
    /// every 8th block, the group bytes and the rows before it (and, last, in all).
    Bytes { present: &'a [u8], at: &'a [u32], groups: &'a [u8] },
}

/// For a block's presence byte, where each of its eight row bytes comes from among the stored ones
/// (255: nowhere, the group is empty).
const SPREAD: [[u8; 8]; 256] = {
    let mut t = [[255u8; 8]; 256];
    let mut p = 0;
    while p < 256 {
        let (mut k, mut n) = (0, 0);
        while k < 8 {
            if p >> k & 1 == 1 {
                t[p][k] = n;
                n += 1;
            }
            k += 1;
        }
        p += 1;
    }
    t
};

/// Eight bytes of `bytes` from `at`, as a word.
#[inline]
fn word_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

/// A block's row mask from its presence byte and its stored row bytes (the low bytes of `stored`).
#[inline]
fn spread(stored: u64, present: u8) -> u64 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of aarch64's baseline; the load reads one 8-byte row of the table.
    unsafe {
        use std::arch::aarch64::*;
        vget_lane_u64(vreinterpret_u64_u8(vtbl1_u8(vcreate_u8(stored), vld1_u8(SPREAD[present as usize].as_ptr()))), 0)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (mut m, mut stored, mut p) = (0u64, stored, present);
        while p != 0 {
            m |= (stored & 255) << (8 * p.trailing_zeros());
            stored >>= 8;
            p &= p - 1;
        }
        m
    }
}

/// Row bytes of the blocks before `b` in its group of eight: the bits set in their presence bytes.
#[inline]
fn groups_before(present: &[u8], b: usize) -> usize {
    (word_at(present, b & !7) & ((1u64 << (8 * (b & 7))) - 1)).count_ones() as usize
}

/// `first` as `Postings::block` returns it when finding it would cost extra: ask `Postings::first`.
pub(crate) const UNKNOWN: usize = usize::MAX;

/// A term's frequencies, by posting.
#[derive(Clone, Copy)]
enum Tfs<'a> {
    Ones,
    Inline([u32; 2]),
    Small { codes: &'a [u64], rank: &'a [u32], nibs: &'a [u64], big: &'a [u64] }, // rank: escapes before every 256th posting
    Bytes { bytes: &'a [u64], big: &'a [u64] },
}

const ESCAPES: u64 = 0x5555_5555_5555_5555; // low bit of each two-bit code

/// A term's postings in one segment: its rows in ascending order, a tf for each.
#[derive(Clone, Copy)]
pub(crate) struct Postings<'a> {
    pub(crate) df: usize,
    rows: Rows<'a>,
    tfs: Tfs<'a>,
    /// A byte per block, for terms in at least a quarter of them: the largest tf_part any of its rows
    /// there has (under the segment's own average length), in 255ths of k1 + 1, rounded up; 0 where
    /// the term is absent.
    pub(crate) bounds: Option<&'a [u64]>,
    /// The largest such byte over all the term's rows: the most it can score anywhere.
    pub(crate) top: u8,
    /// For an Elias-Fano term without bound bytes that spans several of its sampled stretches (2^s
    /// buckets each), the largest byte in each stretch, and log2 of the blocks a stretch covers.
    segs: Option<(&'a [u8], u32)>,
    positions: Positions<'a>,
}

impl<'a> Postings<'a> {
    /// How the rows and the frequencies are stored, for reports.
    pub(crate) fn forms(&self) -> (&'static str, &'static str) {
        let rows = match self.rows {
            Rows::Inline(_) => "inline",
            Rows::Ef(_) => "elias-fano",
            Rows::Bitmap { .. } => "bitmap",
            Rows::Bytes { .. } => "bytes",
        };
        let tfs = match self.tfs {
            Tfs::Ones | Tfs::Inline(_) => "ones",
            Tfs::Small { .. } => "2-bit",
            Tfs::Bytes { .. } => "bytes",
        };
        (rows, tfs)
    }

    /// The term's rows in block `b` as a mask, and the index of the first among its postings
    /// (`UNKNOWN` where that takes more work: see `first`).
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn block(&self, b: u32) -> Option<(u64, usize)> {
        match &self.rows {
            Rows::Inline(r) => {
                let (mut m, mut first) = (0u64, usize::MAX);
                for (i, &row) in r[..self.df].iter().enumerate() {
                    if row >> SHIFT == b {
                        m |= 1 << (row & 63);
                        first = first.min(i);
                    }
                }
                (m != 0).then_some((m, first))
            }
            Rows::Bitmap { words, .. } => match words[b as usize] {
                0 => None,
                w => Some((w, UNKNOWN)),
            },
            Rows::Bytes { present, at, groups } => {
                let p = present[b as usize];
                if p == 0 {
                    return None;
                }
                let o = at[2 * (b as usize >> 3)] as usize + groups_before(present, b as usize);
                Some((spread(word_at(groups, o), p), UNKNOWN))
            }
            Rows::Ef(ef) if ef.l == SHIFT => {
                let (first, n) = ef.bucket(b as usize);
                (n != 0).then(|| (low_mask(ef, first, n), first))
            }
            Rows::Ef(ef) => {
                let sub_bits = ef.l - SHIFT;
                let (first, n) = ef.bucket((b >> sub_bits) as usize);
                let sub = b & ((1 << sub_bits) - 1);
                let (mut m, mut at) = (0u64, usize::MAX);
                for i in first..first + n {
                    let low = ef.low(i);
                    if low >> SHIFT == sub {
                        m |= 1 << (low & 63);
                        at = at.min(i);
                    }
                }
                (m != 0).then_some((m, at))
            }
        }
    }

    /// ORs the term's rows in blocks `from..from + out.len()` into `out`, a word per block.
    #[cfg_attr(feature = "prof", inline(never))]
    pub(crate) fn or_into(&self, from: u32, out: &mut [u64]) {
        let to = from + out.len() as u32;
        match &self.rows {
            Rows::Bitmap { words, .. } => out.iter_mut().zip(&words[from as usize..]).for_each(|(o, w)| *o |= w),
            Rows::Bytes { present, at, groups } => {
                let start = from as usize;
                let mut o = at[2 * (start >> 3)] as usize + groups_before(present, start);
                for (out, &p) in out.iter_mut().zip(&present[start..]) {
                    if p != 0 {
                        *out |= spread(word_at(groups, o), p);
                        o += p.count_ones() as usize;
                    }
                }
            }
            Rows::Ef(ef) => ef.scan(from << SHIFT, |_, row| {
                let b = row >> SHIFT;
                if b < to {
                    out[(b - from) as usize] |= 1 << (row & 63);
                }
                b < to
            }),
            Rows::Inline(_) => self.for_each_block(from..to, |b, m, _| out[(b - from) as usize] |= m),
        }
    }

    /// The index, among the term's postings, of its first row in block `b` (or the first after it).
    pub(crate) fn first(&self, b: u32) -> usize {
        match &self.rows {
            Rows::Bitmap { words, rank } => {
                let b = b as usize;
                rank[b >> 3] as usize + words[b & !7..b].iter().map(|w| w.count_ones() as usize).sum::<usize>()
            }
            Rows::Bytes { present, at, groups } => {
                // rows before this group of eight blocks, plus those in its row bytes up to this block
                let (g, n) = (b as usize >> 3, groups_before(present, b as usize));
                let (o, mut rows) = (at[2 * g] as usize, at[2 * g + 1] as usize);
                for i in (0..n).step_by(8) {
                    let part = word_at(groups, o + i);
                    rows += (if n - i < 8 { part & ((1u64 << (8 * (n - i))) - 1) } else { part }).count_ones() as usize;
                }
                rows
            }
            _ => self.block(b).map_or(0, |c| c.1), // the other forms always say
        }
    }

    /// Calls `f(block, rows mask, index of the first posting)` for each of `blocks` the term is in,
    /// in order.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn for_each_block(&self, blocks: std::ops::Range<u32>, mut f: impl FnMut(u32, u64, usize)) {
        match &self.rows {
            Rows::Inline(_) => {
                let mut last = NONE;
                for i in 0..self.df {
                    let Rows::Inline(r) = &self.rows else { unreachable!() };
                    let b = r[i] >> SHIFT;
                    if b != last && blocks.contains(&b) {
                        let (m, first) = self.block(b).expect("the row is in this block");
                        f(b, m, first);
                    }
                    last = b;
                }
            }
            Rows::Bitmap { words, rank } => {
                let start = blocks.start as usize;
                if start >= words.len() {
                    return;
                }
                let before: u32 = words[start & !7..start].iter().map(|w| w.count_ones()).sum();
                let mut idx = (rank[start >> 3] + before) as usize;
                for b in start..(blocks.end as usize).min(words.len()) {
                    let w = words[b];
                    if w != 0 {
                        f(b as u32, w, idx);
                        idx += w.count_ones() as usize;
                    }
                }
            }
            Rows::Bytes { present, at, groups } => {
                let (start, end) = (blocks.start as usize, (blocks.end as usize).min(present.len())); // blocks past the last are absent
                if start >= end {
                    return;
                }
                let mut idx = self.first(blocks.start);
                let mut o = at[2 * (start >> 3)] as usize + groups_before(present, start);
                for b in start..end {
                    let p = present[b];
                    if p != 0 {
                        let m = spread(word_at(groups, o), p);
                        f(b as u32, m, idx);
                        o += p.count_ones() as usize;
                        idx += m.count_ones() as usize;
                    }
                }
            }
            Rows::Ef(ef) if ef.l == SHIFT => ef.runs(blocks.start as usize, |b, first, n| {
                if b as u32 >= blocks.end {
                    return false;
                }
                f(b as u32, low_mask(ef, first, n), first);
                true
            }),
            Rows::Ef(ef) => {
                // a bucket spans several blocks: split its rows by block
                let sub_bits = ef.l - SHIFT;
                ef.runs((blocks.start >> sub_bits) as usize, |h, first, n| {
                    let (mut cur, mut mask, mut at) = (NONE, 0u64, 0);
                    for i in first..first + n {
                        let low = ef.low(i);
                        let b = (h as u32) << sub_bits | low >> SHIFT;
                        if b != cur {
                            if cur != NONE && cur >= blocks.start {
                                f(cur, mask, at);
                            }
                            if b >= blocks.end {
                                return false;
                            }
                            (cur, mask, at) = (b, 0, i);
                        }
                        mask |= 1 << (low & 63);
                    }
                    if cur >= blocks.start {
                        f(cur, mask, at); // a bucket handed over is not empty
                    }
                    true
                });
            }
        }
    }

    /// The tf of posting `i`.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn tf(&self, i: usize) -> u32 {
        let big = |big: &[u64]| {
            let at = big.partition_point(|&e| (e >> 32) < i as u64);
            big[at] as u32 // an escaped posting always has an entry
        };
        match &self.tfs {
            Tfs::Ones => 1,
            Tfs::Inline(t) => t[i],
            Tfs::Bytes { bytes, big: table } => match (bytes[i >> 3] >> (8 * (i & 7))) as u8 {
                u8::MAX => big(table),
                tf => tf as u32,
            },
            Tfs::Small { codes, nibs, big: table, .. } => {
                let code = bits::get(codes, 2 * i, 2);
                if code < 3 {
                    return code + 1;
                }
                match bits::get(nibs, 4 * self.escapes_before(i), 4) {
                    15 => big(table),
                    nib => 4 + nib,
                }
            }
        }
    }

    /// Like `for_each_block`, also passing the largest tf in the block. One pass over the postings,
    /// their tfs read in sequence: the way to read a term that keeps no bounds.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn for_each_bounded(&self, blocks: std::ops::Range<u32>, mut f: impl FnMut(u32, u64, usize, u32)) {
        let Rows::Ef(ef) = &self.rows else {
            return self.for_each_block(blocks, |b, m, first| f(b, m, first, self.max_tf(first, m.count_ones() as usize)));
        };
        // Two-bit codes: an escape's value is the next nibble, so count escapes as they pass.
        let mut escapes = None;
        let mut tf = |i: usize| match &self.tfs {
            Tfs::Small { codes, nibs, .. } => match bits::get(codes, 2 * i, 2) {
                3 => {
                    let n = escapes.get_or_insert_with(|| self.escapes_before(i));
                    *n += 1;
                    match bits::get(nibs, 4 * (*n - 1), 4) {
                        15 => self.tf(i),
                        nib => 4 + nib,
                    }
                }
                code => code + 1,
            },
            _ => self.tf(i),
        };
        let (mut cur, mut mask, mut first, mut max) = (NONE, 0u64, 0usize, 0u32);
        ef.scan(blocks.start << SHIFT, |i, row| {
            let b = row >> SHIFT;
            if b != cur {
                if cur != NONE {
                    f(cur, mask, first, max);
                }
                if b >= blocks.end {
                    cur = NONE;
                    return false;
                }
                (cur, mask, first, max) = (b, 0, i, 0);
            }
            mask |= 1 << (row & 63);
            max = max.max(tf(i));
            true
        });
        if cur != NONE {
            f(cur, mask, first, max);
        }
    }

    /// Postings before `i` whose two-bit code is an escape.
    fn escapes_before(&self, i: usize) -> usize {
        let Tfs::Small { codes, rank, .. } = &self.tfs else { return 0 };
        let (mut n, first, last) = (rank[i >> 8] as usize, (i >> 8) * 8, i >> 5);
        for w in first..last {
            n += (codes[w] & codes[w] >> 1 & ESCAPES).count_ones() as usize;
        }
        let part = codes[last] & ((1u64 << (2 * (i & 31))) - 1);
        n + (part & part >> 1 & ESCAPES).count_ones() as usize
    }

    /// The largest tf among postings `first..first + n`.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn max_tf(&self, first: usize, n: usize) -> u32 {
        let (mut max, end) = (0, first + n);
        let mut i = first;
        while i < end {
            // Two-bit codes: 32 postings at a time, unless one of them is an escape.
            let mut k = 1;
            if let Tfs::Small { codes, .. } = &self.tfs {
                k = (end - i).min(32);
                let w = bits::window(codes, 2 * i) & (!0u64 >> (64 - 2 * k));
                if w & w >> 1 & ESCAPES == 0 {
                    max = max.max(if w & ESCAPES << 1 != 0 { 3 } else if w != 0 { 2 } else { 1 });
                    i += k;
                    continue;
                }
            }
            max = (i..i + k).fold(max, |m, j| m.max(self.tf(j)));
            i += k;
        }
        max
    }

    /// The term's bound byte for block `b` (0: absent), if it keeps bounds.
    #[inline]
    pub(crate) fn bound(&self, b: u32) -> Option<u8> {
        self.bounds.map(|s| (s[b as usize >> 3] >> (8 * (b & 7))) as u8)
    }

    /// The largest bound byte of the term's rows in the blocks `blocks`: from its stretches' bytes
    /// where it keeps them, else `top` (rows may be absent: this only bounds).
    #[inline]
    pub(crate) fn bound_over(&self, blocks: std::ops::Range<u32>) -> u8 {
        match self.segs {
            Some((segs, shift)) if !blocks.is_empty() => {
                segs[(blocks.start >> shift) as usize..=((blocks.end - 1) >> shift) as usize].iter().copied().max().unwrap_or(0)
            }
            _ => self.top,
        }
    }

    /// The blocks of each stretch the term keeps a byte for, in order (all of `blocks` as one when
    /// it keeps none).
    pub(crate) fn stretches(&self, blocks: u32) -> impl Iterator<Item = std::ops::Range<u32>> {
        let shift = self.segs.map_or(32, |(_, shift)| shift);
        (0..blocks.div_ceil(1u32.checked_shl(shift).unwrap_or(u32::MAX).max(1)))
            .map(move |g| g.checked_shl(shift).unwrap_or(0)..(g + 1).checked_shl(shift).unwrap_or(blocks).min(blocks))
    }

    /// The largest tf in block `b`, given what `block` returned for it.
    #[inline]
    pub(crate) fn block_tf(&self, b: u32, mask: u64, first: usize) -> u32 {
        self.max_tf(if first == UNKNOWN { self.first(b) } else { first }, mask.count_ones() as usize)
    }

    /// Calls `f(row, tf, index)` for every posting, in row order.
    pub(crate) fn for_each(&self, blocks: u32, mut f: impl FnMut(u32, u32, usize)) {
        self.for_each_block(0..blocks, |b, mut m, mut i| {
            while m != 0 {
                f(b << SHIFT | m.trailing_zeros(), self.tf(i), i);
                i += 1;
                m &= m - 1;
            }
        });
    }

    /// Replaces `out` with posting `i`'s positions, ascending; `tf` is its tf. Empty if none are
    /// stored. `at` remembers where the last posting read was, so reading postings in ascending
    /// order doesn't walk a chunk again for each.
    pub(crate) fn positions(&self, i: usize, tf: u32, at: &mut PosCursor, out: &mut Vec<u32>) {
        out.clear();
        match self.positions {
            Positions::None => {}
            Positions::Inline { lex, at } => {
                let (mut pos, mut last) = (at[i], 0);
                for _ in 0..tf {
                    last += get_varint(lex, &mut pos) as u32;
                    out.push(last);
                }
            }
            Positions::Packed { chunks, data } => {
                let width = (chunks[i >> 6] >> 56) as u32;
                // A posting's bits start where the previous one's end: count from the chunk's start,
                // or from the last posting read if that is in this chunk and not past this one.
                if at.posting >> 6 != i >> 6 || at.posting > i {
                    *at = PosCursor { posting: i & !63, bit: (chunks[i >> 6] & (!0 >> 8)) as usize };
                }
                at.bit += self.pos_bits_between(at.posting, i, width);
                at.posting = i;
                read_pos(data, at.bit, tf, width, out);
            }
        }
    }

    /// Bits the positions of postings `from..to` take, `width` wide: by their tfs, which for two-bit
    /// codes are counted a word (32 postings) at a time, the rare escapes read one by one.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    fn pos_bits_between(&self, from: usize, to: usize, width: u32) -> usize {
        let Tfs::Small { codes, nibs, .. } = &self.tfs else {
            return match self.tfs {
                Tfs::Ones => (to - from) * width as usize,
                _ => (from..to).map(|j| pos_bits(self.tf(j), width)).sum(),
            };
        };
        let (b1, b2, b3) = (pos_bits(1, width), pos_bits(2, width), pos_bits(3, width));
        let (mut bits, mut i, mut nib) = (0, from, None);
        while i < to {
            let k = (to - i).min(32);
            let w = bits::window(codes, 2 * i) & (!0u64 >> (64 - 2 * k));
            let (lo, hi) = (w & ESCAPES, w >> 1 & ESCAPES);
            let (twos, threes, mut escapes) = ((lo & !hi).count_ones() as usize, (hi & !lo).count_ones() as usize, lo & hi);
            bits += (k - twos - threes - escapes.count_ones() as usize) * b1 + twos * b2 + threes * b3;
            while escapes != 0 {
                // an escape's tf is 4 plus the next nibble, or in the table when that is 15
                let j = i + escapes.trailing_zeros() as usize / 2;
                let n = nib.get_or_insert_with(|| self.escapes_before(j));
                bits += pos_bits(match bits::get(nibs, 4 * *n, 4) { 15 => self.tf(j), v => 4 + v }, width);
                *n += 1;
                escapes &= escapes - 1;
            }
            i += k;
        }
        bits
    }
}

/// The row mask of one block from the `n` six-bit low parts starting at value `first`.
#[inline]
fn low_mask(ef: &Ef, first: usize, n: usize) -> u64 {
    let (mut m, mut i, end) = (0u64, first, first + n);
    while i < end {
        let (mut w, k) = (ef.lows(i), (end - i).min(10)); // ten six-bit parts per word
        for _ in 0..k {
            m |= 1 << (w & 63);
            w >>= 6;
        }
        i += k;
    }
    m
}

/// Appends a term's postings (`rows` ascending, a tf for each) to `post`; returns where they are.
fn write_term(rows: &[u32], tfs: &[u32], dl: &[u32], avgdl: f64, params: Params, post: &mut Vec<u64>) -> Meta {
    let n_rows = dl.len();
    let df = rows.len();
    let mut m = Meta { df: df as u32, ..Default::default() };
    if df <= 2 {
        for i in 0..df {
            m.inline[i] = (rows[i], tfs[i]);
        }
        return m;
    }
    m.off = post.len() as u64;
    let blocks = n_rows.div_ceil(64);

    // A term in at least a quarter of the blocks keeps bounds (below). In at least half, its
    // Elias-Fano samples every 8th block instead of every 64th bucket, since queries will look it up
    // in most blocks they score.
    let entries = 1 + rows.windows(2).filter(|w| w[0] >> SHIFT != w[1] >> SHIFT).count();
    let summary = 4 * entries >= blocks;
    if 2 * entries >= blocks {
        m.kind |= DENSE;
    }
    let l = (n_rows / df).max(1).ilog2().max(SHIFT);
    let s = ef_samples(m.kind & DENSE != 0, l);
    let rank_words = blocks.div_ceil(8).div_ceil(2);
    let n_groups = 1 + rows.windows(2).filter(|w| w[0] >> 3 != w[1] >> 3).count(); // 8-row groups with a row
    let at_words = blocks.div_ceil(8) + 1; // two u32s for every 8th block, and for the end
    let (bitmap, bytes) = (64 * (blocks + rank_words), 64 * (blocks.div_ceil(8) + at_words + n_groups.div_ceil(8) + 1));
    // The two mask forms decode a block several times faster than Elias-Fano, so the smaller of
    // them is taken when it costs at most 30% more.
    if bitmap.min(bytes) * 10 > Ef::bits(df, n_rows, l, s) * 13 {
        m.l = l as u8;
        Ef::write(rows, n_rows, l, s, post);
    } else if bitmap <= bytes {
        m.kind |= FORM_BITMAP;
        let at = post.len();
        post.resize(at + blocks + rank_words, 0);
        for &r in rows {
            post[at + (r >> SHIFT) as usize] |= 1 << (r & 63);
        }
        let mut before = 0u64;
        for g in 0..blocks.div_ceil(8) {
            post[at + blocks + g / 2] |= before << (32 * (g % 2));
            before += post[at + 8 * g..at + (8 * g + 8).min(blocks)].iter().map(|w| w.count_ones() as u64).sum::<u64>();
        }
    } else {
        m.kind |= FORM_BYTES;
        let present = post.len();
        post.resize(present + blocks.div_ceil(8) + at_words + n_groups.div_ceil(8) + 1, 0); // a spare word: see Rows::Bytes
        let at = present + blocks.div_ceil(8);
        let groups = at + at_words;
        let (mut n, mut last) = (0usize, NONE); // row bytes so far; the last group written
        let mut sampled = 0; // blocks whose samples are written: (row bytes, rows) before every 8th
        let mut sample = |post: &mut Vec<u64>, upto: usize, n: usize, i: usize| {
            while sampled < upto {
                if sampled % 8 == 0 {
                    post[at + sampled / 8] = n as u64 | (i as u64) << 32;
                }
                sampled += 1;
            }
        };
        for (i, &r) in rows.iter().enumerate() {
            let b = (r >> SHIFT) as usize;
            sample(post, b + 1, n, i);
            if r >> 3 != last {
                post[present + b / 8] |= 1 << (8 * (b % 8) as u32 + (r >> 3 & 7));
                (n, last) = (n + 1, r >> 3);
            }
            post[groups + (n - 1) / 8] |= 1 << (8 * ((n - 1) % 8) as u32 + (r & 7));
        }
        sample(post, blocks, n, df);
        post[at + blocks.div_ceil(8)] = n as u64 | (df as u64) << 32; // the totals
    }

    // Frequencies: nothing when all are 1, else the smaller of two-bit codes and bytes.
    if tfs.iter().any(|&tf| tf != 1) {
        let count = |min: u32| tfs.iter().filter(|&&tf| tf >= min).count();
        let (n_esc, big_small, big_bytes) = (count(4), count(19), count(255));
        let small = 2 * df + 32 * df.div_ceil(256) + 4 * n_esc + 64 * big_small;
        let (kind, cap) = if small <= 8 * df + 64 * big_bytes { (TF_SMALL, 19) } else { (TF_BYTES, 255) };
        m.kind |= kind;
        let mut p = Packer::new(post);
        if kind == TF_SMALL {
            (m.n_esc, m.n_big) = (n_esc as u32, big_small as u32);
            tfs.iter().for_each(|&tf| p.push(tf.min(4) - 1, 2));
            let mut escapes = 0u32;
            let rank: Vec<u32> = tfs
                .chunks(256)
                .map(|c| {
                    let before = escapes;
                    escapes += c.iter().filter(|&&tf| tf >= 4).count() as u32;
                    before
                })
                .collect();
            let mut p = Packer::new(post);
            rank.iter().for_each(|&r| p.push(r, 32));
            let mut p = Packer::new(post);
            tfs.iter().filter(|&&tf| tf >= 4).for_each(|&tf| p.push((tf - 4).min(15), 4));
        } else {
            m.n_big = big_bytes as u32;
            tfs.iter().for_each(|&tf| p.push(tf.min(255), 8));
        }
        post.extend(tfs.iter().enumerate().filter(|(_, &tf)| tf >= cap).map(|(i, &tf)| (i as u64) << 32 | tf as u64));
    }

    // Bounds: the largest tf_part of the term's rows, over all of them and, for terms in at least a
    // quarter of the blocks, per block; for other Elias-Fano terms, per sampled stretch.
    let at = post.len();
    if summary {
        m.kind |= BOUNDS;
        post.resize(at + blocks.div_ceil(8), 0);
    }
    let stretches = stretches(m.kind, n_rows, m.l as u32);
    let segs = post.len();
    if let Some((n, _)) = stretches {
        post.resize(segs + n.div_ceil(8), 0);
    }
    let put = |post: &mut Vec<u64>, at: usize, i: usize, byte: u8| {
        let (w, o) = (at + i / 8, 8 * (i % 8));
        post[w] = post[w] & !(255 << o) | (post[w] >> o & 255).max(byte as u64) << o;
    };
    for (&r, &tf) in rows.iter().zip(tfs) {
        let byte = bound_byte(tf, dl[r as usize], avgdl, params);
        m.top = m.top.max(byte);
        if summary {
            put(post, at, (r >> SHIFT) as usize, byte);
        }
        if let Some((_, shift)) = stretches {
            put(post, segs, (r >> SHIFT >> shift) as usize, byte);
        }
    }
    m
}

/// For an Elias-Fano term without bound bytes: how many stretches of 2^s buckets (one Elias-Fano
/// sample each) its rows span, and log2 of the blocks one covers, if more than one. Each gets the
/// largest bound byte of its rows, a block bound for terms too sparse to keep one per block: about
/// one byte per 64 to 128 postings.
fn stretches(kind: u8, n_rows: usize, l: u32) -> Option<(usize, u32)> {
    if kind & (FORM_MASK | BOUNDS) != 0 {
        return None; // a mask form, or bound bytes
    }
    let s = ef_samples(false, l);
    let buckets = ((n_rows.max(1) - 1) >> l) + 1;
    let n = buckets.div_ceil(1 << s);
    (n > 1).then_some((n, s + l - SHIFT))
}

/// Appends a term's positions (each posting's, back to back, in posting order) to `out`.
fn write_positions(rows: &[u32], tfs: &[u32], positions: &[u32], dl: &[u32], out: &mut Vec<u64>) {
    let chunks = out.len();
    out.resize(chunks + rows.len().div_ceil(64), 0);
    let (data, mut bits, mut at) = (out.len(), 0usize, 0usize); // first data word; bits written; positions consumed
    for (c, (rows, tfs)) in rows.chunks(64).zip(tfs.chunks(64)).enumerate() {
        // wide enough for any position in the chunk's longest row
        let longest = rows.iter().map(|&r| dl[r as usize]).max().expect("chunks are not empty");
        let width = (32 - (longest.max(2) - 1).leading_zeros()).max(1);
        out[chunks + c] = bits as u64 | (width as u64) << 56;
        for &tf in tfs {
            // each posting starts where the last ended, so pack them one after another
            out.truncate(data + bits.div_ceil(64));
            let mut p = Packer::resume(out, data, bits);
            write_pos(&mut p, &positions[at..at + tf as usize], width);
            bits += pos_bits(tf, width);
            at += tf as usize;
        }
    }
    out.resize(data + bits.div_ceil(64), 0);
}

/// Elias-Fano's sample interval (as a shift): every 8th bucket when buckets are blocks and the term
/// is in most of them, else every 64th.
fn ef_samples(dense: bool, l: u32) -> u32 {
    if dense && l == SHIFT {
        3
    } else {
        6
    }
}

/// Builds the sorted, front-coded lexicon: terms must come in ascending order.
#[derive(Default)]
struct LexWriter {
    heads: Vec<u32>, // where each group of LEX_GROUP terms starts in `bytes`
    bytes: Vec<u8>,
    prev: Vec<u8>,
    n: usize,
}

impl LexWriter {
    fn push(&mut self, term: &[u8], meta: &Meta, positions: Option<&[u32]>) {
        let shared = if self.n.is_multiple_of(LEX_GROUP) {
            self.heads.push(self.bytes.len() as u32);
            0
        } else {
            self.prev.iter().zip(term).take_while(|(a, b)| a == b).count()
        };
        put_varint(&mut self.bytes, shared as u64);
        put_varint(&mut self.bytes, (term.len() - shared) as u64);
        self.bytes.extend_from_slice(&term[shared..]);
        meta.write(&mut self.bytes, positions);
        self.prev.clear();
        self.prev.extend_from_slice(term);
        self.n += 1;
    }
}

pub struct Index {
    lex_heads: Arr<u32>,
    lex: Arr<u8>,
    post: Arr<u64>,
    pub(crate) dl: Arr<u32>,  // exact length of each row
    pub(crate) ids: Arr<u64>, // each row's id
    pub(crate) blen: Arr<u8>, // per block, the length class of its shortest row
    pos: Option<Arr<u64>>,    // positions, if stored
    n_terms: usize,
    order: Order,
    pub(crate) stats: Stats,
    crcs: Option<[u32; SECTIONS]>, // of each section, when opened from a file
    by_id: std::sync::OnceLock<Vec<u32>>, // rows sorted by id, built when a host first asks for a row
}

const MAGIC: &[u8; 8] = b"BITBM25\0";
const VERSION: u32 = 4;
const SECTIONS: usize = 7;
// Header: magic, version u32, section count u32, k1 f64, b f64, block shift u32, flags u32 (bit 0:
// rows as added; bit 1: positions), rows u64, total length u64, terms u64, then per section (offset u64, bytes u64, crc32 u32,
// zero u32), then a crc32 of all of that. Sections follow, each 8-byte aligned.
pub(crate) const TABLE_AT: usize = 64;
const HEADER_CRC_AT: usize = TABLE_AT + 24 * SECTIONS;
const DATA_AT: usize = (HEADER_CRC_AT + 4).next_multiple_of(8);

impl Index {
    /// Builds an in-memory index with default params and the `whitespace` analyzer; id = position in
    /// `docs`. Panics past the limits `Builder::add` reports as errors.
    pub fn build<S: AsRef<str>>(docs: &[S]) -> Self {
        let mut b = Builder::new(Params::default(), Analyzer::whitespace()).expect("default params are valid");
        for (id, d) in docs.iter().enumerate() {
            b.add(id as u64, d.as_ref()).expect("within a segment's limits");
        }
        b.finish()
    }

    /// Number of rows, deleted or not.
    pub fn rows(&self) -> usize {
        self.dl.len()
    }

    pub(crate) fn blocks(&self) -> u32 {
        self.dl.len().div_ceil(64) as u32
    }

    /// Each row's id, by row: how a host maps its own ids to rows (to build an excluded-rows bitset).
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    /// The row holding `id`, if any. The first call sorts the rows by id (4 bytes a row, kept).
    pub fn row_of(&self, id: u64) -> Option<usize> {
        let by_id = self.by_id.get_or_init(|| {
            let mut rows: Vec<u32> = (0..self.rows() as u32).collect();
            rows.sort_unstable_by_key(|&r| self.ids[r as usize]);
            rows
        });
        let at = by_id.partition_point(|&r| self.ids[r as usize] < id);
        by_id.get(at).map(|&r| r as usize).filter(|&r| self.ids[r] == id)
    }

    pub fn params(&self) -> Params {
        self.stats.params
    }

    pub fn order(&self) -> Order {
        self.order
    }

    /// Whether positions are stored, which phrase queries need.
    pub fn has_positions(&self) -> bool {
        self.pos.is_some()
    }

    /// Where `term`'s postings are, if the segment has it.
    pub(crate) fn lookup(&self, term: &[u8]) -> Option<Meta> {
        // The last group whose first term is <= `term`, then its entries in order. `matched` is how
        // many leading bytes of `term` the previous (smaller) entry shares with it.
        let first = |g: usize| {
            let mut pos = self.lex_heads[g] as usize + 1; // a group's first entry shares nothing
            let len = get_varint(&self.lex, &mut pos) as usize;
            &self.lex[pos..pos + len]
        };
        let (mut lo, mut hi) = (0, self.lex_heads.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if first(mid) <= term {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let g = lo.checked_sub(1)?;
        let (mut pos, mut matched) = (self.lex_heads[g] as usize, 0);
        for _ in 0..LEX_GROUP.min(self.n_terms - g * LEX_GROUP) {
            let shared = get_varint(&self.lex, &mut pos) as usize;
            let len = get_varint(&self.lex, &mut pos) as usize;
            let suffix = &self.lex[pos..pos + len];
            pos += len;
            let meta = Meta::read(&self.lex, &mut pos, self.pos.is_some());
            if shared < matched {
                return None; // this entry already differs from `term` where the last one agreed
            }
            if shared == matched {
                let rest = &term[matched..];
                let common = suffix.iter().zip(rest).take_while(|(a, b)| a == b).count();
                if common == suffix.len() && common == rest.len() {
                    return Some(meta);
                }
                if common == rest.len() || (common < suffix.len() && suffix[common] > rest[common]) {
                    return None; // past `term`
                }
                matched += common;
            }
        }
        None
    }

    /// The terms in ascending order, one `next` at a time.
    pub(crate) fn terms(&self) -> TermIter<'_> {
        TermIter { idx: self, pos: 0, left: self.n_terms, term: Vec::new() }
    }

    pub(crate) fn postings(&self, m: &Meta) -> Postings<'_> {
        self.postings_sized(m).0
    }

    /// A term's postings, and the words its rows, frequencies and summary take in `post`.
    pub(crate) fn postings_sized(&self, m: &Meta) -> (Postings<'_>, [usize; 3]) {
        let df = m.df as usize;
        if df <= 2 {
            let (rows, tfs) = ([m.inline[0].0, m.inline[1].0], [m.inline[0].1, m.inline[1].1]);
            let positions = match &self.pos {
                Some(_) => Positions::Inline { lex: &self.lex, at: [m.pos as u32 as usize, (m.pos >> 32) as usize] },
                None => Positions::None,
            };
            let byte = |&(row, tf): &(u32, u32)| bound_byte(tf, self.dl[row as usize], self.stats.avgdl, self.stats.params);
            let top = m.inline[..df].iter().map(byte).max().unwrap_or(0);
            return (Postings { df, rows: Rows::Inline(rows), tfs: Tfs::Inline(tfs), bounds: None, top, segs: None, positions }, [0; 3]);
        }
        let (n_rows, blocks) = (self.rows(), self.rows().div_ceil(64));
        let w = &mut &self.post[m.off as usize..];
        let start = w.len();
        let rows = if m.kind & FORM_MASK == FORM_BITMAP {
            let words = take(w, blocks);
            Rows::Bitmap { words, rank: bits::as_u32(take(w, blocks.div_ceil(8).div_ceil(2))) }
        } else if m.kind & FORM_MASK == FORM_BYTES {
            let present = bits::as_u8(take(w, blocks.div_ceil(8))); // whole words: bytes past the last block are 0
            let at = bits::as_u32(take(w, blocks.div_ceil(8) + 1));
            let n_groups = at[2 * blocks.div_ceil(8)] as usize;
            Rows::Bytes { present, at, groups: bits::as_u8(take(w, n_groups.div_ceil(8) + 1)) }
        } else {
            let (ef, used) = Ef::read(w, df, n_rows, m.l as u32, ef_samples(m.kind & DENSE != 0, m.l as u32));
            take(w, used);
            Rows::Ef(ef)
        };
        let stretches = stretches(m.kind, n_rows, m.l as u32);
        let (big, after_rows) = (m.n_big as usize, w.len());
        let tfs = match m.kind & TF_MASK {
            TF_SMALL => {
                let codes = take(w, (2 * df).div_ceil(64));
                let rank = bits::as_u32(take(w, df.div_ceil(256).div_ceil(2)));
                let nibs = take(w, (4 * m.n_esc as usize).div_ceil(64));
                Tfs::Small { codes, rank, nibs, big: take(w, big) }
            }
            TF_BYTES => Tfs::Bytes { bytes: take(w, df.div_ceil(8)), big: take(w, big) },
            _ => Tfs::Ones,
        };
        let after_tfs = w.len();
        let bounds = (m.kind & BOUNDS != 0).then(|| take(w, blocks.div_ceil(8)));
        let segs = stretches.map(|(n, shift)| (bits::as_u8(take(w, n.div_ceil(8))), shift));
        let positions = match &self.pos {
            Some(pos) => {
                let (chunks, data) = pos[m.pos as usize..].split_at(df.div_ceil(64));
                Positions::Packed { chunks, data }
            }
            None => Positions::None,
        };
        (Postings { df, rows, tfs, bounds, top: m.top, segs, positions }, [start - after_rows, after_rows - after_tfs, after_tfs - w.len()])
    }

    /// Size of the index's arrays, which is also its file size less a few hundred bytes of header.
    pub fn size_bytes(&self) -> usize {
        self.sections().iter().map(|s| s.len()).sum()
    }

    /// Every array as bytes, in file order.
    pub(crate) fn sections(&self) -> [&[u8]; SECTIONS] {
        let pos = self.pos.as_deref().unwrap_or(&[]);
        [as_bytes(&self.lex_heads), as_bytes(&self.lex), as_bytes(&self.post), as_bytes(&self.dl), as_bytes(&self.ids), as_bytes(&self.blen), as_bytes(pos)]
    }

    /// Writes the index to `path`, atomically: it goes to `path.tmp`, is synced, then renamed, so a
    /// crash leaves the old file or the new one, and processes with the old file open keep reading it.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        replace_file(path.as_ref(), |w| self.write_to(w))
    }

    /// Writes the index in its file format: what `save` puts in a file and `from_bytes` reads.
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        let flags = self.order as u32 | (self.pos.is_some() as u32) << 1;
        let parts = self.sections().map(Part::Bytes);
        write_segment(w, self.stats.params, flags, self.dl.len() as u64, self.stats.total_len, self.n_terms as u64, parts)
    }

    /// Opens a saved index by mapping it: nothing is read until queries touch it, so this is
    /// instant at any size, and the OS page cache is the only copy in memory. Checks the header and
    /// that the sections fit together; `verify` checks their contents.
    ///
    /// The file must not be modified while open (`save` never does; it replaces files).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Index> {
        let file = File::open(path)?;
        // SAFETY: see above. Writes to the file by another process would be visible through the
        // map; bitbm25 itself never writes to a file in place.
        Self::from_bytes(Arc::new(unsafe { Mmap::map(&file)? }))
    }

    /// Opens an index from bytes in its file format, held by the host: a mapped file, a cache, any
    /// buffer whose address is 8-byte aligned. Nothing is copied; the index keeps the buffer alive.
    /// Checks the header and that the sections fit together; `verify` checks their contents.
    pub fn from_bytes(buf: Arc<dyn AsRef<[u8]> + Send + Sync>) -> io::Result<Index> {
        let map: Buf = buf;
        let h: &[u8] = (*map).as_ref();
        if !(h.as_ptr() as usize).is_multiple_of(8) {
            return Err(invalid("buffer is not 8-byte aligned"));
        }
        if h.len() < DATA_AT || &h[..8] != MAGIC {
            return Err(invalid("not a bitbm25 index"));
        }
        let u32_at = |o: usize| u32::from_le_bytes(h[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(h[o..o + 8].try_into().unwrap());
        let version = u32_at(8);
        if version != VERSION {
            return Err(invalid(format!("index format version {version}; this build reads version {VERSION}")));
        }
        if u32_at(HEADER_CRC_AT) != crc32fast::hash(&h[..HEADER_CRC_AT]) {
            return Err(invalid("index header is corrupt (checksum mismatch)"));
        }
        let params = Params { k1: f64::from_bits(u64_at(16)), b: f64::from_bits(u64_at(24)) };
        let (shift, flags) = (u32_at(32), u32_at(36));
        if u32_at(12) as usize != SECTIONS || shift != SHIFT || flags > 3 || !params.valid() {
            return Err(invalid(format!("index built with {params:?}, block shift {shift}, flags {flags}")));
        }
        let (rows, total_len, n_terms) = (u64_at(40), u64_at(48), u64_at(56) as usize);
        let sec = |i: usize| (u64_at(TABLE_AT + 24 * i), u64_at(TABLE_AT + 24 * i + 8));
        let crcs = std::array::from_fn(|i| u32_at(TABLE_AT + 24 * i + 16));
        macro_rules! arr {
            ($i:expr) => {{
                let (off, bytes) = sec($i);
                Arr::mapped(&map, off, bytes)?
            }};
        }
        let idx = Index {
            lex_heads: arr!(0),
            lex: arr!(1),
            post: arr!(2),
            dl: arr!(3),
            ids: arr!(4),
            blen: arr!(5),
            pos: (flags & 2 != 0).then_some(arr!(6)),
            n_terms,
            order: if flags & 1 == 0 { Order::Length } else { Order::Added },
            stats: Stats::new(rows as usize, total_len, params),
            crcs: Some(crcs),
            by_id: Default::default(),
        };
        let consistent = idx.dl.len() as u64 == rows
            && idx.ids.len() == idx.dl.len()
            && idx.blen.len() == idx.dl.len().div_ceil(64)
            && idx.lex_heads.len() == n_terms.div_ceil(LEX_GROUP);
        if !consistent {
            return Err(invalid("index sections have inconsistent sizes"));
        }
        Ok(idx)
    }

    /// Reads every byte of an opened index and checks it against the checksums written by `save`.
    /// An index that was built in memory has nothing to check against and always passes.
    pub fn verify(&self) -> io::Result<()> {
        let Some(crcs) = self.crcs else { return Ok(()) };
        for (i, (s, want)) in self.sections().iter().zip(crcs).enumerate() {
            if crc32fast::hash(s) != want {
                return Err(invalid(format!("index section {i} is corrupt (checksum mismatch)")));
            }
        }
        Ok(())
    }
}

/// The first `n` words of `w`, which then starts after them.
fn take<'a>(w: &mut &'a [u64], n: usize) -> &'a [u64] {
    let (head, rest) = w.split_at(n);
    *w = rest;
    head
}

/// A walk over a segment's terms in ascending order.
pub(crate) struct TermIter<'a> {
    idx: &'a Index,
    pos: usize,
    left: usize,
    term: Vec<u8>,
}

impl TermIter<'_> {
    /// Moves to the next term; `term` then names it.
    pub(crate) fn next(&mut self) -> Option<Meta> {
        self.left = self.left.checked_sub(1)?;
        let lex = &self.idx.lex;
        let shared = get_varint(lex, &mut self.pos) as usize;
        let len = get_varint(lex, &mut self.pos) as usize;
        self.term.truncate(shared);
        self.term.extend_from_slice(&lex[self.pos..self.pos + len]);
        self.pos += len;
        Some(Meta::read(lex, &mut self.pos, self.idx.pos.is_some()))
    }

    pub(crate) fn term(&self) -> &[u8] {
        &self.term
    }
}

/// Where an assembler's postings (or positions) go: memory, or a file that is copied into the
/// segment file at the end, so that assembling needs memory for one term at a time.
enum Words {
    Mem(Vec<u64>),
    File { w: BufWriter<File>, path: PathBuf, words: u64, crc: crc32fast::Hasher },
}

impl Words {
    fn len(&self) -> u64 {
        match self {
            Words::Mem(v) => v.len() as u64,
            Words::File { words, .. } => *words,
        }
    }

    fn put(&mut self, more: &[u64]) -> io::Result<()> {
        match self {
            Words::Mem(v) => v.extend_from_slice(more),
            Words::File { w, words, crc, .. } => {
                w.write_all(as_bytes(more))?;
                crc.update(as_bytes(more));
                *words += more.len() as u64;
            }
        }
        Ok(())
    }
}

/// One section of a segment file about to be written: bytes at hand, or a file of them with its
/// length in bytes and checksum.
enum Part<'a> {
    Bytes(&'a [u8]),
    File(&'a Path, u64, u32),
}

/// Writes a segment file: the header, then each part, 8-byte aligned.
fn write_segment(w: &mut impl Write, params: Params, flags: u32, rows: u64, total_len: u64, terms: u64, parts: [Part; SECTIONS]) -> io::Result<()> {
    let mut h = Vec::with_capacity(DATA_AT);
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&VERSION.to_le_bytes());
    h.extend_from_slice(&(SECTIONS as u32).to_le_bytes());
    h.extend_from_slice(&params.k1.to_le_bytes());
    h.extend_from_slice(&params.b.to_le_bytes());
    h.extend_from_slice(&SHIFT.to_le_bytes());
    h.extend_from_slice(&flags.to_le_bytes());
    h.extend_from_slice(&rows.to_le_bytes());
    h.extend_from_slice(&total_len.to_le_bytes());
    h.extend_from_slice(&terms.to_le_bytes());
    let mut off = DATA_AT as u64;
    for part in &parts {
        let (len, crc) = match part {
            Part::Bytes(s) => (s.len() as u64, crc32fast::hash(s)),
            Part::File(_, len, crc) => (*len, *crc),
        };
        h.extend_from_slice(&off.to_le_bytes());
        h.extend_from_slice(&len.to_le_bytes());
        h.extend_from_slice(&crc.to_le_bytes());
        h.extend_from_slice(&[0; 4]);
        off = (off + len).next_multiple_of(8);
    }
    h.extend_from_slice(&crc32fast::hash(&h).to_le_bytes());
    h.resize(DATA_AT, 0);
    w.write_all(&h)?;
    for part in parts {
        let len = match part {
            Part::Bytes(s) => {
                w.write_all(s)?;
                s.len() as u64
            }
            Part::File(path, len, _) => {
                io::copy(&mut File::open(path)?, w)?;
                len
            }
        };
        w.write_all(&[0; 8][..(len.next_multiple_of(8) - len) as usize])?;
    }
    Ok(())
}

/// Puts a segment together from its rows (already in their final order) and then each term's
/// postings, terms in ascending order.
pub(crate) struct Assembler {
    params: Params,
    order: Order,
    avgdl: f64, // of this segment's rows: what its bound bytes are computed under
    dl: Vec<u32>,
    ids: Vec<u64>,
    post: Words,
    pos: Option<Words>,           // positions, if the segment stores them
    term: (Vec<u64>, Vec<u64>), // the term being added: its postings and positions, before they are appended
    lex: LexWriter,
}

impl Assembler {
    pub(crate) fn new(params: Params, order: Order, positions: bool, dl: Vec<u32>, ids: Vec<u64>) -> Self {
        let avgdl = Stats::new(dl.len(), dl.iter().map(|&d| d as u64).sum(), params).avgdl;
        let (post, pos) = (Words::Mem(Vec::new()), positions.then(|| Words::Mem(Vec::new())));
        Assembler { params, order, avgdl, dl, ids, post, pos, term: Default::default(), lex: LexWriter::default() }
    }

    /// Sends postings and positions to files in `dir` instead of memory; `finish_file` then copies
    /// them into the segment file. Call before adding terms.
    pub(crate) fn spill(mut self, dir: &Path) -> io::Result<Self> {
        let file = |name: &str| {
            let path = dir.join(name);
            let w = BufWriter::with_capacity(1 << 20, File::create(&path)?);
            Ok::<_, io::Error>(Words::File { w, path, words: 0, crc: crc32fast::Hasher::new() })
        };
        self.post = file("post")?;
        if self.pos.is_some() {
            self.pos = Some(file("pos")?);
        }
        Ok(self)
    }

    /// Adds a term: its rows ascending, a tf for each and, if the segment stores positions, each
    /// row's positions ascending, back to back. Terms must be added in ascending order.
    pub(crate) fn term(&mut self, name: &[u8], rows: &[u32], tfs: &[u32], positions: &[u32]) -> io::Result<()> {
        let (post, pos) = &mut self.term;
        post.clear();
        let mut meta = write_term(rows, tfs, &self.dl, self.avgdl, self.params, post);
        meta.off += self.post.len(); // write_term counted from the start of this term's words
        self.post.put(post)?;
        if let Some(out) = self.pos.as_mut().filter(|_| rows.len() > 2) {
            pos.clear();
            write_positions(rows, tfs, positions, &self.dl, pos);
            meta.pos = out.len();
            out.put(pos)?;
        }
        self.lex.push(name, &meta, self.pos.as_ref().map(|_| positions));
        Ok(())
    }

    fn block_lengths(&self) -> Vec<u8> {
        self.dl.chunks(64).map(|c| len_class(*c.iter().min().expect("chunks are not empty"))).collect()
    }

    /// The segment, in memory. Only for an assembler that was not spilled.
    pub(crate) fn finish(self) -> Index {
        let total_len = self.dl.iter().map(|&d| d as u64).sum();
        let blen = self.block_lengths();
        let words = |w: Words| match w {
            Words::Mem(v) => v,
            Words::File { .. } => unreachable!("a spilled assembler is finished with finish_file"),
        };
        Index {
            lex_heads: self.lex.heads.into(),
            lex: self.lex.bytes.into(),
            post: words(self.post).into(),
            stats: Stats::new(self.dl.len(), total_len, self.params),
            dl: self.dl.into(),
            ids: self.ids.into(),
            blen: blen.into(),
            pos: self.pos.map(|p| words(p).into()),
            n_terms: self.lex.n,
            order: self.order,
            crcs: None,
            by_id: Default::default(),
        }
    }

    /// Writes the segment to `path` (atomically, as `Index::save` does) without holding its
    /// postings or positions in memory, if the assembler was spilled.
    pub(crate) fn finish_file(self, path: &Path) -> io::Result<()> {
        let blen = self.block_lengths();
        let flags = self.order as u32 | (self.pos.is_some() as u32) << 1;
        let total_len = self.dl.iter().map(|&d| d as u64).sum();
        // close the spill files; what is left of each is its path, length and checksum
        let close = |w: Words| match w {
            Words::Mem(v) => Ok((None, v)),
            Words::File { w, path, words, crc } => {
                w.into_inner().map_err(|e| e.into_error())?;
                Ok::<_, io::Error>((Some((path, words * 8, crc.finalize())), Vec::new()))
            }
        };
        let (post, pos) = (close(self.post)?, self.pos.map(close).transpose()?.unwrap_or_default());
        fn part(closed: &(Option<(PathBuf, u64, u32)>, Vec<u64>)) -> Part<'_> {
            match closed {
                (Some((path, len, crc)), _) => Part::File(path, *len, *crc),
                (None, mem) => Part::Bytes(as_bytes(mem)),
            }
        }
        replace_file(path, |w| {
            let parts = [
                Part::Bytes(as_bytes(&self.lex.heads)),
                Part::Bytes(&self.lex.bytes),
                part(&post),
                Part::Bytes(as_bytes(&self.dl)),
                Part::Bytes(as_bytes(&self.ids)),
                Part::Bytes(&blen),
                part(&pos),
            ];
            write_segment(w, self.params, flags, self.dl.len() as u64, total_len, self.lex.n as u64, parts)
        })?;
        for (file, _) in [post, pos] {
            if let Some((path, ..)) = file {
                std::fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

/// Per-term state while building: per row, (row delta, tf, then the gaps between its positions if
/// they are kept) as LEB128 varints, which is far smaller than the final arrays' growth slack would be.
struct BuildTerm {
    bytes: Vec<u8>,
    last: u32, // last row added, or NONE
}

/// Streaming index construction: docs are added one at a time and not retained, so a corpus never
/// has to be in memory alongside its index. Ids are the host's, in any order.
pub struct Builder {
    params: Params,
    order: Order,
    positions: bool,
    analyzer: Analyzer,
    text: String,                       // one doc's terms, back to back, reused
    spans: Vec<std::ops::Range<usize>>, // each term's place in `text`, in order of position
    by_text: Vec<u32>,                  // positions, sorted by their term
    terms: std::collections::HashMap<Vec<u8>, u32>,
    postings: Vec<BuildTerm>,
    dl: Vec<u32>,
    ids: Vec<u64>,
}

impl Builder {
    /// A builder for a segment scored with `params`, turning text into terms with `analyzer`.
    pub fn new(params: Params, analyzer: Analyzer) -> io::Result<Builder> {
        if !params.valid() {
            return Err(bad_input(format!("k1 must be in [0, 1000] and b in [0, 1]; got {params:?}")));
        }
        Ok(Builder {
            params,
            order: Order::default(),
            positions: true,
            analyzer,
            text: String::new(),
            spans: Vec::new(),
            by_text: Vec::new(),
            terms: Default::default(),
            postings: Vec::new(),
            dl: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// How the segment will number its rows (`Order::Length` unless set).
    pub fn order(mut self, order: Order) -> Builder {
        self.order = order;
        self
    }

    /// Whether to store positions (they are unless turned off). Without them the index is about
    /// half the size and phrase queries are refused.
    pub fn positions(mut self, positions: bool) -> Builder {
        self.positions = positions;
        self
    }

    /// Adds `text` as a row with the host's `id`. Fails, changing nothing, past a segment's limits:
    /// 2^32 - 1 rows, terms in a row, or distinct terms.
    pub fn add(&mut self, id: u64, text: &str) -> io::Result<()> {
        if self.dl.len() >= NONE as usize {
            return Err(bad_input("a segment holds at most 2^32 - 1 rows"));
        }
        let (buf, spans) = (&mut self.text, &mut self.spans);
        buf.clear();
        spans.clear();
        self.analyzer.for_each_term(text, |t| {
            buf.push_str(t);
            spans.push(buf.len() - t.len()..buf.len());
        });
        let len = u32::try_from(spans.len()).map_err(|_| bad_input("a row holds at most 2^32 - 1 terms"))?;
        let (buf, spans) = (&self.text, &self.spans);
        // positions grouped by term, ascending within each
        let by_text = &mut self.by_text;
        by_text.clear();
        by_text.extend(0..len);
        by_text.sort_unstable_by(|&a, &b| buf[spans[a as usize].clone()].cmp(&buf[spans[b as usize].clone()]).then(a.cmp(&b)));
        let same = |a: &u32, b: &u32| buf[spans[*a as usize].clone()] == buf[spans[*b as usize].clone()];
        let new_terms = by_text.chunk_by(same).filter(|r| !self.terms.contains_key(buf[spans[r[0] as usize].clone()].as_bytes())).count();
        if self.postings.len() + new_terms > NONE as usize {
            return Err(bad_input("a segment holds at most 2^32 - 1 distinct terms"));
        }
        let row = self.dl.len() as u32;
        self.dl.push(len);
        self.ids.push(id);
        for run in by_text.chunk_by(same) {
            let name = buf[spans[run[0] as usize].clone()].as_bytes();
            let term = match self.terms.get(name) {
                Some(&t) => t,
                None => {
                    self.postings.push(BuildTerm { bytes: Vec::new(), last: NONE });
                    self.terms.insert(name.to_vec(), self.postings.len() as u32 - 1);
                    self.postings.len() as u32 - 1
                }
            };
            let p = &mut self.postings[term as usize];
            put_varint(&mut p.bytes, if p.last == NONE { row } else { row - p.last } as u64);
            put_varint(&mut p.bytes, run.len() as u64);
            if self.positions {
                let mut last = 0;
                for &at in run {
                    put_varint(&mut p.bytes, (at - last) as u64);
                    last = at;
                }
            }
            p.last = row;
        }
        Ok(())
    }

    /// Rows added so far.
    pub fn rows(&self) -> usize {
        self.dl.len()
    }

    pub fn finish(self) -> Index {
        let Builder { params, order, positions, terms, mut postings, dl, ids, .. } = self;
        // Final row of each added row: by length (ties in the order added), or as added.
        let mut by_new: Vec<u32> = (0..dl.len() as u32).collect();
        if order == Order::Length {
            by_new.sort_by_key(|&r| dl[r as usize]);
        }
        let mut new_of = vec![0u32; dl.len()];
        by_new.iter().enumerate().for_each(|(new, &old)| new_of[old as usize] = new as u32);
        let mut asm = Assembler::new(
            params,
            order,
            positions,
            by_new.iter().map(|&r| dl[r as usize]).collect(),
            by_new.iter().map(|&r| ids[r as usize]).collect(),
        );
        let mut names: Vec<(Vec<u8>, u32)> = terms.into_iter().collect();
        names.sort_unstable();
        // One term's postings as (new row, tf, where its positions start in `at`), then in row order.
        let (mut entries, mut at) = (Vec::new(), Vec::new());
        let (mut rows, mut tfs, mut sorted) = (Vec::new(), Vec::new(), Vec::new());
        for (name, t) in names {
            let bytes = std::mem::take(&mut postings[t as usize].bytes); // freed after this term
            let (mut pos, mut row) = (0, 0u32);
            entries.clear();
            at.clear();
            while pos < bytes.len() {
                row += get_varint(&bytes, &mut pos) as u32;
                let tf = get_varint(&bytes, &mut pos) as u32;
                entries.push((new_of[row as usize], tf, at.len()));
                if positions {
                    let mut last = 0;
                    for _ in 0..tf {
                        last += get_varint(&bytes, &mut pos) as u32;
                        at.push(last);
                    }
                }
            }
            if order == Order::Length {
                entries.sort_unstable();
            }
            rows.clear();
            tfs.clear();
            sorted.clear();
            for &(row, tf, start) in &entries {
                rows.push(row);
                tfs.push(tf);
                if positions {
                    sorted.extend_from_slice(&at[start..start + tf as usize]);
                }
            }
            asm.term(&name, &rows, &tfs, &sorted).expect("appending to memory can't fail");
        }
        asm.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_classes_bound_from_below() {
        assert!(EDGES.windows(2).all(|w| w[0] < w[1]));
        for dl in (0..5000).chain([u32::MAX / 2, u32::MAX]) {
            let c = len_class(dl) as usize;
            assert!(EDGES[c] <= dl && (c == 255 || dl < EDGES[c + 1]), "dl {dl} class {c}");
        }
    }

    /// Every posting shape reads back: densities from two rows to all rows, tfs across every code.
    #[test]
    fn postings_read_back() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for n_rows in [1usize, 64, 65, 1000, 5000] {
            let mut cases = Vec::new(); // (rows, tfs) per term
            for (keep_one_in, tf_range) in [(1u64, 1u64), (1, 300), (2, 3), (3, 20), (5, 4), (10, 1), (10, 5), (20, 2), (30, 9), (50, 300), (200, 2), (1000, 1), (1000, 40)] {
                let rows: Vec<u32> = (0..n_rows as u32).filter(|_| next() % keep_one_in == 0).collect();
                let tfs: Vec<u32> = rows.iter().map(|_| 1 + (next() % tf_range) as u32 * (next() % 3 == 0) as u32).collect();
                cases.push((rows, tfs));
            }
            cases.push((vec![0], vec![7]));
            cases.push((vec![n_rows as u32 - 1], vec![1]));
            // rows of every length, and for each posting `tf` ascending positions below its row's length
            let dl: Vec<u32> = (0..n_rows).map(|r| 400 + (r as u32 * 7919) % 3000).collect();
            let positions: Vec<Vec<u32>> = cases
                .iter()
                .map(|(rows, tfs)| {
                    let mut out = Vec::new();
                    for (&row, &tf) in rows.iter().zip(tfs) {
                        let (len, mut at) = (dl[row as usize], next() as u32 % 3);
                        for k in 0..tf {
                            out.push(at.min(len - tf + k)); // room is left for the rest
                            at = out[out.len() - 1] + 1 + next() as u32 % (len / tf).max(1);
                        }
                    }
                    out
                })
                .collect();
            let mut asm = Assembler::new(Params::default(), Order::Added, true, dl, (0..n_rows as u64).collect());
            for (i, (rows, tfs)) in cases.iter().enumerate() {
                asm.term(format!("term{i:03}").as_bytes(), rows, tfs, &positions[i]).unwrap();
            }
            let idx = asm.finish();
            let blocks = idx.blocks();
            assert!(idx.lookup(b"term").is_none() && idx.lookup(b"term0000").is_none() && idx.lookup(b"zzz").is_none());
            let (mut terms, mut seen) = (idx.terms(), 0);
            while let Some(meta) = terms.next() {
                assert_eq!(idx.lookup(terms.term()), Some(meta));
                seen += 1;
            }
            assert_eq!(seen, cases.len());
            for (i, (rows, tfs)) in cases.iter().enumerate() {
                let meta = idx.lookup(format!("term{i:03}").as_bytes()).unwrap_or_else(|| panic!("term {i} of {n_rows} rows"));
                let p = idx.postings(&meta);
                assert_eq!(p.df, rows.len());
                let (mut got, mut got_pos, mut buf, mut cursor) = (Vec::new(), Vec::new(), Vec::new(), PosCursor::default());
                p.for_each(blocks, |row, tf, at| {
                    got.push((row, tf));
                    p.positions(at, tf, &mut cursor, &mut buf);
                    got_pos.extend_from_slice(&buf);
                });
                // in any order too: every 7th posting, backwards
                let mut start = 0;
                let starts: Vec<usize> = tfs.iter().map(|&tf| (start, start += tf as usize).0).collect();
                for at in (0..rows.len()).rev().step_by(7) {
                    p.positions(at, tfs[at], &mut cursor, &mut buf);
                    assert_eq!(buf, positions[i][starts[at]..starts[at] + tfs[at] as usize], "posting {at} of term {i}");
                }
                let want: Vec<(u32, u32)> = rows.iter().copied().zip(tfs.iter().copied()).collect();
                assert_eq!(got, want, "term {i} of {n_rows} rows");
                assert_eq!(got_pos, positions[i], "positions of term {i} of {n_rows} rows");
                for b in 0..blocks {
                    let in_block: Vec<usize> = (0..rows.len()).filter(|&j| rows[j] >> SHIFT == b).collect();
                    let mask = in_block.iter().fold(0u64, |m, &j| m | 1 << (rows[j] & 63));
                    let known = |c: (u64, usize)| (c.0, if c.1 == UNKNOWN { p.first(b) } else { c.1 });
                    assert_eq!(p.block(b).map(known), in_block.first().map(|&first| (mask, first)), "term {i} block {b}");
                    if let Some(&first) = in_block.first() {
                        let max = in_block.iter().map(|&j| tfs[j]).max().unwrap();
                        assert_eq!(p.block_tf(b, mask, first), max, "term {i} block {b}");
                    }
                    if let Some(byte) = p.bound(b) {
                        assert_eq!(byte == 0, in_block.is_empty(), "term {i} block {b}");
                    }
                    // a scan may start at any block
                    let mut ored = vec![0u64; (blocks - b).min(70) as usize];
                    p.or_into(b, &mut ored);
                    let want: Vec<u64> = (b..b + ored.len() as u32).map(|b| p.block(b).map_or(0, |c| c.0)).collect();
                    assert_eq!(ored, want, "term {i} from block {b}");
                    let mut bounded = Vec::new();
                    p.for_each_bounded(b..blocks, |b, m, first, tf| {
                        assert_eq!(tf, p.max_tf(first, m.count_ones() as usize), "term {i} block {b}");
                        bounded.push((b, m, first));
                    });
                    let mut from_here = Vec::new();
                    p.for_each_block(b..blocks, |b, m, first| from_here.push((b, m, first)));
                    let want: Vec<(u32, u64, usize)> = (b..blocks).filter_map(|b| p.block(b).map(|(m, _)| (b, m, p.first(b)))).collect();
                    assert_eq!(from_here, want, "term {i} from block {b}");
                    assert_eq!(bounded, want, "term {i} from block {b}");
                }
            }
        }
    }
}
