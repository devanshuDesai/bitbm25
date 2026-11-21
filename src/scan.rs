//! Runs a planned query over one segment.
//!
//! Blocks are handled a window at a time (`WINDOW` blocks), so per-query state stays in L1 however
//! large the segment is. For ranked ORs and trees:
//!   1. every scoring term adds its bound for each block of the window into a `u16` accumulator.
//!      Terms in most blocks keep a bound byte per block and add sixteen blocks per step; for the
//!      others the bound comes from the largest tf the term has in the block and the block's
//!      shortest row, found while its postings for the window are decoded (once: step 3 reuses them);
//!   2. blocks whose summed bound can't reach the current k-th score are skipped;
//!   3. in a block that can, each row's own bound (the sum over the terms that row holds) is formed
//!      for all 64 rows at once, and only rows that can still qualify are scored exactly.
//!
//! ANDs walk the rarest term's blocks and look the others up there, skipping a block before any
//! lookup when a term with bound bytes is missing from it or the bounds can't reach the threshold.
//! Unranked, they AND whole windows of row masks instead when the rarest term is common. Counts and
//! unscored visits never look at frequencies. Phrases are ANDs whose rows are then checked against
//! positions; ranked, a row is scored first and its positions read only if it would enter the top k.
//!
//! Bounds are rounded up and thresholds down wherever they are quantized, and ties on the k-th
//! score are kept, so results equal exhaustive scoring exactly. Scores and sums of bounds are added
//! up in f64, so their rounding stays far inside those margins however many terms a query has.

use crate::seg::{Index, PosCursor, Postings, EDGES, SHIFT, UNKNOWN};
use crate::{Node, Shape, Stats, TopK};

const WINDOW: u32 = 4096; // blocks per window: 8 KB of accumulators
const SEED: usize = 32; // blocks scored first, best bound first, while the heap is filling

/// Where ranked scans spend their time, summed over calls (with the `prof` feature): nanoseconds
/// setting up, summing bounds, and scoring blocks; then blocks scored and rows scored.
#[cfg(feature = "prof")]
pub static WORK: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Experiment knobs (`prof` builds): blocks seeded per window, and whether windows after the heap
/// is full seed too.
#[cfg(feature = "prof")]
pub static KNOBS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

#[cfg(feature = "prof")]
fn work(i: usize, n: u64) {
    WORK[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// A plan's term as one segment holds it.
pub(crate) struct QTerm<'a> {
    pub(crate) p: Postings<'a>,
    pub(crate) idf: f32,
}

/// What to do with matches.
pub(crate) enum Sink<'s> {
    /// Keep the best; `prune` off scores every match (same answer, slowly, for checking).
    Top { top: &'s mut TopK, prune: bool },
    Count(&'s mut u64),
    Each(&'s mut dyn FnMut(u64)),
}

/// One segment's share of a query: `terms` are the plan's, the first `n_pos` of them score.
pub(crate) struct Seg<'a> {
    pub(crate) idx: &'a Index,
    pub(crate) del: &'a [u64], // rows to leave out, one bit each, or empty
    pub(crate) st: &'a Stats,  // statistics over every segment
    pub(crate) terms: &'a [Option<QTerm<'a>>],
    pub(crate) n_pos: usize,
}

/// A saturating u8 sum per row of a block: sixty-four lanes. Terms are added one at a time (the rows
/// holding the term, its weight), and `reaching` gives the rows whose sum is at least a value.
#[cfg(target_arch = "aarch64")]
struct Lanes([std::arch::aarch64::uint8x16_t; 4]);

#[cfg(target_arch = "aarch64")]
impl Lanes {
    const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // which byte of the mask each lane's row is in
    const SPREAD: [[u8; 16]; 4] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ];

    #[inline]
    fn new() -> Self {
        // SAFETY: NEON is part of aarch64's baseline, here and below.
        Lanes([unsafe { std::arch::aarch64::vdupq_n_u8(0) }; 4])
    }

    #[inline]
    fn add(&mut self, rows: u64, weight: u8) {
        use std::arch::aarch64::*;
        // SAFETY: the loads read the 16-byte constants above.
        unsafe {
            let (bit, mv, qv) = (vld1q_u8(Self::BIT.as_ptr()), vreinterpretq_u8_u64(vdupq_n_u64(rows)), vdupq_n_u8(weight));
            for (c, acc) in self.0.iter_mut().enumerate() {
                // each lane gets its row's byte of the mask, then its own bit of that byte
                let has = vtstq_u8(vqtbl1q_u8(mv, vld1q_u8(Self::SPREAD[c].as_ptr())), bit);
                *acc = vqaddq_u8(*acc, vandq_u8(has, qv));
            }
        }
    }

    /// Whether any row's sum is at least `need`.
    #[inline]
    fn any_reaching(&self, need: u8) -> bool {
        use std::arch::aarch64::*;
        // SAFETY: as above.
        unsafe { vmaxvq_u8(vmaxq_u8(vmaxq_u8(self.0[0], self.0[1]), vmaxq_u8(self.0[2], self.0[3]))) >= need }
    }

    #[inline]
    fn reaching(&self, need: u8) -> u64 {
        use std::arch::aarch64::*;
        // SAFETY: as above.
        unsafe {
            let (bit, needv, mut out) = (vld1q_u8(Self::BIT.as_ptr()), vdupq_n_u8(need), 0u64);
            for (c, a) in self.0.iter().enumerate() {
                let ge = vandq_u8(vcgeq_u8(*a, needv), bit);
                let bits = vaddv_u8(vget_low_u8(ge)) as u64 | (vaddv_u8(vget_high_u8(ge)) as u64) << 8;
                out |= bits << (16 * c);
            }
            out
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
struct Lanes([u8; 64]);

#[cfg(not(target_arch = "aarch64"))]
impl Lanes {
    fn new() -> Self {
        Lanes([0; 64])
    }

    fn add(&mut self, mut rows: u64, weight: u8) {
        while rows != 0 {
            let r = rows.trailing_zeros() as usize;
            self.0[r] = self.0[r].saturating_add(weight);
            rows &= rows - 1;
        }
    }

    fn any_reaching(&self, need: u8) -> bool {
        self.0.iter().any(|&a| a >= need)
    }

    fn reaching(&self, need: u8) -> u64 {
        (0..64).filter(|&r| self.0[r] >= need).fold(0, |o, r| o | 1 << r)
    }
}

/// Adds `ceil(byte * mul / 256)` to `acc[b - from]` for blocks `from..to`, whose bound bytes are in
/// `bounds`; `mul` is at most 256.
#[cfg_attr(feature = "prof", inline(never))]
#[cfg_attr(not(feature = "prof"), inline)]
fn add_bounds(acc: &mut [u16], bounds: &[u64], from: u32, to: u32, mul: u16) {
    let byte = |b: u32| (bounds[b as usize >> 3] >> (8 * (b & 7))) as u8 as u32;
    let scalar = |acc: &mut [u16], lo: u32, hi: u32| {
        for b in lo..hi {
            let a = &mut acc[(b - from) as usize];
            *a = a.saturating_add(((byte(b) * mul as u32 + 255) >> 8) as u16);
        }
    };
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let (first, last) = (from.next_multiple_of(8).min(to), to & !7);
        if first >= last {
            return scalar(acc, from, to);
        }
        scalar(acc, from, first);
        // SAFETY: NEON is baseline on aarch64. Each step reads one whole word of `bounds` (bounds
        // checked) and 8 accumulators, whose range the slice below checks.
        unsafe {
            let round = vdupq_n_u16(255);
            for b in (first..last).step_by(8) {
                let w = vmulq_n_u16(vmovl_u8(vcreate_u8(bounds[(b / 8) as usize])), mul); // at most 255 * 256
                let a = acc[(b - from) as usize..(b - from) as usize + 8].as_mut_ptr();
                vst1q_u16(a, vqaddq_u16(vld1q_u16(a), vshrq_n_u16(vaddq_u16(w, round), 8)));
            }
        }
        scalar(acc, last, to);
    }
    #[cfg(not(target_arch = "aarch64"))]
    scalar(acc, from, to);
}

/// The first index at or after `from` whose accumulator is at least `min` (which is at least 1).
#[inline]
fn next_reaching(acc: &[u16], from: usize, min: u16) -> Option<usize> {
    let mut i = from;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; each step reads 8 accumulators inside the slice.
    unsafe {
        use std::arch::aarch64::*;
        let minv = vdupq_n_u16(min);
        while i + 8 <= acc.len() {
            // one byte per accumulator: 255 where it reaches
            let hit = vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(vcgeq_u16(vld1q_u16(acc[i..i + 8].as_ptr()), minv))), 0);
            if hit != 0 {
                return Some(i + hit.trailing_zeros() as usize / 8);
            }
            i += 8;
        }
    }
    acc[i.min(acc.len())..].iter().position(|&a| a >= min).map(|p| i + p)
}

/// Runs the query over one segment into `sink`. The segment must be able to match: a scoring term
/// present, and for `Shape::All` every term present.
pub(crate) fn run(seg: &Seg, shape: &Shape, sink: &mut Sink) {
    let mut s = Scan::new(seg, shape);
    let present: Vec<usize> = s.scoring().map(|(j, _)| j).collect();
    match (shape, sink) {
        (Shape::All(phrases), sink) => s.all(&present, phrases, sink),
        (_, Sink::Top { top, prune }) => s.ranked(top, *prune),
        (_, sink) => s.matches(sink),
    }
}

/// Scratch for checking a phrase in a row.
#[derive(Default)]
struct Adjacency {
    /// Per plan term: its positions in the row last read, where they were read from, and that row.
    pos: Vec<(Vec<u32>, PosCursor, u64)>,
    order: Vec<(u32, usize, usize)>, // the phrase's words as (tf in the row, place in the phrase, posting)
    starts: Vec<u32>,                // positions where the phrase can still start
}

/// A term's (block, rows there, first posting there, largest tf there).
type Listed = (u32, u64, usize, u32);

struct Scan<'a, 's> {
    seg: &'a Seg<'a>,
    shape: &'s Shape,
    norm: [f32; 256],                 // by length class: k1 (1 - b + b len / avgdl) at the class's lower edge
    unit: f32,                        // score per bound-byte unit (before idf), nudged up; grows if the average length has
    cur: Vec<Option<(u64, usize)>>,   // each plan term's rows in the block at hand, and its first posting there
    masks: Vec<u64>,                  // the same rows, for evaluating a tree
    later: Vec<(usize, u8)>,          // scratch: the terms of the block at hand and their weights
    // Ranked scans decode a term that keeps no bounds once per window, while summing bounds: its
    // (block, rows, first posting, largest tf) there, read back in block order when blocks are scored.
    lists: Vec<Option<Vec<Listed>>>,
    read: Vec<usize>, // how far into each list
    tf: Vec<u32>,     // each term's largest tf in the block at hand, when its list gave it
    positions: std::cell::RefCell<Adjacency>, // scratch for `adjacent`
}

impl<'a, 's> Scan<'a, 's> {
    #[cfg_attr(feature = "prof", inline(never))]
    fn new(seg: &'a Seg<'a>, shape: &'s Shape) -> Self {
        let st = seg.st;
        let norm = std::array::from_fn(|c| st.c0 + st.c1 * EDGES[c] as f32);
        // Bound bytes were computed under the segment's own average length. A larger one now raises
        // every tf_part by at most the ratio, so the bytes are scaled by it.
        let unit = (st.params.k1 + 1.0) / 255.0 * (st.avgdl / seg.idx.stats.avgdl).max(1.0) * (1.0 + 1e-6);
        let n = seg.terms.len();
        let lists = (0..n).map(|_| None).collect();
        Scan { seg, shape, norm, unit: unit as f32, cur: vec![None; n], masks: vec![0; n], later: Vec::with_capacity(n), lists, read: vec![0; n], tf: vec![0; n], positions: Adjacency { pos: vec![Default::default(); n], ..Default::default() }.into() }
    }

    /// The scoring terms this segment has, with their place in the plan.
    fn scoring(&self) -> impl Iterator<Item = (usize, &'a QTerm<'a>)> + '_ {
        self.seg.terms[..self.seg.n_pos].iter().enumerate().filter_map(|(j, t)| t.as_ref().map(|t| (j, t)))
    }

    fn excluded(&self, b: u32) -> u64 {
        self.seg.del.get(b as usize).copied().unwrap_or(0)
    }

    /// Looks every plan term up in block `b`; returns the rows matching the query there.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    fn locate(&mut self, b: u32) -> u64 {
        let mut any = 0;
        for (j, t) in self.seg.terms.iter().enumerate() {
            self.cur[j] = match (&self.lists[j], t) {
                (Some(list), _) => {
                    let at = &mut self.read[j];
                    while list.get(*at).is_some_and(|e| e.0 < b) {
                        *at += 1;
                    }
                    list.get(*at).filter(|e| e.0 == b).map(|e| {
                        self.tf[j] = e.3;
                        (e.1, e.2)
                    })
                }
                (None, Some(t)) if t.p.bound(b) == Some(0) => None,
                (None, Some(t)) => t.p.block(b),
                (None, None) => None,
            };
            self.masks[j] = self.cur[j].map_or(0, |c| c.0);
            if j < self.seg.n_pos {
                any |= self.masks[j];
            }
        }
        let matched = match self.shape {
            Shape::Tree(node) => node.eval(&self.masks, &|phrase, rows| self.adjacent(b, rows, phrase)),
            _ => any,
        };
        matched & !self.excluded(b)
    }

    /// The rows among `rows` of block `b` (whose terms `cur` holds) where the terms of `phrase`
    /// follow one another. In each row the words are taken rarest first: the first gives the places
    /// the phrase could start, each further one strikes out those where it is missing, and a row is
    /// dropped as soon as none is left, usually before its frequent words' positions are read.
    fn adjacent(&self, b: u32, mut rows: u64, phrase: &[usize]) -> u64 {
        let (mut out, mut scratch) = (0, self.positions.borrow_mut());
        let Adjacency { pos, order, starts } = &mut *scratch;
        while rows != 0 {
            let r = rows.trailing_zeros();
            rows &= rows - 1;
            // each word's place in the phrase, with its posting and tf in this row
            order.clear();
            for (k, &j) in phrase.iter().enumerate() {
                let t = self.seg.terms[j].as_ref().expect("a phrase's terms are all in a matching row");
                let (m, first) = self.cur[j].expect("and in its block");
                let first = if first == UNKNOWN { t.p.first(b) } else { first };
                let i = first + (m & ((1u64 << r) - 1)).count_ones() as usize;
                order.push((t.p.tf(i), k, i));
            }
            order.sort_unstable();
            let stamp = (b << SHIFT | r) as u64 + 1; // marks positions read for this row
            for (n, &(tf, k, i)) in order.iter().enumerate() {
                let (list, cursor, read) = &mut pos[phrase[k]];
                if *read != stamp {
                    let t = self.seg.terms[phrase[k]].as_ref().expect("checked above");
                    t.p.positions(i, tf, cursor, list);
                    *read = stamp;
                }
                match n {
                    0 => {
                        starts.clear();
                        starts.extend(list.iter().filter(|&&p| p >= k as u32).map(|&p| p - k as u32));
                    }
                    _ => starts.retain(|&s| list.binary_search(&(s + k as u32)).is_ok()),
                }
                if starts.is_empty() {
                    break;
                }
            }
            if !starts.is_empty() {
                out |= 1 << r;
            }
        }
        out
    }

    /// Exact score of `row` (in block `b`, whose terms `cur` holds): its scoring terms in plan order,
    /// each in f32, summed in f64 and rounded once.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    fn score(&self, b: u32, r: u32) -> f32 {
        let (st, row) = (self.seg.st, (b << SHIFT | r) as usize);
        let norm = st.c0 + st.c1 * self.seg.idx.dl[row] as f32;
        let mut score = 0f64;
        for (j, t) in self.scoring() {
            if let Some((m, first)) = self.cur[j] {
                if m >> r & 1 == 1 {
                    let first = if first == UNKNOWN { t.p.first(b) } else { first };
                    let tf = t.p.tf(first + (m & ((1u64 << r) - 1)).count_ones() as usize) as f32;
                    score += (t.idf * tf * st.k1p1 / (tf + norm)) as f64;
                }
            }
        }
        score as f32
    }

    /// Scores `rows` of block `b` into `top`.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    fn push_rows(&self, b: u32, mut rows: u64, top: &mut TopK) {
        while rows != 0 {
            let r = rows.trailing_zeros();
            rows &= rows - 1;
            top.push(self.seg.idx.ids[(b << SHIFT | r) as usize], self.score(b, r));
        }
    }

    /// The most plan term `j` can score in block `b`, where `cur` holds it: from its bound byte if
    /// it keeps them, else from its largest tf there and the block's shortest row.
    #[inline]
    fn bound(&self, j: usize, t: &QTerm, b: u32) -> f32 {
        if let Some(byte) = t.p.bound(b) {
            return t.idf * byte as f32 * self.unit;
        }
        let (m, first) = self.cur[j].expect("only asked for terms in the block");
        let tf = if self.lists[j].is_some() { self.tf[j] } else { t.p.block_tf(b, m, first) } as f32;
        t.idf * self.seg.st.k1p1 * tf / (tf + self.norm[self.seg.idx.blen[b as usize] as usize])
    }

    /// Ranked OR or tree.
    fn ranked(&mut self, top: &mut TopK, prune: bool) {
        #[cfg(feature = "prof")]
        let mut clock = std::time::Instant::now();
        #[cfg(feature = "prof")]
        let mut lap = |i: usize| {
            work(i, clock.elapsed().as_nanos() as u64);
            clock = std::time::Instant::now();
        };
        let (idx, st) = (self.seg.idx, self.seg.st);
        let mut ts: Vec<(usize, &QTerm)> = self.scoring().collect();
        ts.sort_by(|a, b| b.1.idf.total_cmp(&a.1.idf)); // heaviest first: see `ranked_or_block`
        // Accumulators count units of a 255th of the largest bound any term can have here.
        let top_idf = ts.iter().map(|(_, t)| t.idf).fold(f32::MIN_POSITIVE, f32::max);
        let unit = top_idf * self.unit.max(st.k1p1 / 255.0);
        // A bound byte times idf / top_idf is its units; as a multiplier of 256ths, rounded up.
        let muls: Vec<u16> = ts.iter().map(|(_, t)| ((t.idf / top_idf * 256.0 * (1.0 + 1e-6)).ceil() as u16).min(256)).collect();
        // The fewest units a block needs: the threshold rounded down, or 1 while the heap is filling.
        let need = |top: &TopK| match prune && top.heap.len() == top.k {
            true => (top.threshold() / (1.0 + 1e-5) / unit).floor().clamp(1.0, 65535.0) as u16,
            false => 1,
        };
        let mut acc = [0u16; WINDOW as usize];
        let blocks = idx.blocks();
        #[cfg(feature = "prof")]
        lap(0);
        for b0 in (0..blocks).step_by(WINDOW as usize) {
            let b1 = (b0 + WINDOW).min(blocks);
            let acc = &mut acc[..(b1 - b0) as usize];
            acc.fill(0);
            for ((j, t), &mul) in ts.iter().zip(&muls) {
                match t.p.bounds {
                    Some(bounds) => add_bounds(acc, bounds, b0, b1, mul),
                    None => {
                        let list = self.lists[*j].get_or_insert_default();
                        list.clear();
                        let (scale, norm) = (t.idf * st.k1p1 / unit * (1.0 + 1e-6), &self.norm);
                        // units for tf = 1, which most postings have, by length class
                        let ones: [u16; 256] = std::array::from_fn(|c| (scale / (1.0 + norm[c])).ceil() as u16);
                        t.p.for_each_bounded(b0..b1, |b, m, first, tf| {
                            list.push((b, m, first, tf));
                            let c = idx.blen[b as usize] as usize;
                            let units = if tf == 1 { ones[c] } else { (scale * tf as f32 / (tf as f32 + norm[c])).ceil() as u16 };
                            let a = &mut acc[(b - b0) as usize];
                            *a = a.saturating_add(units);
                        });
                    }
                }
            }
            self.read.fill(0);
            #[cfg(feature = "prof")]
            lap(1);
            // While the heap is filling, score the window's best blocks first: the threshold they set
            // spares most of the rest.
            #[cfg(feature = "prof")]
            let (seed, always) = (KNOBS[0].load(std::sync::atomic::Ordering::Relaxed) as usize, KNOBS[1].load(std::sync::atomic::Ordering::Relaxed) == 1);
            #[cfg(feature = "prof")]
            let seed = if seed == 0 { SEED } else { seed };
            #[cfg(not(feature = "prof"))]
            let (seed, always) = (SEED, false);
            if prune && (always || top.heap.len() < top.k) {
                let min = need(top);
                let mut best: Vec<u32> = (b0..b1).filter(|&b| acc[(b - b0) as usize] >= min).collect();
                if best.len() > seed {
                    best.select_nth_unstable_by_key(seed, |&b| std::cmp::Reverse(acc[(b - b0) as usize]));
                    best.truncate(seed);
                }
                best.sort_unstable();
                for b in best {
                    self.ranked_block(b, top, prune, &ts);
                    acc[(b - b0) as usize] = 0;
                }
                self.read.fill(0);
            }
            let (mut theta, mut min, mut at) = (top.threshold(), need(top), 0);
            while let Some(i) = next_reaching(acc, at, min) {
                self.ranked_block(b0 + i as u32, top, prune, &ts);
                if top.threshold() != theta {
                    (theta, min) = (top.threshold(), need(top));
                }
                at = i + 1;
            }
            #[cfg(feature = "prof")]
            lap(2);
            #[cfg(feature = "prof")]
            if std::env::var("TRACE").is_ok() {
                eprintln!("  window {:>3}: lengths {:>5}..  live-eligible {:>4}  threshold {:.3}  scored so far {} blocks {} rows", b0 / WINDOW, self.seg.idx.dl[(b0 << SHIFT) as usize],
                    acc.iter().filter(|&&a| a >= need(top)).count(), top.threshold(), WORK[3].load(std::sync::atomic::Ordering::Relaxed), WORK[4].load(std::sync::atomic::Ordering::Relaxed));
            }
        }
    }

    /// Scores block `b`: only the rows whose own bound (the bounds of the terms they hold, summed)
    /// reaches the threshold, once the heap is full.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    fn ranked_block(&mut self, b: u32, top: &mut TopK, prune: bool, ts: &[(usize, &QTerm)]) {
        #[cfg(feature = "prof")]
        work(3, 1);
        let filter = prune && top.heap.len() == top.k;
        if filter && matches!(self.shape, Shape::Any) {
            return self.ranked_or_block(b, top, ts);
        }
        let mut rows = self.locate(b);
        if rows == 0 {
            return;
        }
        if filter {
            // Bounds in 200ths of the threshold, rounded up, so a row needs 200 of them.
            let per_unit = 200.0 * (1.0 + 2e-5) / top.threshold().max(f32::MIN_POSITIVE);
            let mut lanes = Lanes::new();
            for &(j, t) in ts {
                if let Some((m, _)) = self.cur[j] {
                    lanes.add(m, (self.bound(j, t, b) * per_unit).ceil().clamp(1.0, 255.0) as u8);
                }
            }
            rows &= lanes.reaching(200);
        }
        #[cfg(feature = "prof")]
        work(4, rows.count_ones() as u64);
        self.push_rows(b, rows, top);
    }

    /// `ranked_block` for a plain OR with the heap full. `ts` is in descending order of idf. Each
    /// term's bound in the block is known before its rows are (a bound byte, or its window list), so
    /// terms are added heaviest first and the block is dropped as soon as no row can qualify even if
    /// it held every term not yet added: usually after two or three of them.
    #[inline]
    fn ranked_or_block(&mut self, b: u32, top: &mut TopK, ts: &[(usize, &QTerm)]) {
        // Bounds in 200ths of the threshold, rounded up, so a row needs 200 of them.
        let per_unit = 200.0 * (1.0 + 2e-5) / top.threshold().max(f32::MIN_POSITIVE);
        let units = |bound: f32| (bound * per_unit).ceil().clamp(1.0, 255.0) as u8;
        self.later.clear();
        let mut rest = 0u32; // units of the terms not yet added
        for &(j, t) in ts {
            self.cur[j] = None;
            let u = if let Some(list) = &self.lists[j] {
                let at = &mut self.read[j];
                while list.get(*at).is_some_and(|e| e.0 < b) {
                    *at += 1;
                }
                let Some(e) = list.get(*at).filter(|e| e.0 == b) else { continue };
                (self.cur[j], self.tf[j]) = (Some((e.1, e.2)), e.3);
                units(self.bound(j, t, b))
            } else {
                match t.p.bound(b) {
                    Some(byte) if byte != 0 => units(t.idf * byte as f32 * self.unit),
                    _ => continue,
                }
            };
            rest += u as u32;
            self.later.push((j, u));
        }
        let mut lanes = Lanes::new();
        for i in 0..self.later.len() {
            let (j, u) = self.later[i];
            if self.cur[j].is_none() {
                let t = self.seg.terms[j].as_ref().expect("a scoring term of this segment");
                self.cur[j] = t.p.block(b);
            }
            lanes.add(self.cur[j].expect("its bound byte says the term is here").0, u);
            rest -= u as u32;
            if rest < 200 && !lanes.any_reaching((200 - rest) as u8) {
                return;
            }
        }
        let mut rows = lanes.reaching(200) & !self.excluded(b);
        if rows.count_ones() > 4 {
            rows = self.enough_tf(b, rows, top.threshold());
        }
        #[cfg(feature = "prof")]
        work(4, rows.count_ones() as u64);
        self.push_rows(b, rows, top);
    }

    /// Narrows `rows` of block `b` (whose terms are in `later` and `cur`) by term frequency, for
    /// blocks where frequent terms leave the row filter many rows. If the other terms' bounds here
    /// fall short of `theta` by x, a row needs this term to score x, so it needs a tf of at least
    /// some value (found from the block's shortest row, with no division per row): rows with less
    /// are dropped with a compare.
    fn enough_tf(&self, b: u32, mut rows: u64, theta: f32) -> u64 {
        let (st, norm) = (self.seg.st, self.norm[self.seg.idx.blen[b as usize] as usize]);
        let term = |j: usize| self.seg.terms[j].as_ref().expect("a scoring term of this segment");
        let total: f64 = self.later.iter().map(|&(j, _)| self.bound(j, term(j), b) as f64).sum();
        for &(j, _) in &self.later {
            let t = term(j);
            // what the term must score, as a share of the most any tf could: tf / (tf + norm) >= share
            let others = total - self.bound(j, t, b) as f64;
            let share = ((theta as f64 / (1.0 + 1e-5) - others * (1.0 + 1e-5)) / (t.idf * st.k1p1) as f64) as f32;
            let least = (share * norm / (1.0 - share) * (1.0 - 1e-4)).ceil();
            if share <= 0.0 || share >= 1.0 || least < 2.0 {
                continue; // no row is ruled out (a share of 1 or more is left to exact scoring)
            }
            let (m, first) = self.cur[j].expect("a term of this block");
            let first = if first == UNKNOWN { t.p.first(b) } else { first };
            let mut left = rows & m;
            rows &= m; // a row without the term can't make up for it
            while left != 0 {
                let r = left.trailing_zeros();
                left &= left - 1;
                if (t.p.tf(first + (m & ((1u64 << r) - 1)).count_ones() as usize) as f32) < least {
                    rows &= !(1 << r);
                }
            }
        }
        rows
    }

    /// Count or visit every match of an OR or tree.
    fn matches(&mut self, sink: &mut Sink) {
        let idx = self.seg.idx;
        let ts: Vec<&QTerm> = self.scoring().map(|(_, t)| t).collect();
        let mut any = [0u64; WINDOW as usize];
        let blocks = idx.blocks();
        for b0 in (0..blocks).step_by(WINDOW as usize) {
            let b1 = (b0 + WINDOW).min(blocks);
            let any = &mut any[..(b1 - b0) as usize];
            any.fill(0);
            for t in &ts {
                t.p.or_into(b0, any);
            }
            // A plain OR's count is a popcount of the window, less the rows left out.
            if let (Sink::Count(n), Shape::Any) = (&mut *sink, self.shape) {
                let del = self.seg.del.get(b0 as usize..b1 as usize);
                **n += match del {
                    Some(del) => any.iter().zip(del).map(|(a, d)| (a & !d).count_ones() as u64).sum::<u64>(),
                    None => any.iter().map(|a| a.count_ones() as u64).sum::<u64>(),
                };
                continue;
            }
            for b in b0..b1 {
                let mut rows = any[(b - b0) as usize];
                if rows == 0 {
                    continue;
                }
                rows = match self.shape {
                    Shape::Tree(_) => self.locate(b), // every match holds a scoring term, so it is in such a block
                    _ => rows & !self.excluded(b),
                };
                emit(idx, b, rows, sink);
            }
        }
    }

    /// The most term `t` can score in any row of this segment.
    fn top_bound(&self, t: &QTerm) -> f32 {
        t.idf * t.p.top as f32 * self.unit
    }

    /// Rows holding every term of `need` (the plan's scoring terms), those of each of `phrases`
    /// adjacent.
    ///
    /// The rarest term's blocks are walked and the others looked up there, after their bound bytes
    /// (where they keep them) say they are present and, ranked, that the block can reach the
    /// threshold. Unranked, if the rarest term is in many blocks, each window's row masks are ANDed
    /// whole instead, which costs about a nanosecond a block per term where a lookup costs ten.
    fn all(&mut self, need: &[usize], phrases: &[Vec<usize>], sink: &mut Sink) {
        let (idx, seg) = (self.seg.idx, self.seg);
        let term = |j: usize| seg.terms[j].as_ref().expect("a term of this segment");
        self.lists.iter_mut().for_each(|l| *l = None); // bounds come from the postings here, not from window lists
        let mut need = need.to_vec();
        need.sort_by_key(|&j| term(j).p.df); // rarest first
        let driver = need[0];
        let blocks = idx.blocks();

        // Ranked, the bounds spare most lookups whatever the driver; unranked, a common driver's
        // blocks are cheaper to AND a window at a time.
        if matches!(sink, Sink::Top { prune: true, .. }) || (term(driver).p.df as u64) * 8 < blocks as u64 {
            let lead = term(driver).p;
            return lead.for_each_block(0..blocks, |b, m, first| {
                let mut rows = m & !self.excluded(b);
                self.cur[driver] = Some((m, first));
                // Before any lookup: is every term with bound bytes here, and can a row qualify if
                // each other term scored its most (in this block, where its bytes say)?
                let mut others = 0f64;
                for &j in &need[1..] {
                    let t = term(j);
                    others += match t.p.bound(b) {
                        Some(0) => return,
                        Some(byte) => (t.idf * byte as f32 * self.unit) as f64,
                        None => self.top_bound(t) as f64,
                    };
                }
                if theta(sink).is_some_and(|theta| (self.bound(driver, term(driver), b) as f64 + others) * (1.0 + 1e-5) < theta as f64) {
                    return;
                }
                for &j in &need[1..] {
                    if rows == 0 {
                        return;
                    }
                    self.cur[j] = term(j).p.block(b);
                    rows &= self.cur[j].map_or(0, |c| c.0);
                }
                if rows != 0 {
                    self.all_block(b, rows, &need, phrases, sink);
                }
            });
        }

        let (mut rows, mut other) = ([0u64; WINDOW as usize], [0u64; WINDOW as usize]);
        for b0 in (0..blocks).step_by(WINDOW as usize) {
            let n = ((b0 + WINDOW).min(blocks) - b0) as usize;
            let (rows, other) = (&mut rows[..n], &mut other[..n]);
            rows.fill(0);
            term(driver).p.or_into(b0, rows);
            for &j in &need[1..] {
                if rows.iter().all(|&r| r == 0) {
                    break;
                }
                other.fill(0);
                term(j).p.or_into(b0, other);
                rows.iter_mut().zip(other.iter()).for_each(|(r, o)| *r &= o);
            }
            if let Some(del) = seg.del.get(b0 as usize..b0 as usize + n) {
                rows.iter_mut().zip(del).for_each(|(r, d)| *r &= !d);
            }
            if let (Sink::Count(count), true) = (&mut *sink, phrases.is_empty()) {
                **count += rows.iter().map(|r| r.count_ones() as u64).sum::<u64>();
                continue;
            }
            for (b, &rows) in (b0..).zip(rows.iter()).filter(|(_, &rows)| rows != 0) {
                need.iter().for_each(|&j| self.cur[j] = term(j).p.block(b));
                self.all_block(b, rows, &need, phrases, sink);
            }
        }
    }

    /// `all` for one block: `rows` hold every needed term, and `cur` has those terms there.
    #[inline]
    fn all_block(&mut self, b: u32, mut rows: u64, need: &[usize], phrases: &[Vec<usize>], sink: &mut Sink) {
        let (idx, seg) = (self.seg.idx, self.seg);
        let term = |j: usize| seg.terms[j].as_ref().expect("a term of this segment");
        let theta = theta(sink);
        let Sink::Top { top, .. } = &mut *sink else {
            let rows = phrases.iter().fold(rows, |rows, p| self.adjacent(b, rows, p));
            return emit(idx, b, rows, sink);
        };
        // Every match holds every term, so the block's bound is each term's, summed.
        if theta.is_some_and(|theta| need.iter().map(|&j| self.bound(j, term(j), b) as f64).sum::<f64>() * (1.0 + 1e-5) < theta as f64) {
            return;
        }
        if phrases.is_empty() {
            return self.push_rows(b, rows, top);
        }
        // Score first: positions are read only for rows that would enter the top k.
        while rows != 0 {
            let r = rows.trailing_zeros();
            rows &= rows - 1;
            let (id, score) = (idx.ids[(b << SHIFT | r) as usize], self.score(b, r));
            if top.takes(id, score) && phrases.iter().all(|p| self.adjacent(b, 1 << r, p) != 0) {
                top.push(id, score);
            }
        }
    }
}

/// The score a row must reach to matter to `sink`, if it prunes and has one yet.
#[inline]
fn theta(sink: &Sink) -> Option<f32> {
    match sink {
        Sink::Top { top, prune: true } if top.heap.len() == top.k => Some(top.threshold()),
        _ => None,
    }
}

/// Counts or visits `rows` of block `b`.
#[cfg_attr(feature = "prof", inline(never))]
#[cfg_attr(not(feature = "prof"), inline)]
fn emit(idx: &Index, b: u32, mut rows: u64, sink: &mut Sink) {
    match sink {
        Sink::Count(n) => **n += rows.count_ones() as u64,
        Sink::Each(f) => {
            while rows != 0 {
                f(idx.ids[(b << SHIFT | rows.trailing_zeros()) as usize]);
                rows &= rows - 1;
            }
        }
        Sink::Top { .. } => unreachable!("ranked sinks score rows instead"),
    }
}

impl Node {
    /// Rows of one block matching this node, from each term's row mask in that block; `adjacent`
    /// narrows the rows holding all of a phrase's terms to those where they follow one another.
    pub(crate) fn eval(&self, masks: &[u64], adjacent: &dyn Fn(&[usize], u64) -> u64) -> u64 {
        match self {
            Node::Term(i) => masks[*i],
            Node::Phrase(terms) => match terms.iter().map(|&i| masks[i]).reduce(|a, b| a & b) {
                Some(rows) if rows != 0 => adjacent(terms, rows),
                _ => 0, // some term is missing, or the phrase has none
            },
            Node::And(kids) if kids.is_empty() => 0,
            Node::And(kids) => kids.iter().fold(u64::MAX, |m, k| match k {
                Node::Not(k) => m & !k.eval(masks, adjacent),
                k => m & k.eval(masks, adjacent),
            }),
            Node::Or(kids) => kids.iter().fold(0, |m, k| m | k.eval(masks, adjacent)),
            Node::Not(_) => 0, // only valid under And, handled there (Query::check)
        }
    }
}
