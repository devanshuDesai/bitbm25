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
//! ANDs walk the rarest term's blocks and look the others up there. Counts and unscored visits
//! combine row masks and never look at frequencies.
//!
//! Bounds are rounded up and thresholds down wherever they are quantized, and ties on the k-th
//! score are kept, so results equal exhaustive scoring exactly.

use crate::seg::{Index, Postings, EDGES, SHIFT, UNKNOWN};
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

/// Rows whose summed weight reaches `need`: `terms` are (rows holding the term, its weight), and
/// sums saturate at 255. Sixty-four u8 lanes, one per row.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(feature = "prof", inline(never))]
#[cfg_attr(not(feature = "prof"), inline)]
fn rows_reaching(terms: &[(u64, u8)], need: u8) -> u64 {
    use std::arch::aarch64::*;
    const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    const SPREAD: [[u8; 16]; 4] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ];
    // SAFETY: NEON is part of aarch64's baseline; the loads read the 16-byte constants above.
    unsafe {
        let bit = vld1q_u8(BIT.as_ptr());
        let spread = [vld1q_u8(SPREAD[0].as_ptr()), vld1q_u8(SPREAD[1].as_ptr()), vld1q_u8(SPREAD[2].as_ptr()), vld1q_u8(SPREAD[3].as_ptr())];
        let mut acc = [vdupq_n_u8(0); 4];
        for &(m, q) in terms {
            let (mv, qv) = (vreinterpretq_u8_u64(vdupq_n_u64(m)), vdupq_n_u8(q));
            for c in 0..4 {
                // each lane gets its row's byte of the mask, then its own bit of that byte
                let has = vtstq_u8(vqtbl1q_u8(mv, spread[c]), bit);
                acc[c] = vqaddq_u8(acc[c], vandq_u8(has, qv));
            }
        }
        let (needv, mut out) = (vdupq_n_u8(need), 0u64);
        for (c, a) in acc.iter().enumerate() {
            let ge = vandq_u8(vcgeq_u8(*a, needv), bit);
            let bits = vaddv_u8(vget_low_u8(ge)) as u64 | (vaddv_u8(vget_high_u8(ge)) as u64) << 8;
            out |= bits << (16 * c);
        }
        out
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn rows_reaching(terms: &[(u64, u8)], need: u8) -> u64 {
    let mut acc = [0u8; 64];
    for &(mut m, q) in terms {
        while m != 0 {
            let r = m.trailing_zeros() as usize;
            acc[r] = acc[r].saturating_add(q);
            m &= m - 1;
        }
    }
    (0..64).filter(|&r| acc[r] >= need).fold(0, |o, r| o | 1 << r)
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

/// Runs the query over one segment into `sink`. The segment must be able to match: a scoring term
/// present, and for `Shape::All` every term present.
pub(crate) fn run(seg: &Seg, shape: &Shape, sink: &mut Sink) {
    let mut s = Scan::new(seg, shape);
    match (shape, sink) {
        (Shape::All, sink) => s.all(sink),
        (_, Sink::Top { top, prune }) => s.ranked(top, *prune),
        (_, sink) => s.matches(sink),
    }
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
    mq: Vec<(u64, u8)>,               // scratch for `rows_reaching`
    // Ranked scans decode a term that keeps no bounds once per window, while summing bounds: its
    // (block, rows, first posting, largest tf) there, read back in block order when blocks are scored.
    lists: Vec<Option<Vec<Listed>>>,
    read: Vec<usize>, // how far into each list
    tf: Vec<u32>,     // each term's largest tf in the block at hand, when its list gave it
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
        Scan { seg, shape, norm, unit: unit as f32, cur: vec![None; n], masks: vec![0; n], mq: Vec::with_capacity(n), lists, read: vec![0; n], tf: vec![0; n] }
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
            Shape::Tree(node) => node.eval(&self.masks),
            _ => any,
        };
        matched & !self.excluded(b)
    }

    /// Exact score of `row` (in block `b`, whose terms `cur` holds): its scoring terms in plan order.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    fn score(&self, b: u32, r: u32) -> f32 {
        let (st, row) = (self.seg.st, (b << SHIFT | r) as usize);
        let norm = st.c0 + st.c1 * self.seg.idx.dl[row] as f32;
        let mut score = 0f32;
        for (j, t) in self.scoring() {
            if let Some((m, first)) = self.cur[j] {
                if m >> r & 1 == 1 {
                    let first = if first == UNKNOWN { t.p.first(b) } else { first };
                    let tf = t.p.tf(first + (m & ((1u64 << r) - 1)).count_ones() as usize) as f32;
                    score += t.idf * tf * st.k1p1 / (tf + norm);
                }
            }
        }
        score
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
        let ts: Vec<(usize, &QTerm)> = self.scoring().collect();
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
                        t.p.for_each_bounded(b0..b1, |b, m, first, tf| {
                            list.push((b, m, first, tf));
                            let units = (scale * tf as f32 / (tf as f32 + norm[idx.blen[b as usize] as usize])).ceil();
                            let a = &mut acc[(b - b0) as usize];
                            *a = a.saturating_add(units as u16);
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
            let (mut theta, mut min) = (top.threshold(), need(top));
            for b in b0..b1 {
                let a = acc[(b - b0) as usize];
                if a == 0 || a < min {
                    continue;
                }
                self.ranked_block(b, top, prune, &ts);
                if top.threshold() != theta {
                    (theta, min) = (top.threshold(), need(top));
                }
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
        let mut rows = self.locate(b);
        #[cfg(feature = "prof")]
        work(3, 1);
        if rows == 0 {
            return;
        }
        if prune && top.heap.len() == top.k {
            // Bounds in 200ths of the threshold, rounded up, so a row needs 200 of them.
            let per_unit = 200.0 * (1.0 + 2e-5) / top.threshold().max(f32::MIN_POSITIVE);
            self.mq.clear();
            for &(j, t) in ts {
                if let Some((m, _)) = self.cur[j] {
                    let units = (self.bound(j, t, b) * per_unit).ceil();
                    self.mq.push((m, units.clamp(1.0, 255.0) as u8));
                }
            }
            rows &= rows_reaching(&self.mq, 200);
        }
        #[cfg(feature = "prof")]
        work(4, rows.count_ones() as u64);
        self.push_rows(b, rows, top);
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

    /// AND of all the plan's terms: walk the rarest one's blocks and look the others up there.
    fn all(&mut self, sink: &mut Sink) {
        let idx = self.seg.idx;
        let ts: Vec<(usize, &QTerm)> = self.scoring().collect();
        let (driver, lead) = *ts.iter().min_by_key(|(_, t)| t.p.df).expect("a segment that can match has terms");
        let lead_p = lead.p;
        lead_p.for_each_block(0..idx.blocks(), |b, m, first| {
            let mut rows = m & !self.excluded(b);
            self.cur[driver] = Some((m, first));
            for &(j, t) in &ts {
                if j == driver {
                    continue;
                }
                if rows == 0 {
                    return;
                }
                self.cur[j] = t.p.block(b);
                rows &= self.cur[j].map_or(0, |c| c.0);
            }
            if rows == 0 {
                return;
            }
            let Sink::Top { top, prune } = &mut *sink else { return emit(idx, b, rows, sink) };
            if *prune && top.heap.len() == top.k {
                // Every match holds every term, so the block's bound is each term's, summed.
                let bound: f32 = ts.iter().map(|&(j, t)| self.bound(j, t, b)).sum();
                if bound * (1.0 + 1e-5) < top.threshold() {
                    return;
                }
            }
            self.push_rows(b, rows, top);
        });
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
    /// Rows of one block matching this node, from each term's row mask in that block.
    pub(crate) fn eval(&self, masks: &[u64]) -> u64 {
        match self {
            Node::Term(i) => masks[*i],
            Node::And(kids) if kids.is_empty() => 0,
            Node::And(kids) => kids.iter().fold(u64::MAX, |m, k| match k {
                Node::Not(k) => m & !k.eval(masks),
                k => m & k.eval(masks),
            }),
            Node::Or(kids) => kids.iter().fold(0, |m, k| m | k.eval(masks)),
            Node::Not(_) => 0, // only valid under And, handled there (Query::check)
        }
    }
}
