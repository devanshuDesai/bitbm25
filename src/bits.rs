//! Bit-packed data over little-endian `u64` words: fixed-width fields and Elias-Fano sequences.
//!
//! Readers index plain slices, so a corrupt index panics on a bounds check instead of reading
//! outside its buffer.

/// `width` bits (at most 32) starting at bit `pos`.
#[inline]
pub(crate) fn get(words: &[u64], pos: usize, width: u32) -> u32 {
    let (w, o) = (pos >> 6, (pos & 63) as u32);
    let mut v = words[w] >> o;
    if o + width > 64 {
        v |= words[w + 1] << (64 - o);
    }
    (v & ((1u64 << width) - 1)) as u32
}

/// Position of the `k`-th set bit of `w` (k = 0 is the lowest); `w` must have more than `k`.
#[inline]
pub(crate) fn select(mut w: u64, mut k: u32) -> u32 {
    let mut pos = 0;
    for shift in [32, 16, 8, 4, 2, 1] {
        let c = (w & ((1u64 << shift) - 1)).count_ones();
        if k >= c {
            k -= c;
            w >>= shift;
            pos += shift;
        }
    }
    pos
}

/// Appends fixed-width fields to a word buffer, starting at a word boundary.
pub(crate) struct Packer<'a> {
    out: &'a mut Vec<u64>,
    start: usize, // first word of this run
    len: usize,   // bits written
}

impl<'a> Packer<'a> {
    pub(crate) fn new(out: &'a mut Vec<u64>) -> Self {
        let start = out.len();
        Packer { out, start, len: 0 }
    }

    /// Appends the low `width` bits (at most 32) of `v`.
    #[inline]
    pub(crate) fn push(&mut self, v: u32, width: u32) {
        let (w, o) = (self.start + (self.len >> 6), (self.len & 63) as u32);
        if w >= self.out.len() {
            self.out.push(0);
        }
        self.out[w] |= (v as u64) << o;
        if o + width > 64 {
            self.out.push((v as u64) >> (64 - o));
        }
        self.len += width as usize;
    }
}

/// An ascending sequence of `n` values below `universe`, in Elias-Fano form: each value's low `l`
/// bits packed, and its high part as the position of a one in a bit vector (value `i` sets bit
/// `(value >> l) + i`), so the ones between two zeros are one *bucket* of `2^l` values. The starting
/// bit of every `2^s`-th bucket is sampled, which makes a bucket O(1) to find.
#[derive(Clone, Copy)]
pub(crate) struct Ef<'a> {
    low: &'a [u64],
    high: &'a [u64],
    samp: &'a [u32],
    buckets: usize,
    pub(crate) l: u32,
    s: u32,
}

/// 64 bits of `words` starting at bit `pos` (zeros past the end).
#[inline]
pub(crate) fn window(words: &[u64], pos: usize) -> u64 {
    let (w, o) = (pos >> 6, (pos & 63) as u32);
    let lo = words[w] >> o;
    match words.get(w + 1) {
        Some(&next) if o > 0 => lo | next << (64 - o),
        _ => lo,
    }
}

impl<'a> Ef<'a> {
    /// Words used by (low bits, high bits, samples) for `n` values below `universe`.
    pub(crate) fn words(n: usize, universe: usize, l: u32, s: u32) -> (usize, usize, usize) {
        let buckets = ((universe.max(1) - 1) >> l) + 1;
        ((n * l as usize).div_ceil(64), (n + buckets).div_ceil(64), buckets.div_ceil(1 << s).div_ceil(2))
    }

    /// Size in bits, for choosing a form.
    pub(crate) fn bits(n: usize, universe: usize, l: u32, s: u32) -> usize {
        let (a, b, c) = Self::words(n, universe, l, s);
        (a + b + c) * 64
    }

    /// Appends the sequence to `out`, sampling every `2^s`-th bucket.
    pub(crate) fn write(values: &[u32], universe: usize, l: u32, s: u32, out: &mut Vec<u64>) {
        let (lw, hw, sw) = Self::words(values.len(), universe, l, s);
        let low_at = out.len();
        if l > 0 {
            let mut p = Packer::new(out);
            for &v in values {
                p.push(v & ((1u64 << l) - 1) as u32, l);
            }
        }
        debug_assert_eq!(out.len(), low_at + lw);
        let high_at = out.len();
        out.resize(high_at + hw + sw, 0);
        let buckets = ((universe.max(1) - 1) >> l) + 1;
        let (mut bucket, mut samples) = (0usize, Vec::with_capacity(buckets.div_ceil(1 << s)));
        let start_of = |bucket: usize, i: usize| (bucket + i) as u32; // bucket b starts after b zeros and i ones
        for (i, &v) in values.iter().enumerate() {
            let h = (v >> l) as usize;
            while bucket <= h {
                if bucket % (1 << s) == 0 {
                    samples.push(start_of(bucket, i));
                }
                bucket += 1;
            }
            let pos = h + i;
            out[high_at + (pos >> 6)] |= 1 << (pos & 63);
        }
        while bucket < buckets {
            if bucket % (1 << s) == 0 {
                samples.push(start_of(bucket, values.len()));
            }
            bucket += 1;
        }
        for (j, s) in samples.iter().enumerate() {
            out[high_at + hw + j / 2] |= (*s as u64) << (32 * (j % 2));
        }
    }

    /// The sequence of `n` values below `universe` stored at the start of `words`.
    pub(crate) fn read(words: &'a [u64], n: usize, universe: usize, l: u32, s: u32) -> (Self, usize) {
        let (lw, hw, sw) = Self::words(n, universe, l, s);
        let (low, rest) = words.split_at(lw);
        let (high, rest) = rest.split_at(hw);
        (Ef { low, high, samp: as_u32(&rest[..sw]), buckets: ((universe.max(1) - 1) >> l) + 1, l, s }, lw + hw + sw)
    }

    /// Bit of the high vector where bucket `h` starts; the values before it number `start - h`.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn bucket_start(&self, h: usize) -> usize {
        let (pos, mut k) = (self.samp[h >> self.s] as usize, (h & ((1 << self.s) - 1)) as u32); // zeros still to pass
        if k == 0 {
            return pos;
        }
        let mut base = pos & !63;
        let mut zeros = !self.high[base >> 6] & (!0u64 << (pos & 63));
        loop {
            let c = zeros.count_ones();
            if k <= c {
                return base + select(zeros, k - 1) as usize + 1;
            }
            k -= c;
            base += 64;
            zeros = !self.high[base >> 6];
        }
    }

    #[inline]
    pub(crate) fn low(&self, i: usize) -> u32 {
        if self.l == 0 {
            0
        } else {
            get(self.low, i * self.l as usize, self.l)
        }
    }

    /// Calls `f(index, value)` for every value at or above `from`, in order, until `f` returns false.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn scan(&self, from: u32, mut f: impl FnMut(usize, u32) -> bool) {
        let h = (from >> self.l) as usize;
        if h >= self.buckets {
            return;
        }
        let mut pos = self.bucket_start(h);
        let mut i = pos - h;
        let mut w = self.high[pos >> 6] & (!0u64 << (pos & 63));
        loop {
            while w == 0 {
                pos = (pos | 63) + 1;
                if pos >> 6 >= self.high.len() {
                    return;
                }
                w = self.high[pos >> 6];
            }
            let p = (pos & !63) + w.trailing_zeros() as usize;
            w &= w - 1;
            let v = ((p - i) as u32) << self.l | self.low(i);
            if v >= from && !f(i, v) {
                return;
            }
            i += 1;
        }
    }

    /// 64 bits of the low parts, starting with value `i`'s.
    #[inline]
    pub(crate) fn lows(&self, i: usize) -> u64 {
        window(self.low, i * self.l as usize)
    }

    /// Calls `f(bucket, index of its first value, how many)` for every non-empty bucket from `from`
    /// on, in order, until `f` returns false.
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn runs(&self, from: usize, mut f: impl FnMut(usize, usize, usize) -> bool) {
        if from >= self.buckets {
            return;
        }
        let (mut h, mut pos) = (from, self.bucket_start(from));
        let mut i = pos - h;
        let bits = self.high.len() * 64;
        while h < self.buckets && pos < bits {
            let w = self.high[pos >> 6] >> (pos & 63);
            let room = 64 - (pos & 63); // bits of `w` that are the vector's
            // empty buckets: one per zero
            let zeros = if w == 0 { room } else { w.trailing_zeros() as usize };
            if zeros > 0 {
                h += zeros;
                pos += zeros;
                continue;
            }
            // a bucket's ones, which may run on into the next words
            let mut n = (w.trailing_ones() as usize).min(room);
            let mut end = pos + n;
            while n > 0 && end & 63 == 0 && end < bits {
                let more = self.high[end >> 6].trailing_ones() as usize;
                n += more;
                end += more;
                if more < 64 {
                    break;
                }
            }
            if !f(h, i, n) {
                return;
            }
            (h, i, pos) = (h + 1, i + n, end + 1);
        }
    }

    /// The values of bucket `h` as (index of the first, how many).
    #[cfg_attr(feature = "prof", inline(never))]
    #[cfg_attr(not(feature = "prof"), inline)]
    pub(crate) fn bucket(&self, h: usize) -> (usize, usize) {
        let pos = self.bucket_start(h);
        let mut ones = (self.high[pos >> 6] >> (pos & 63)).trailing_ones() as usize;
        if ones == 64 - (pos & 63) {
            // the run reaches the end of its word: keep counting in the next ones
            let mut w = (pos >> 6) + 1;
            while w < self.high.len() && self.high[w] == !0 {
                ones += 64;
                w += 1;
            }
            if w < self.high.len() {
                ones += self.high[w].trailing_ones() as usize;
            }
        }
        (pos - h, ones)
    }
}

/// A word slice viewed as bytes (lowest first).
pub(crate) fn as_u8(words: &[u64]) -> &[u8] {
    // SAFETY: every bit pattern is a valid u8 and bytes need no alignment; on the little-endian
    // targets this crate supports a word's lowest byte comes first.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 8) }
}

/// A word slice viewed as `u32`s (two per word, low half first).
pub(crate) fn as_u32(words: &[u64]) -> &[u32] {
    // SAFETY: u64 storage is aligned for u32 and every bit pattern is a valid u32; on the
    // little-endian targets this crate supports the low half of each word comes first.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), words.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    #[test]
    fn fields_round_trip_at_every_width() {
        let mut next = rng(1);
        for width in 1..=32u32 {
            let vals: Vec<u32> = (0..200).map(|_| (next() as u32) & (((1u64 << width) - 1) as u32)).collect();
            let mut out = vec![7u64]; // packing starts at a word boundary after existing data
            let mut p = Packer::new(&mut out);
            vals.iter().for_each(|&v| p.push(v, width));
            for (i, &v) in vals.iter().enumerate() {
                assert_eq!(get(&out[1..], i * width as usize, width), v, "width {width} field {i}");
            }
        }
    }

    #[test]
    fn select_finds_every_set_bit() {
        let mut next = rng(2);
        for _ in 0..200 {
            let w = next() & next();
            let mut bits = w;
            for k in 0..w.count_ones() {
                assert_eq!(select(w, k), bits.trailing_zeros());
                bits &= bits - 1;
            }
        }
    }

    #[test]
    fn elias_fano_matches_the_plain_list() {
        let mut next = rng(3);
        for (n, universe) in [(0usize, 1usize), (1, 1), (3, 1000), (500, 520), (500, 100_000), (4000, 5000), (300, 1 << 20)] {
            let mut vals: Vec<u32> = (0..universe as u32).collect();
            // a random subset of size n, ascending
            for i in 0..n {
                let j = i + (next() as usize) % (universe - i);
                vals.swap(i, j);
            }
            vals.truncate(n);
            vals.sort_unstable();
            for (l, s) in [(0u32, 6u32), (3, 3), (6, 3), (6, 6), (9, 6)] {
                let mut out = vec![1u64, 2, 3];
                Ef::write(&vals, universe, l, s, &mut out);
                let (ef, used) = Ef::read(&out[3..], n, universe, l, s);
                assert_eq!(used, out.len() - 3, "n {n} universe {universe} l {l}");
                for from in [0usize, 1, 7, 8, 9, 63, 64, 65, ((universe - 1) >> l) / 2, (universe - 1) >> l] {
                    let mut got = Vec::new();
                    ef.runs(from, |h, first, count| {
                        got.extend((first..first + count).map(|i| (h as u32) << l | ef.low(i)));
                        if l > 0 {
                            assert_eq!(ef.lows(first) & ((1u64 << l) - 1), ef.low(first) as u64);
                        }
                        true
                    });
                    let want: Vec<u32> = vals.iter().copied().filter(|&v| (v >> l) as usize >= from).collect();
                    assert_eq!(got, want, "runs from {from}: n {n} universe {universe} l {l} s {s}");
                }
                let mut got = Vec::new();
                ef.scan(0, |i, v| {
                    assert_eq!(i, got.len());
                    got.push(v);
                    true
                });
                assert_eq!(got, vals, "n {n} universe {universe} l {l}");
                for from in [0u32, 1, 63, 64, 65, (universe / 2) as u32, universe as u32 - 1] {
                    let mut got = Vec::new();
                    ef.scan(from, |_, v| {
                        got.push(v);
                        true
                    });
                    let want: Vec<u32> = vals.iter().copied().filter(|&v| v >= from).collect();
                    assert_eq!(got, want, "scan from {from}: n {n} universe {universe} l {l}");
                }
                for h in 0..=((universe - 1) >> l) {
                    let (first, count) = ef.bucket(h);
                    let want: Vec<u32> = vals.iter().copied().filter(|&v| (v >> l) as usize == h).collect();
                    assert_eq!(count, want.len(), "bucket {h}: n {n} universe {universe} l {l}");
                    for (j, w) in want.iter().enumerate() {
                        assert_eq!((h as u32) << l | ef.low(first + j), *w);
                    }
                }
            }
        }
    }
}
