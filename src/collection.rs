//! A collection: an index that takes adds and deletes. It is a directory of immutable segment files
//! (the `Index` format), a deleted-rows bitset per segment, and a MANIFEST naming the current ones.
//!
//! Adds go to an in-memory builder and deletes to a pending list; `commit` writes the builder out as
//! a new segment, applies the deletes, merges segments, and then replaces the MANIFEST, which is the
//! commit point: a crash before it leaves the previous commit, and unreferenced files are removed on
//! the next open or commit. A commit that fails keeps its adds and deletes for the next one. Queries
//! see the last commit and never wait for one.
//!
//! Each row has an id, assigned in increasing order and kept through merges. Segments hold disjoint,
//! increasing ranges of ids; a segment finds an id's row with `Index::row_of`.
//!
//! Deleted rows still count toward n, df and avgdl until a merge drops them (as in Lucene, Tantivy
//! and ParadeDB); a segment more than a fifth deleted is rewritten at the next commit.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use crate::seg::SHIFT;
use crate::{invalid, merge, replace_file, Analyzer, Builder, Index, Params, Query, SegView};

const MANIFEST_HEADER: &str = "bitbm25-collection 1";

/// One segment as of one commit.
#[derive(Clone)]
struct Seg {
    file: String, // "<n>.seg", in the collection's directory
    idx: Arc<Index>,
    del: Vec<u64>, // one bit per row, set if deleted; empty if none are
    deleted: usize,
    del_file: Option<String>, // where `del` is saved; None if not yet
}

impl Seg {
    fn live(&self) -> usize {
        self.idx.rows() - self.deleted
    }
}

struct Writer {
    buf: Builder,
    // Rows a commit took from `buf` and then failed to commit, oldest first: saved as segment files,
    // then (if saving failed) one still in memory. The next commit starts from them.
    staged: Vec<Seg>,
    unsaved: Option<Index>,
    pending: Vec<u64>, // ids to delete at the next commit
    next_id: u64,
    next_file: u64,
}

pub struct Collection {
    dir: PathBuf,
    _lock: File, // held for the collection's lifetime: one writer per directory
    snap: RwLock<Arc<Vec<Seg>>>,
    w: Mutex<Writer>,
}

impl Collection {
    /// Opens the collection in `dir`, creating it if needed. Fails if another `Collection` (in this
    /// process or another) has it open.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let ours = dir.join("LOCK").exists(); // opened as a collection before: its leftovers are ours to remove
        let lock = OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("LOCK"))?;
        lock.try_lock().map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "collection is open elsewhere"))?;
        let (mut next_id, mut next_file, mut segs) = (0, 0, Vec::new());
        match fs::read_to_string(dir.join("MANIFEST")) {
            Ok(text) => {
                let mut lines = text.lines();
                let num = |line: Option<&str>, key: &str| {
                    line.and_then(|l| l.strip_prefix(key)).and_then(|v| v.parse().ok()).ok_or_else(|| invalid("bad MANIFEST"))
                };
                if lines.next() != Some(MANIFEST_HEADER) {
                    return Err(invalid("not a bitbm25 collection, or a newer format"));
                }
                next_id = num(lines.next(), "next_id ")?;
                next_file = num(lines.next(), "next_file ")?;
                for line in lines {
                    let (file, del_file) = line.split_once(' ').ok_or_else(|| invalid("bad MANIFEST"))?;
                    let idx = Arc::new(Index::open(dir.join(own_file(file)?))?);
                    let (del, del_file) = match del_file {
                        "-" => (Vec::new(), None),
                        f => (read_del(&dir.join(own_file(f)?), idx.rows())?, Some(f.to_string())),
                    };
                    let deleted = del.iter().map(|w| w.count_ones() as usize).sum();
                    segs.push(Seg { file: file.to_string(), idx, del, deleted, del_file });
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        if ours {
            remove_unreferenced(&dir, &segs)?;
        }
        Ok(Collection {
            dir,
            _lock: lock,
            snap: RwLock::new(Arc::new(segs)),
            w: Mutex::new(Writer { buf: new_builder(), staged: Vec::new(), unsaved: None, pending: Vec::new(), next_id, next_file }),
        })
    }

    /// Adds a row (pre-tokenized: whitespace-separated terms) and returns its id. Visible to
    /// queries after the next `commit`. Ids of rows not yet committed are reused after a crash.
    pub fn add(&self, doc: &str) -> io::Result<u64> {
        let mut w = self.w.lock().unwrap();
        let id = w.next_id;
        w.buf.add(id, doc)?;
        w.next_id += 1;
        Ok(id)
    }

    /// Deletes the row with this id at the next `commit`; ids that don't exist are ignored, including
    /// ones not assigned yet (a later `add` gets them).
    pub fn delete(&self, id: u64) {
        let mut w = self.w.lock().unwrap();
        if id < w.next_id {
            w.pending.push(id);
        }
    }

    /// Makes adds and deletes since the last commit durable and visible, and merges segments.
    pub fn commit(&self) -> io::Result<()> {
        let mut guard = self.w.lock().unwrap();
        let w = &mut *guard;
        // Nothing below changes `w` but file numbers until the MANIFEST is written: a commit that
        // fails leaves its rows staged and its deletes pending, and the next one does it all again.
        loop {
            if w.unsaved.is_none() {
                if w.buf.rows() == 0 {
                    break;
                }
                w.unsaved = Some(std::mem::replace(&mut w.buf, new_builder()).finish());
            }
            let seg = self.write_segment(&mut w.next_file, w.unsaved.as_ref().expect("set above"))?;
            w.staged.push(seg);
            w.unsaved = None; // the mapped file serves the rows now, so the heap copy goes
        }
        let mut segs: Vec<Seg> = self.snapshot().as_ref().clone();
        segs.extend(w.staged.iter().cloned());
        for &id in &w.pending {
            let Some((s, row)) = segs.iter_mut().find_map(|s| s.idx.row_of(id).map(|row| (s, row))) else { continue };
            if s.del.is_empty() {
                s.del = vec![0; s.idx.rows().div_ceil(64)];
            }
            let bit = 1u64 << (row & 63);
            if s.del[row >> SHIFT] & bit == 0 {
                s.del[row >> SHIFT] |= bit;
                s.deleted += 1;
                s.del_file = None;
            }
        }

        // ponytail: merges run inside commit, blocking adds (not queries); a background merger if that hurts.
        segs.retain(|s| s.live() > 0);
        for i in 0..segs.len() {
            if segs[i].deleted * 5 > segs[i].idx.rows() {
                segs[i] = self.merged(&mut w.next_file, &segs[i..=i])?;
            }
        }
        // Merge the newest adjacent pair whose older segment is no larger than the newer, until
        // live sizes strictly decrease: at most log2(rows) segments, and each merge at least doubles
        // a row's segment, so a row is rewritten at most log2(rows) times.
        while let Some(i) = (1..segs.len()).rev().find(|&i| segs[i - 1].live() <= segs[i].live()) {
            let m = self.merged(&mut w.next_file, &segs[i - 1..=i])?;
            segs.splice(i - 1..=i, [m]);
        }

        for s in segs.iter_mut().filter(|s| s.deleted > 0 && s.del_file.is_none()) {
            let file = format!("{}.del", w.next_file);
            w.next_file += 1;
            replace_file(&self.dir.join(&file), |f| {
                let bytes = crate::as_bytes(&s.del);
                f.write_all(bytes)?;
                f.write_all(&crc32fast::hash(bytes).to_le_bytes())
            })?;
            s.del_file = Some(file);
        }
        let mut manifest = format!("{MANIFEST_HEADER}\nnext_id {}\nnext_file {}\n", w.next_id, w.next_file);
        for s in &segs {
            manifest += &format!("{} {}\n", s.file, s.del_file.as_deref().unwrap_or("-"));
        }
        replace_file(&self.dir.join("MANIFEST"), |f| f.write_all(manifest.as_bytes()))?;
        (w.staged, w.pending) = (Vec::new(), Vec::new()); // committed
        *self.snap.write().unwrap() = Arc::new(segs.clone());
        // Open snapshots keep their files mapped. Files left here are removed by the next commit or open.
        remove_unreferenced(&self.dir, &segs)
    }

    /// Saves `idx` as the next segment file and maps it.
    fn write_segment(&self, next_file: &mut u64, idx: &Index) -> io::Result<Seg> {
        let file = format!("{next_file}.seg");
        *next_file += 1;
        idx.save(self.dir.join(&file))?;
        let idx = Arc::new(Index::open(self.dir.join(&file))?);
        Ok(Seg { file, idx, del: Vec::new(), deleted: 0, del_file: None })
    }

    fn merged(&self, next_file: &mut u64, segs: &[Seg]) -> io::Result<Seg> {
        self.write_segment(next_file, &merge(&views(segs))?)
    }

    fn snapshot(&self) -> Arc<Vec<Seg>> {
        self.snap.read().unwrap().clone()
    }

    /// Exact BM25 top-k over committed rows as (id, score), best first. Ties go to the lower id.
    pub fn top_k(&self, q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>> {
        crate::top_k(&views(&self.snapshot()), q, k)
    }

    /// `top_k` without early stopping, for checking.
    pub fn top_k_exhaustive(&self, q: &Query, k: usize) -> io::Result<Vec<(u64, f32)>> {
        crate::top_k_exhaustive(&views(&self.snapshot()), q, k)
    }

    /// Exact number of matching committed rows.
    pub fn count(&self, q: &Query) -> io::Result<u64> {
        crate::count(&views(&self.snapshot()), q)
    }

    /// Committed rows that are not deleted.
    pub fn len(&self) -> u64 {
        self.snapshot().iter().map(|s| s.live() as u64).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks every segment file and deleted-rows file against its checksums.
    pub fn verify(&self) -> io::Result<()> {
        self.snapshot().iter().try_for_each(|s| s.idx.verify())
    }
}

/// A collection's segments: pre-split text, default k1/b.
fn new_builder() -> Builder {
    Builder::new(Params::default(), Analyzer::whitespace()).expect("default params are valid")
}

fn views(segs: &[Seg]) -> Vec<SegView<'_>> {
    segs.iter().map(|s| (&*s.idx, &s.del[..])).collect()
}

/// A file name from the MANIFEST, refused if it could point outside the directory.
fn own_file(name: &str) -> io::Result<&str> {
    let ok = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '.') && !name.starts_with('.');
    ok.then_some(name).ok_or_else(|| invalid(format!("bad file name in MANIFEST: {name:?}")))
}

fn read_del(path: &Path, rows: usize) -> io::Result<Vec<u64>> {
    let bytes = fs::read(path)?;
    let words = rows.div_ceil(64);
    if bytes.len() != words * 8 + 4 || crc32fast::hash(&bytes[..words * 8]).to_le_bytes() != bytes[words * 8..] {
        return Err(invalid(format!("{} is corrupt", path.display())));
    }
    Ok(bytes[..words * 8].as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect())
}

/// Removes segment, deleted-rows and temporary files that `segs` doesn't use: left by merges, or by
/// a commit that failed or crashed before writing its MANIFEST. Only names a collection gives its
/// files are touched: `<n>.seg`, `<n>.del`, those with `.tmp` added, and `MANIFEST.tmp`.
fn remove_unreferenced(dir: &Path, segs: &[Seg]) -> io::Result<()> {
    let keep: HashSet<&str> = segs.iter().flat_map(|s| [Some(s.file.as_str()), s.del_file.as_deref()]).flatten().collect();
    let named = |name: &str| {
        let name = name.strip_suffix(".tmp").unwrap_or(name);
        let num = name.strip_suffix(".seg").or_else(|| name.strip_suffix(".del"));
        name == "MANIFEST" || num.is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
    };
    for entry in fs::read_dir(dir)? {
        let name = entry?.file_name();
        let Some(name) = name.to_str() else { continue };
        if named(name) && name != "MANIFEST" && !keep.contains(name) {
            fs::remove_file(dir.join(name))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{brute_rows, check_query, corpus, TempPath};
    use crate::{Analyzer, Params};
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn matches_brute_force_through_adds_deletes_merges_and_reopens() {
        let dir = TempPath::new("collection");
        // Short and long rows in alternate batches give segments very different average lengths,
        // so their bounds need scaling to the collection's.
        let pools = [corpus(3_000, 0x5EED, 6), corpus(3_000, 0xFEED, 150)];
        let mut c = Collection::open(&dir.0).unwrap();
        assert!(Collection::open(&dir.0).is_err(), "a second writer is refused");
        // The model: committed live rows, and the text of every row ever committed (deleted rows
        // still count toward statistics until merged away).
        let mut live: BTreeMap<u64, String> = BTreeMap::new();
        let mut text: HashMap<u64, String> = HashMap::new();
        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut rand = move |n: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };
        let queries: &[&[&str]] = &[&["t0"], &["t1", "t3"], &["t2", "t150", "t299"], &["t5", "t40", "t80", "t200"], &["t0", "missing"]];
        let mut next = 0;
        let mut phrase_hits = 0;
        for round in 0..14 {
            // Batches of very different sizes, so the merge policy has work to do.
            let mut added = Vec::new();
            for _ in 0..[1, 40, 700, 5, 300, 0, 2][round % 7] {
                let doc = &pools[round % 2][next % 3_000];
                next += 1;
                added.push((c.add(doc).unwrap(), doc.clone()));
            }
            let assigned = live.keys().last().map_or(0, |&id| id + 1) + added.len() as u64;
            let deleted: Vec<u64> = (0..rand(150)).map(|_| rand(assigned + 20)).collect(); // some twice, uncommitted or unassigned
            deleted.iter().for_each(|&id| c.delete(id));
            if round == 6 {
                drop(c); // a crash: the adds and deletes are never committed
                c = Collection::open(&dir.0).unwrap();
            } else {
                c.commit().unwrap();
                text.extend(added.iter().cloned());
                live.extend(added);
                deleted.iter().for_each(|id| drop(live.remove(id)));
            }
            if round % 5 == 4 {
                drop(c);
                c = Collection::open(&dir.0).unwrap();
            }
            c.verify().unwrap();
            assert_eq!(c.len(), live.len() as u64);
            let segs = c.snapshot();
            let rows: Vec<(u64, &str, bool)> = segs
                .iter()
                .flat_map(|s| (0..s.idx.rows()).map(|r| s.idx.id(r)).map(|id| (id, text[&id].as_str(), live.contains_key(&id))))
                .collect();
            assert_eq!(rows.iter().filter(|r| r.2).count(), live.len(), "every live row is stored");
            assert!(segs.len() as f64 <= (live.len().max(1) as f64).log2() + 2.0, "{} segments", segs.len());
            for w in segs.windows(2) {
                let ids = |s: &Seg| (0..s.idx.rows()).map(|r| s.idx.id(r)).collect::<Vec<_>>();
                assert!(w[0].live() > w[1].live() && ids(&w[0]).iter().max() < ids(&w[1]).iter().min());
            }
            for q in queries {
                for and in [false, true] {
                    let q = Query::words(q, and);
                    check_query(&views(&segs), &q, &brute_rows(&rows, &q, Params::default(), &Analyzer::whitespace()));
                }
            }
            // positions survive merges: phrases of common words, which occur in a row
            for phrase in [&["t0", "t0"][..], &["t1", "t0"], &["t0", "t2", "t0"]] {
                let q = Query::Phrase(phrase.iter().map(|t| t.to_string()).collect());
                let want = brute_rows(&rows, &q, Params::default(), &Analyzer::whitespace());
                phrase_hits += want.len();
                check_query(&views(&segs), &q, &want);
            }
        }
        assert!(phrase_hits > 100, "the phrases match: {phrase_hits}");
        let files: HashSet<String> = fs::read_dir(&dir.0).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        let used = c.snapshot().iter().flat_map(|s| [Some(s.file.clone()), s.del_file.clone()]).flatten().count();
        assert_eq!(files.len(), used + 2, "only live files, MANIFEST and LOCK remain: {files:?}");
    }

    #[test]
    fn a_failed_commit_keeps_its_adds_and_deletes() {
        let dir = TempPath::new("late-fail");
        let c = Collection::open(&dir.0).unwrap();
        let keep = c.add("alpha beta").unwrap();
        c.commit().unwrap();
        c.add("gamma delta").unwrap();
        c.delete(keep);
        let count = |c: &Collection, w: &str| c.count(&Query::words(&[w], false)).unwrap();
        // The MANIFEST can't be replaced (its temp file can't be created over a directory), and a
        // segment can't be saved either (its temp name is taken the same way): two kinds of failure.
        for blocked in ["MANIFEST.tmp", "2.seg.tmp"] {
            fs::create_dir(dir.0.join(blocked)).unwrap();
            assert!(c.commit().is_err());
            fs::remove_dir(dir.0.join(blocked)).unwrap();
            assert_eq!((count(&c, "gamma"), count(&c, "alpha")), (0, 1), "a failed commit changes nothing visible");
            c.add("epsilon").unwrap(); // rows added after a failed commit come after its rows
        }
        c.commit().unwrap();
        let check = |c: &Collection| {
            assert_eq!((count(c, "gamma"), count(c, "alpha"), count(c, "epsilon")), (1, 0, 2), "the retry committed what failed");
            assert_eq!(c.len(), 3);
        };
        check(&c);
        drop(c);
        check(&Collection::open(&dir.0).unwrap());
    }

    #[test]
    fn deletes_touch_only_assigned_ids_and_open_only_its_own_files() {
        let dir = TempPath::new("foreign");
        fs::create_dir_all(&dir.0).unwrap();
        let theirs = ["wiki.seg", "download.tmp", "rows.del", "7.seg"];
        theirs.iter().for_each(|f| fs::write(dir.0.join(f), b"someone else's file").unwrap());
        let c = Collection::open(&dir.0).unwrap();
        c.delete(0); // nothing has id 0 yet: ignored, not left to delete the row that gets it
        c.add("alpha").unwrap();
        c.commit().unwrap();
        assert_eq!(c.count(&Query::words(&["alpha"], false)).unwrap(), 1);
        drop(c);
        drop(Collection::open(&dir.0).unwrap());
        assert!(theirs[..3].iter().all(|f| dir.0.join(f).exists()), "files a collection never names are left alone");
        assert!(!dir.0.join("7.seg").exists(), "a segment name in a directory opened as a collection is the collection's");
    }

    #[test]
    fn a_crash_before_the_manifest_leaves_the_last_commit() {
        let dir = TempPath::new("crash");
        let c = Collection::open(&dir.0).unwrap();
        c.add("a b").unwrap();
        c.commit().unwrap();
        c.add("a c").unwrap();
        c.commit().unwrap();
        drop(c);
        // Simulate a crash mid-commit: a later segment and a temp file exist, the MANIFEST doesn't name them.
        fs::write(dir.0.join("99.seg"), b"partial").unwrap();
        fs::write(dir.0.join("MANIFEST.tmp"), b"partial").unwrap();
        let c = Collection::open(&dir.0).unwrap();
        assert_eq!(c.count(&Query::words(&["a"], false)).unwrap(), 2);
        assert!(!dir.0.join("99.seg").exists() && !dir.0.join("MANIFEST.tmp").exists());
    }
}
