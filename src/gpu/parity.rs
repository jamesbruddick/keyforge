//! Differential tests: every kernel primitive against the CPU function it mirrors.
//!
//! This is the highest-value layer of the GPU's verification, and it is worth being
//! explicit about why. The end-to-end test in this module's siblings can tell you that a
//! hash160 came out wrong. It cannot tell you that `fe_mul` drops a carry at limb 5. With
//! two implementations of the same arithmetic sitting next to each other, the cheap thing
//! is to run both over the same random inputs and diff them, and the cheap thing is also
//! the one that localises a fault to a single function.
//!
//! Each test dispatches a small "echo" kernel -- one thread per case, read inputs, apply
//! one operation, write outputs -- and compares against the CPU. The kernels live in
//! `kernels/parity.h` and are compiled only by these tests, so nothing here ships in a
//! scanning binary.
//!
//! Tests skip rather than fail where there is no device, so `cargo test --features metal`
//! on a headless box reports the machine's shape rather than a defect.

use anyhow::Result;

use super::source;
use super::{Arg, Backend, Layout};
use crate::scan::derive::Scope;

/// The parity kernels, appended to the ordinary translation unit.
const PARITY_H: &str = include_str!("../../kernels/parity.h");

/// How many random cases each test runs. Large enough that a carry bug in one limb of one
/// operation shows up, small enough that the whole suite stays a few seconds.
const CASES: usize = 4096;

/// A device with the parity kernels compiled, or `None` where there is no device.
///
/// Returning `None` rather than erroring is what makes the skip behaviour uniform: every
/// test opens one of these and returns early if there is nothing to talk to.
struct Harness {
    backend: Box<dyn Backend>,
    /// Held for the harness's whole life. These tests drive their own device directly
    /// rather than through `Gpu`, so they have to honour the same constraint `Gpu::run`
    /// does: two devices in one process return wrong results without reporting an error.
    /// See `gpu::ONE_LAUNCH_AT_A_TIME`.
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Harness {
    /// The harness the primitive tests use: the shared kernels, assembled against one
    /// arbitrary vulnerability because they do not touch `vuln_expand`.
    fn open() -> Option<Self> {
        Self::open_for(&crate::vuln::mt19937::MilkSad)
    }

    /// The same device, with `vuln_expand` from `v` rather than the default.
    ///
    /// Exactly one `vuln_expand` exists per translation unit, so testing a plugin's
    /// kernel means assembling the source around that plugin.
    fn open_for(v: &dyn crate::vuln::Vulnerability) -> Option<Self> {
        let serial = super::ONE_LAUNCH_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut backend = match super::open() {
            Ok(b) => b,
            Err(e) if super::is_unavailable(&e) => {
                println!("skipping parity tests: {e}");
                return None;
            }
            // A device that exists and then fails is a defect, not a reason to skip.
            Err(e) => panic!("opening a GPU: {e:#}"),
        };
        // The scope does not matter here -- these kernels test primitives, not the
        // pipeline -- but the source still has to carry a complete set of defines.
        let layout = Layout::new(&Scope::default(), 1);
        let source = format!(
            "{}\n// ---------------- kernels/parity.h ----------------\n{}",
            source::assemble(&layout, v, source::dialect()),
            PARITY_H
        );
        // Through `gpu::compile`, not `Backend::compile`: a machine with no kernel
        // compiler must skip these, not fail them with a panic from inside cudarc.
        match super::compile(&mut *backend, &source) {
            Ok(()) => {}
            Err(e) if super::is_unavailable(&e) => {
                println!("skipping parity tests: {e}");
                return None;
            }
            Err(e) => panic!("the parity kernels did not compile:\n{e:#}"),
        }
        Some(Self {
            backend,
            _serial: serial,
        })
    }

    /// Run `kernel` over `cases` threads with one input buffer and one output buffer,
    /// returning the output bytes.
    fn run(&mut self, kernel: &str, cases: usize, input: &[u8], out_len: usize) -> Result<Vec<u8>> {
        self.run_with(kernel, cases, cases, input, out_len, None)
    }

    /// The same, with the thread count and the element count given separately, and an
    /// optional third buffer.
    ///
    /// The two counts are usually equal -- one thread per case -- but not always: the
    /// batched inversion is a scan, so one thread walks every element. Conflating them is
    /// how the inversion test first "passed" for its single element and silently skipped
    /// the other 511.
    fn run_with(
        &mut self,
        kernel: &str,
        threads: usize,
        n: usize,
        input: &[u8],
        out_len: usize,
        extra: Option<&[u8]>,
    ) -> Result<Vec<u8>> {
        let inb = self.backend.buffer_from(input)?;
        let outb = self.backend.buffer(out_len)?;
        let n = (n as u32).to_le_bytes();
        let mut args = vec![Arg::Buffer(inb), Arg::Buffer(outb), Arg::Scalar(&n)];
        if let Some(data) = extra {
            args.push(Arg::Buffer(self.backend.buffer_from(data)?));
        }
        self.backend.dispatch(kernel, threads, &args)?;
        self.backend.sync()?;
        let mut out = vec![0u8; out_len];
        self.backend.read(outb, &mut out)?;
        Ok(out)
    }
}

/// A deterministic stream of test inputs.
///
/// Deterministic on purpose: a parity failure has to be reproducible from the test name
/// alone, and a seeded generator here is worth more than fresh randomness every run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }

    /// A field element as 32 big-endian bytes, already reduced below p.
    fn fe(&mut self) -> [u8; 32] {
        loop {
            let mut b = [0u8; 32];
            for c in b.chunks_mut(8) {
                c.copy_from_slice(&self.next_u64().to_be_bytes());
            }
            // Rejection rather than masking: a masked value is not uniform over the
            // field, and the interesting inputs for a carry bug are the ones near p.
            if crate::crypto::field::Fe::from_bytes(&b).is_some() {
                return b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::field::{self, Fe};
    use crate::vuln::mt19937::{Dist, Draw};

    /// Add, subtract, multiply, square and invert, over random field elements.
    ///
    /// The GPU uses eight 32-bit limbs where `src/field.rs` uses four 64-bit ones, so
    /// this is not a transcription being checked against itself -- it is two different
    /// carry chains being held to the same answers. Inputs come from a rejection sampler
    /// so they reach right up to p, which is where a reduction bug lives.
    #[test]
    fn field_arithmetic_matches_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0xF1E1D);

        let mut input = Vec::with_capacity(CASES * 64);
        let mut pairs = Vec::with_capacity(CASES);
        for _ in 0..CASES {
            let (a, b) = (rng.fe(), rng.fe());
            input.extend_from_slice(&a);
            input.extend_from_slice(&b);
            pairs.push((
                Fe::from_bytes(&a).expect("reduced"),
                Fe::from_bytes(&b).expect("reduced"),
            ));
        }

        // Five results per case, 32 bytes each: a+b, a-b, a*b, a^2, 1/a.
        let out = h
            .run("parity_field", CASES, &input, CASES * 5 * 32)
            .expect("dispatch");

        for (i, (a, b)) in pairs.iter().enumerate() {
            let got = |k: usize| -> [u8; 32] {
                out[(i * 5 + k) * 32..(i * 5 + k + 1) * 32]
                    .try_into()
                    .unwrap()
            };
            let want = [
                a.add(b),
                a.sub(b),
                a.mul(b),
                a.sqr(),
                a.inv(),
            ];
            for (k, w) in want.iter().enumerate() {
                assert_eq!(
                    got(k),
                    w.to_bytes(),
                    "case {i}, op {k}: a={:x?} b={:x?}",
                    a.to_bytes(),
                    b.to_bytes()
                );
            }
        }
    }

    /// The edges a random sampler will not reach in any realistic number of draws, and
    /// which are exactly where a conditional subtract is wrong: zero, one, p-1, and
    /// values straddling a limb boundary.
    #[test]
    fn field_arithmetic_matches_the_cpu_at_the_edges() {
        let Some(mut h) = Harness::open() else { return };

        let p_minus_1 = {
            let mut b = [0xffu8; 32];
            b[28..].copy_from_slice(&0xFFFF_FC2Eu32.to_be_bytes());
            b[24..28].copy_from_slice(&0xFFFF_FFFEu32.to_be_bytes());
            b
        };
        let mut one = [0u8; 32];
        one[31] = 1;
        let mut two_32 = [0u8; 32];
        two_32[27] = 1; // 2^32, straddling the CPU's first limb boundary
        let edges: Vec<[u8; 32]> = vec![[0u8; 32], one, two_32, p_minus_1];

        let mut input = Vec::new();
        let mut pairs = Vec::new();
        for a in &edges {
            for b in &edges {
                input.extend_from_slice(a);
                input.extend_from_slice(b);
                pairs.push((
                    Fe::from_bytes(a).expect("reduced"),
                    Fe::from_bytes(b).expect("reduced"),
                ));
            }
        }
        let cases = pairs.len();
        let out = h
            .run("parity_field", cases, &input, cases * 5 * 32)
            .expect("dispatch");

        for (i, (a, b)) in pairs.iter().enumerate() {
            // Inversion of zero is not defined and the two implementations are not
            // required to agree on it; every other operation is.
            let ops: &[(&str, Fe)] = &[
                ("add", a.add(b)),
                ("sub", a.sub(b)),
                ("mul", a.mul(b)),
                ("sqr", a.sqr()),
            ];
            for (k, (name, w)) in ops.iter().enumerate() {
                let got: [u8; 32] = out[(i * 5 + k) * 32..(i * 5 + k + 1) * 32]
                    .try_into()
                    .unwrap();
                assert_eq!(
                    got,
                    w.to_bytes(),
                    "case {i} ({name}): a={:x?} b={:x?}",
                    a.to_bytes(),
                    b.to_bytes()
                );
            }
        }
        let _ = field::ONE;
    }

    /// Build the 64-byte input records `parity_sha256` and `parity_hash160` share: one
    /// length byte then the message. Lengths cover every shape this program hashes -- a
    /// 22-byte script, a 33-byte compressed key, a 65-byte uncompressed one -- plus the
    /// one- and two-block boundary at 55/56, where the padding decision flips.
    fn short_messages(rng: &mut Rng) -> (Vec<u8>, Vec<Vec<u8>>) {
        let lengths: Vec<usize> = (0..CASES)
            .map(|i| match i % 11 {
                0 => 0,
                1 => 16,
                2 => 22,
                3 => 32,
                4 => 33,
                // 55/56 is where the one-block padding stops fitting, and 64/65 is the
                // uncompressed public key -- the longest thing this program hashes, and
                // the case a 63-byte ceiling silently skipped.
                5 => 55,
                6 => 56,
                7 => 63,
                8 => 64,
                9 => 65,
                _ => 119,
            })
            .collect();
        let mut input = Vec::with_capacity(CASES * 120);
        let mut msgs = Vec::with_capacity(CASES);
        for len in lengths {
            let msg = rng.bytes(len);
            input.push(len as u8);
            input.extend_from_slice(&msg);
            input.resize(input.len() + (119 - len), 0);
            msgs.push(msg);
        }
        (input, msgs)
    }

    #[test]
    fn sha256_matches_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0x5A256);
        let (input, msgs) = short_messages(&mut rng);
        let out = h
            .run("parity_sha256", CASES, &input, CASES * 32)
            .expect("dispatch");
        for (i, msg) in msgs.iter().enumerate() {
            let want: [u8; 32] = <sha2::Sha256 as sha2::Digest>::digest(msg).into();
            assert_eq!(out[i * 32..(i + 1) * 32], want, "case {i}, len {}", msg.len());
        }
    }

    /// hash160, which is SHA-256 into RIPEMD-160. This is the value the filter is keyed
    /// on, so it is the one that has to be right for a scan to find anything at all.
    #[test]
    fn hash160_matches_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0x160);
        let (input, msgs) = short_messages(&mut rng);
        let out = h
            .run("parity_hash160", CASES, &input, CASES * 20)
            .expect("dispatch");
        for (i, msg) in msgs.iter().enumerate() {
            let want = crate::crypto::hash::hash160(msg);
            assert_eq!(out[i * 20..(i + 1) * 20], want, "case {i}, len {}", msg.len());
        }
    }

    /// SHA-512 over lengths that straddle every padding decision it makes: under one
    /// block, the 111/112 boundary where the length field stops fitting, and multi-block
    /// messages the size of a 24-word mnemonic.
    #[test]
    fn sha512_matches_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0x512);
        let cases = 512;
        let mut input = Vec::with_capacity(cases * 256);
        let mut msgs = Vec::with_capacity(cases);
        for i in 0..cases {
            let len = match i % 8 {
                0 => 0,
                1 => 64,
                2 => 111,
                3 => 112,
                4 => 128,
                5 => 129,
                6 => 216,
                _ => 254,
            };
            let msg = rng.bytes(len);
            input.extend_from_slice(&(len as u16).to_le_bytes());
            input.extend_from_slice(&msg);
            input.resize(input.len() + (254 - len), 0);
            msgs.push(msg);
        }
        let out = h
            .run("parity_sha512", cases, &input, cases * 64)
            .expect("dispatch");
        for (i, msg) in msgs.iter().enumerate() {
            let want: [u8; 64] = <sha2::Sha512 as sha2::Digest>::digest(msg).into();
            assert_eq!(out[i * 64..(i + 1) * 64], want, "case {i}, len {}", msg.len());
        }
    }

    /// The MT19937 stream, which is the whole premise of the program: get this wrong and
    /// every derived address is for a wallet that never existed.
    ///
    /// Seed 1310 is in the list deliberately. It is the first seed whose stream contains a
    /// word in the rejected tail, so it is the one that separates the real
    /// `word / 16777215` distribution from the `word >> 24` a port is tempted to write.
    ///
    /// Every case is run under all three streams, because each is a different code path
    /// on the device as well as on the host -- and neither the libc++ nor the PHP one
    /// rejects anything, so a port that got the branch backwards would still produce
    /// plausible entropy. The PHP stream is the strictest of the three to get right: it
    /// is the only one whose *twist* differs, so an error there is in the state rather
    /// than in the mapping and shows up nowhere until the words diverge.
    #[test]
    fn mt19937_entropy_matches_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut seeds: Vec<(u32, u32)> = vec![
            (0, 0),
            (1, 0),
            (500, 0),
            (1310, 0),
            (u32::MAX, 0),
            (0, 1),
            (1310, 7),
            // The whole neighbourhood the end-to-end sweep walks, so a failure there is
            // never ambiguous about whether the entropy or the derivation is at fault.
            (1305, 0), (1306, 0), (1307, 0), (1308, 0), (1309, 0),
            (1311, 0), (1312, 0), (1313, 0), (1314, 0),
            (12345, 32),
            (999, 4096),
        ];
        let mut rng = Rng::new(0x1937);
        while seeds.len() < 256 {
            seeds.push((rng.next_u64() as u32, (rng.next_u64() % 64) as u32));
        }
        let cases: Vec<(u32, u32, Dist)> = [Dist::Libstdcxx, Dist::Libcxx, Dist::Php]
            .into_iter()
            .flat_map(|dist| {
                seeds
                    .iter()
                    .map(move |(seed, offset)| (*seed, *offset, dist))
            })
            .collect();

        let mut input = Vec::with_capacity(cases.len() * 12);
        for (seed, offset, dist) in &cases {
            input.extend_from_slice(&seed.to_le_bytes());
            input.extend_from_slice(&offset.to_le_bytes());
            input.extend_from_slice(&dist.code().to_le_bytes());
        }
        let out = h
            .run("parity_mt", cases.len(), &input, cases.len() * 32)
            .expect("dispatch");
        for (i, (seed, offset, dist)) in cases.iter().enumerate() {
            let draw = Draw::new(*dist, *offset as usize);
            let want = crate::vuln::mt19937::entropy_for_seed_at(*seed, draw);
            assert_eq!(
                out[i * 32..(i + 1) * 32],
                want,
                "seed {seed} at {} offset {offset}",
                dist.as_str()
            );
        }
    }

    /// `low-int`'s expansion against the host's, which is the whole of that plugin's
    /// device half.
    ///
    /// The one thing this kernel can get wrong is where in the 32-byte key the point
    /// lands, and a wrong answer there has no symptom on a device: the sweep would derive
    /// real, valid keys that are simply not the keys the point names, and report a clean
    /// range. So the cases are the byte and word boundaries the placement turns on, plus
    /// the top of the space and a point above 2^32 for the carry into `point_hi`.
    #[test]
    fn low_int_keys_match_the_cpu() {
        use crate::vuln::{Point, Vulnerability, weak_key::LowInteger};

        let Some(mut h) = Harness::open_for(&LowInteger) else { return };

        let mut points: Vec<u128> = vec![
            1, 2, 3, 255, 256, 257, 65_535, 65_536, 16_777_215, 16_777_216,
            // The end of the space this actually sweeps, and the two points either side
            // of the 32-bit edge -- where the host's carry hands the device a `point_hi`.
            0xffff_fffe, 0xffff_ffff, 0x1_0000_0000, 0x1_0000_0001,
        ];
        let mut rng = Rng::new(0x10ADD);
        while points.len() < 256 {
            points.push(rng.next_u64() as u128);
        }

        let mut input = Vec::with_capacity(points.len() * 12);
        for p in &points {
            input.extend_from_slice(&(*p as u32).to_le_bytes());
            input.extend_from_slice(&((*p >> 32) as u32).to_le_bytes());
            input.extend_from_slice(&0u32.to_le_bytes()); // the one stream
        }
        let out = h
            .run("parity_expand", points.len(), &input, points.len() * 32)
            .expect("dispatch");

        for (i, p) in points.iter().enumerate() {
            let mut want = Vec::new();
            LowInteger.expand(Point::Integer(*p), &mut want);
            assert_eq!(want.len(), 1, "low-int expands to one material per point");
            assert_eq!(out[i * 32..(i + 1) * 32], want[0], "point {p}");
        }
    }

    /// The mnemonic, byte for byte against `bip39::write_mnemonic`, at all three entropy
    /// sizes -- the checksum width changes with each, and so does the final partial word.
    #[test]
    fn bip39_phrases_match_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0xB1F39);
        let cases = 1024;
        let mut input = Vec::with_capacity(cases * 33);
        let mut entropies = Vec::with_capacity(cases);
        for i in 0..cases {
            let size = [16usize, 24, 32][i % 3];
            let mut e = rng.bytes(32);
            e.truncate(size);
            input.push(size as u8);
            input.extend_from_slice(&e);
            input.resize(input.len() + (32 - size), 0);
            entropies.push(e);
        }
        let out = h
            .run_with(
                "parity_bip39",
                cases,
                cases,
                &input,
                cases * 224,
                Some(&super::super::wordlist()),
            )
            .expect("dispatch");

        for (i, e) in entropies.iter().enumerate() {
            let len = u16::from_le_bytes([out[i * 224], out[i * 224 + 1]]) as usize;
            let got = std::str::from_utf8(&out[i * 224 + 2..i * 224 + 2 + len]).expect("utf8");
            assert_eq!(got, crate::wallet::bip39::mnemonic(e), "case {i}, {} bytes", e.len());
        }
    }

    /// PBKDF2-HMAC-SHA512 at the real 2048 iterations, against `pbkdf2::bip39_seed`.
    ///
    /// Deliberately not a shortened iteration count: the midstate reuse this shares with
    /// the CPU is exactly the part that could be wrong, and it only shows up across
    /// iterations. Few cases, because each is 4,098 compressions.
    #[test]
    fn bip39_seeds_match_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0xBDF2);
        let cases = 64;
        let mut input = Vec::with_capacity(cases * 224);
        let mut phrases = Vec::with_capacity(cases);
        for i in 0..cases {
            let size = [16usize, 24, 32][i % 3];
            let mut e = rng.bytes(32);
            e.truncate(size);
            let phrase = crate::wallet::bip39::mnemonic(&e);
            let bytes = phrase.as_bytes();
            input.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            input.extend_from_slice(bytes);
            input.resize(input.len() + (222 - bytes.len()), 0);
            phrases.push(phrase);
        }
        let out = h
            .run("parity_pbkdf2", cases, &input, cases * 64)
            .expect("dispatch");
        for (i, phrase) in phrases.iter().enumerate() {
            let want = crate::crypto::pbkdf2::bip39_seed(phrase);
            assert_eq!(out[i * 64..(i + 1) * 64], want, "case {i}: {phrase}");
        }
    }

    /// `k*G` for random scalars, against `ec::public_key`.
    ///
    /// This is the layer the GPU implements *differently* rather than transcribes: the CPU
    /// shares one field inversion across 256 lanes stepping the comb together, and the GPU
    /// accumulates each scalar in Jacobian coordinates on its own thread. Both encodings
    /// are compared, so a sign error in `y` shows up even when `x` is right.
    #[test]
    fn scalar_multiplication_matches_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0xEC);
        let cases = 2048;

        // Small scalars first: 1 and 2 exercise the top of the digit recoding where almost
        // every row is zero, which the random draws never reach.
        let mut scalars: Vec<[u8; 32]> = Vec::with_capacity(cases);
        for k in 1u8..=4 {
            let mut s = [0u8; 32];
            s[31] = k;
            scalars.push(s);
        }
        while scalars.len() < cases {
            scalars.push(rng.fe()); // below p, and so below n with overwhelming probability
        }

        let mut input = Vec::with_capacity(cases * 32);
        for s in &scalars {
            input.extend_from_slice(s);
        }
        let out = h
            .run_with(
                "parity_ec",
                cases,
                cases,
                &input,
                cases * 104,
                Some(&crate::crypto::ec::comb_for_gpu()),
            )
            .expect("dispatch");

        for (i, s) in scalars.iter().enumerate() {
            let want = crate::crypto::ec::public_key(s);
            assert_eq!(
                out[i * 104..i * 104 + 33],
                want.serialize(),
                "case {i}, compressed, scalar {s:x?}"
            );
            assert_eq!(
                out[i * 104 + 33..i * 104 + 98],
                want.serialize_uncompressed(),
                "case {i}, uncompressed, scalar {s:x?}"
            );
        }
    }

    /// Montgomery's batched inversion, against the per-element one.
    ///
    /// The pipeline converts a whole level of Jacobian points to affine behind one real
    /// inversion, so the scan's correctness rests on the prefix-product walk agreeing with
    /// `fe_inv` for every element -- including the first and last, where an off-by-one in
    /// the peel-back lands.
    #[test]
    fn batched_inversion_matches_element_wise_inversion() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0x117);
        let cases = 512;

        let mut input = Vec::with_capacity(cases * 32);
        let mut values = Vec::with_capacity(cases);
        for _ in 0..cases {
            let v = rng.fe();
            input.extend_from_slice(&v);
            values.push(Fe::from_bytes(&v).expect("reduced"));
        }
        let out = h
            .run_with("parity_batch_invert", 1, cases, &input, cases * 32, None)
            .expect("dispatch");

        for (i, v) in values.iter().enumerate() {
            assert_eq!(
                out[i * 32..(i + 1) * 32],
                v.inv().to_bytes(),
                "element {i} of {cases}"
            );
        }
    }

    /// What this device can actually do, in field multiplies and in whole scalar
    /// multiplications, with nothing else in the way.
    ///
    /// Not an assertion -- a measurement, printed. It is the number that says whether the
    /// pipeline is near the hardware's ceiling (in which case only a cheaper algorithm
    /// helps) or far below it (in which case the plumbing is at fault). Run it with
    /// `cargo test --features metal ceiling -- --nocapture`.
    ///
    /// **Take medians.** These swing by 3x between runs on a machine doing anything else;
    /// a single reading is worth nothing. Both kernels are genuine dependency chains, which
    /// is the only reason they measure anything at all -- two earlier benchmarks here
    /// discarded their own work and reported figures that looked authoritative and were
    /// fiction. If you add one, check that its result cannot be elided.
    /// Ignored by default: it asserts nothing, and it costs a device open -- about eight
    /// seconds on CUDA, where NVRTC compiles the whole kernel set up front. Run it when
    /// the question is "how close is the pipeline to what this card can do":
    /// `cargo test --features cuda ceiling -- --ignored --nocapture`.
    #[test]
    #[ignore = "measures rather than asserts; run with --ignored"]
    fn the_device_ceiling_for_curve_arithmetic() {
        let Some(mut h) = Harness::open() else { return };
        let threads = 1 << 16;

        let rounds = 4096u32;
        let t = std::time::Instant::now();
        h.run_with(
            "bench_fe_mul",
            threads,
            threads,
            &rounds.to_le_bytes(),
            threads * 32,
            None,
        )
        .expect("dispatch");
        let ops = threads as f64 * rounds as f64;
        println!(
            "  fe_mul:  {:.0} M/s",
            ops / t.elapsed().as_secs_f64() / 1e6
        );

        let comb = crate::crypto::ec::comb_for_gpu();
        let rounds = 512u32;
        let t = std::time::Instant::now();
        h.run_with(
            "bench_gej_add",
            threads,
            threads,
            &rounds.to_le_bytes(),
            threads * 32,
            Some(&comb),
        )
        .expect("dispatch");
        let rate = threads as f64 * rounds as f64 / t.elapsed().as_secs_f64();
        // A mixed addition is about ten field multiplies, which is the figure worth
        // comparing against `fe_mul` above.
        println!(
            "  gej_add: {:.1} M/s  (= {:.0} M field mul/s)",
            rate / 1e6,
            rate * 10.0 / 1e6
        );
    }

    /// Both bloom probe schedules, compared as indices rather than as verdicts.
    ///
    /// A filter would only tell us the two agreed on "absent", which is what they say
    /// about almost everything. Comparing the indices directly means a drift in the fold,
    /// the shift schedule or the block mixes surfaces as the wrong number, not as a silent
    /// miss.
    ///
    /// Both, because a sweep runs both: `keyscan bf-gen` writes a blocked primary and a
    /// scattered companion, and only the primary is bound to a device -- but which layout
    /// that is comes out of the file, so the device has to be right about either.
    #[test]
    fn the_bloom_probe_schedules_match_the_cpu() {
        let Some(mut h) = Harness::open() else { return };
        let mut rng = Rng::new(0xB100D);
        let cases = 1024;
        let mut input = Vec::with_capacity(cases * 20);
        let mut hashes = Vec::with_capacity(cases);
        for _ in 0..cases {
            let hash: [u8; 20] = rng.bytes(20).try_into().unwrap();
            input.extend_from_slice(&hash);
            hashes.push(hash);
        }
        // Twenty scattered indices then twenty blocked ones, per hash.
        let out = h
            .run("parity_bloom", cases, &input, cases * 40 * 8)
            .expect("dispatch");

        // The filter size `parity_bloom` derives its block indices for. Deliberately not a
        // power of two, so a block reduction that took a mask instead of the multiply-shift
        // would land somewhere else. Mirrors PARITY_BLOOM_BITS in kernels/parity.h.
        let bits = 1_000_003 * crate::target::bloom::BLOCK_BITS;

        for (i, hash) in hashes.iter().enumerate() {
            let want: Vec<u64> = crate::target::bloom::probe_indices(hash)
                .into_iter()
                .chain(crate::target::bloom::block_probe_indices(hash, bits))
                .collect();
            for (k, w) in want.iter().enumerate() {
                let at = (i * 40 + k) * 8;
                let got = u64::from_le_bytes(out[at..at + 8].try_into().unwrap());
                let schedule = if k < 20 { "scattered" } else { "blocked" };
                assert_eq!(got, *w, "case {i}, {schedule} probe {k}, hash {hash:x?}");
            }
        }
    }
}
