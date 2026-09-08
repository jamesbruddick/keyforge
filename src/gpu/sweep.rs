//! The whole pipeline on a device, held against the same pipeline on the CPU.
//!
//! `parity` checks the primitives one at a time -- field arithmetic, the hashes, the comb,
//! the inversion. This checks the thing built out of them: given a range of points and a
//! filter, does the device pick out the same points the CPU does?
//!
//! That is the claim the whole GPU design rests on. The device is a **filter** and the CPU
//! is the **oracle**: a record says "look at this point" and nothing more, and the host
//! re-derives that point through `derive::Deriver` before anything reaches `matches.txt`.
//! So a device that is wrong cannot write a wrong secret -- but it can silently *miss*
//! wallets, and that failure has no symptom at all. A sweep would finish, report a clean
//! pass, and have skipped whatever the kernels got wrong.
//!
//! These tests are what make that failure loud. They skip with a message where there is no
//! device, so a headless box does not report a machine's shape as a defect -- watch for
//! the message if you expected them to run.

use crate::gpu::{Batch, Gpu, is_unavailable};
use crate::scan::derive::{All, Deriver, Location, Route, Scope};
use crate::target::Target;
use crate::target::bloom::{BloomFilter, testing};
use crate::vuln::{Point, Vulnerability, mt19937::MilkSad};
use crate::wallet::address::HashForm;
use crate::wallet::path::PathSpec;
use std::collections::BTreeSet;

/// Every hash160 the CPU derives for a range of points, with the point that produced it.
fn cpu_hashes(
    v: &dyn Vulnerability,
    scope: &Scope,
    stream: usize,
    range: std::ops::Range<u128>,
) -> Vec<(u128, [u8; 20])> {
    let mut out = Vec::new();
    let mut deriver = Deriver::new();
    let mut expanded = Vec::new();
    for point in range {
        expanded.clear();
        v.expand(Point::Integer(point), &mut expanded);
        let Some(material) = expanded.get(stream).copied() else {
            continue;
        };
        deriver.walk_batch(
            &[material],
            scope,
            &mut All(|_: &Location, hash: &[u8; 20], _: &str| out.push((point, *hash))),
        );
    }
    out
}

/// A filter holding a chosen set of hashes, written to a scratch file.
fn filter_of(name: &str, hashes: &[[u8; 20]]) -> BloomFilter {
    let mut bits = vec![0u64; 1 << 14];
    for h in hashes {
        testing::reference_add(&mut bits, h);
    }
    let path = testing::scratch(name);
    testing::write_filter(&path, &bits);
    BloomFilter::open(&path).expect("the filter opens")
}

/// The scopes worth sweeping: the default, and shapes the old fixed pipeline could not
/// express at all.
fn scopes() -> Vec<(&'static str, Scope)> {
    let path = |p: &str| PathSpec::parse(p).unwrap();
    vec![
        ("default", Scope::default()),
        (
            "one narrow path",
            Scope {
                material_sizes: vec![32],
                routes: vec![Route::Bip39],
                paths: vec![path("m/44'/0'/0'/0/{0..3}")],
                forms: vec![HashForm::Compressed],
            },
        ),
        (
            // Hardened below normal, which the old three-hardened-steps kernel could not
            // do, plus `m` itself, which has no segments to walk at all.
            "interleaved and bare",
            Scope {
                material_sizes: vec![16, 32],
                routes: vec![Route::Bip39, Route::Bip32Seed],
                paths: vec![path("m"), path("m/0/1'/{0,1}"), path("m/{0,1}")],
                forms: vec![HashForm::Compressed, HashForm::P2shP2wpkh],
            },
        ),
        (
            "keys with no path",
            Scope {
                material_sizes: vec![32],
                routes: vec![Route::PrivKey],
                paths: vec![],
                forms: vec![HashForm::Compressed, HashForm::Uncompressed],
            },
        ),
    ]
}

/// The device must report every point the CPU says matches, and no others.
///
/// The filter is built from hashes the CPU actually derived, so there is something real to
/// find rather than a sweep of an empty filter that would pass by finding nothing. Points
/// are compared rather than records: that is the whole of what a record is for.
#[test]
fn the_device_finds_the_points_the_cpu_finds() {
    let points = 0u128..64;

    for (name, scope) in scopes() {
        assert_eq!(scope.validate(), Ok(()), "{name} is not a valid scope");

        let cpu = cpu_hashes(&MilkSad, &scope, 0, points.clone());
        assert!(!cpu.is_empty(), "{name} derived nothing");

        // Plant a handful of the CPU's own hashes, spread across the range so the match
        // is not all in one point or one region of the leaf array.
        let planted: Vec<[u8; 20]> = cpu
            .iter()
            .step_by(cpu.len() / 7 + 1)
            .map(|(_, h)| *h)
            .collect();
        let want: BTreeSet<u128> = cpu
            .iter()
            .filter(|(_, h)| planted.contains(h))
            .map(|(p, _)| *p)
            .collect();
        assert!(want.len() > 1, "{name} planted hashes from only one point");

        let filter = filter_of(&format!("sweep-{}.bf", name.replace(' ', "-")), &planted);
        let target = Target::new(filter, None);

        let mut gpu = match Gpu::open(&scope, &MilkSad, Batch::Fixed(points.end as usize)) {
            Ok(gpu) => gpu,
            Err(e) if is_unavailable(&e) => {
                println!("skipping: {e}");
                return;
            }
            Err(e) => panic!("{name}: {e}"),
        };
        gpu.bind_filter(target.primary()).expect("bind the filter");

        let hits = gpu
            .run(points.start, points.end as usize, 0)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let got: BTreeSet<u128> = hits.iter().map(|h| h.point as u128).collect();

        assert_eq!(got, want, "{name}: the device and the CPU disagree on which points match");

        // Every record's hash must be one the CPU derived for that point. A record the
        // host cannot confirm is counted in a real scan; here it is a failure.
        for hit in &hits {
            let point = hit.point as u128;
            assert!(
                cpu.iter().any(|(p, h)| *p == point && h == &hit.hash),
                "{name}: point {point} reported a hash the CPU never derived"
            );
        }
    }
}

/// A launch shorter than the batch the device was sized for must still be right.
///
/// Region boundaries in the leaf array move with the launch size, so a short launch is
/// where a slot decodes to the wrong point -- and the last launch of every range is short.
#[test]
fn a_short_launch_decodes_its_points_correctly() {
    let scope = Scope::default();
    let capacity = 32usize;
    let short = 5usize;

    let cpu = cpu_hashes(&MilkSad, &scope, 0, 0..short as u128);
    let planted: Vec<[u8; 20]> = cpu.iter().step_by(cpu.len() / 5 + 1).map(|(_, h)| *h).collect();
    let want: BTreeSet<u128> = cpu
        .iter()
        .filter(|(_, h)| planted.contains(h))
        .map(|(p, _)| *p)
        .collect();

    let filter = filter_of("sweep-short.bf", &planted);
    let target = Target::new(filter, None);

    let mut gpu = match Gpu::open(&scope, &MilkSad, Batch::Fixed(capacity)) {
        Ok(gpu) => gpu,
        Err(e) if is_unavailable(&e) => {
            println!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    gpu.bind_filter(target.primary()).expect("bind the filter");

    let hits = gpu.run(0, short, 0).expect("the short launch runs");
    let got: BTreeSet<u128> = hits.iter().map(|h| h.point as u128).collect();
    assert_eq!(got, want, "a short launch decoded its points wrongly");
}

/// Each byte stream is a different wallet, so a launch of stream 1 must find what the CPU
/// finds for stream 1 -- and not what it finds for stream 0.
#[test]
fn each_stream_is_swept_separately() {
    let scope = Scope {
        material_sizes: vec![32],
        routes: vec![Route::Bip39],
        paths: vec![PathSpec::parse("m/0").unwrap()],
        forms: vec![HashForm::Compressed],
    };
    let points = 0u128..32;

    let stream0 = cpu_hashes(&MilkSad, &scope, 0, points.clone());
    let stream1 = cpu_hashes(&MilkSad, &scope, 1, points.clone());
    assert_ne!(stream0, stream1, "the two streams derive the same wallets");

    // A filter holding only stream 1's hashes.
    let planted: Vec<[u8; 20]> = stream1.iter().map(|(_, h)| *h).collect();
    let filter = filter_of("sweep-stream.bf", &planted);
    let target = Target::new(filter, None);

    let mut gpu = match Gpu::open(&scope, &MilkSad, Batch::Fixed(points.end as usize)) {
        Ok(gpu) => gpu,
        Err(e) if is_unavailable(&e) => {
            println!("skipping: {e}");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    gpu.bind_filter(target.primary()).expect("bind the filter");

    let on_one = gpu.run(0, points.end as usize, 1).expect("stream 1 runs");
    assert_eq!(
        on_one.len(),
        stream1.len(),
        "stream 1 did not find every wallet the CPU derived for it"
    );
    for hit in &on_one {
        let point = hit.point as u128;
        assert!(
            stream1.iter().any(|(p, h)| *p == point && h == &hit.hash),
            "stream 1 reported a hash from another stream"
        );
    }
}
