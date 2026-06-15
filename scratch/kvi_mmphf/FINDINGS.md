# .kvi without the offset shuffle — learned MMPHF + Elias-Fano (prototype findings)

Issue: erigontech/erigon#21795. Goal: remove recsplit's permuted offset array
(Alex: *"ordered shuffled array of numbers ... ~5 bytes per number"*) so offsets,
which are already a sorted sequence, can be Elias-Fano compressed.

## Diagnosis (real file: mainnet `v2.0-commitment.0-8192.kvi`, N=702.8M, .kv=107.9 GB)

| component | bits/key | share |
|---|---|---|
| permuted offset array (`N×bytesPerRec`, bytesPerRec=5) | 40 | **78.7%** |
| recsplit MPHF + double-EF + existence filter | 10.8 | 21.3% |
| total | 50.8 | |

Domain `.kvi` uses `Enums:false` → raw offset stored at slot `h(key)`. Because `h`
is a near-random bijection, the array is a **random permutation** (entropy ≈ N·(log₂N−1.44)
≈ 28 bits/key) — fundamentally incompressible. Offsets are monotone in rank order
(AddKey panics otherwise), so the only fix is an **order-preserving hash** → then offsets
index directly as a monotone sequence → Elias-Fano.

## Design (LeMonHash-style, reusing recsplit as the retrieval structure)

1. PGM piecewise-linear model `key → approx rank`, bounded error ε.
2. recsplit stores only the **residual** `local = rank − (predict − ε)` instead of the 40-bit offset.
3. offsets stored once as Elias-Fano, indexed by true rank.

Lookup: `rank = predict(key) − ε + recsplit.Lookup(key)`; `offset = EF.Get(rank)`. 100% correct, verified on every key.

## Results (real mainnet files, step 9024-9056, eps swept)

bit-packed-ideal bits/key (lower better); baseline = equiv recsplit (~34 acct/storage, 50.8 commitment):

| domain | keys | EF offsets | best total (eps≈31) | vs baseline | residual floor |
|---|---|---|---|---|---|
| accounts (uniform 20B) | 2.59M | 7.37 | **16.70** | ~2.0x | tracks ε (→5–7) |
| storage (52B) | 9.26M | 7.44 | **24.51** | ~1.4x | ~14 (outliers) |
| commitment (≤39B) | 18.1M | 10.71 | **26.18** | **1.94x** | ~13 (outliers) |

Verified working prototype (recsplit byte-rounded, eps=255), commitment: 36.6 bits/key = **1.39x**, 100% correct, ~900 ns/op warm lookup.

## The hard part: deep-common-prefix groups

Commitment's biggest tie: **2,622,823 keys** are a contract's storage subtree, all sharing
a 33-byte prefix (`00159e48…eae12`), diverging only in bytes 33–38. A single float64
coordinate provably can't linearize this — tried 3 `fdelta` variants:
- all-limbs (monotonic): loses deep-byte precision → 116K residual outliers
- first-diff-limb scaled by B^k: same magnitude blow-up
- first-diff-limb unscaled: non-monotonic across the segment → 18M outlier

This is exactly why LeMonHash **recurses on long common prefixes** (fresh local coordinate per
level). A recursive model drops storage/commitment residual from ~13–14 to ~ε (5–7 bits),
pushing commitment toward ~20 bits/key (**~2.5x**).

## Conclusions

- **EF of offsets is the unconditional structural win**: 40 → 7.4–10.7 bits/key on the 79%
  of the file that is the permutation array. Requires only an order-preserving index.
- **Flat learned model already wins ~1.9x** on commitment and ~2x on accounts; cheap (≤0.5 bits/key)
  and fast.
- **Recursive (prefix-aware) model** needed to reach ~2.5–3x on prefix-clustered domains
  (commitment/storage); known LeMonHash technique, real implementation effort.
- Halving the resident index (50.8 → ~26) directly addresses #21795's "kvi can't fit in memory"
  (4.46 GB → ~2.3 GB for the 0-8192 file), avoiding the 440–480µs cold path.

## Cold/warm full-Get on real commitment files (15k random keys, vmtouch -e)

| file | keys | baseline .kvi | new | size× | cold base→new | warm base→new |
|---|---|---|---|---|---|---|
| 9024-9056   | 18M  | 115 MB  | 66 MB  | 1.74× | 180→177 µs | 91→99 µs |
| 187416-187418 | 34M | 213 MB | 121 MB | 1.76× | 208→205 µs | 83→93 µs |
| 8704-8960   | 73M  | 464 MB  | 264 MB | 1.76× | 289→273 µs | 96→104 µs |
| 0-102400 (v1) | 228M | 1454 MB | 791 MB | 1.84× | —/372 µs | — |

A/B vs enums=true (14.5 GB file): baseline 213 MB → enums=true 187 MB (1.14×) → new 121 MB
(1.76×). The learned model contributes 187→121 MB (1.55×) — most of the win. enums=true alone
barely helps because the permuted full-rank array is nearly as big as the raw-offset array.

Conclusion: MMPHF+EF is a memory-footprint win (1.74–1.84×, grows with scale) with comparable
cold/warm latency. Cold full-Get is dominated by the shared .kv value-page fault, so a smaller
index doesn't speed the per-lookup cold path — its value is residency (smaller → stays warm →
~90 µs path). Harness reproduces #21795's 14.5 GB row (213 MB / 203 µs). Bench tool: scratch/kvi_coldbench/.

## Recursive prefix-stripping model (solves the deep-prefix blowup)

scratch/kvi_mmphf_rec/. Each node uses an EXACT uint64 coordinate = the 8 key bytes after
that node's longest-common-prefix (LCP), so float precision lands on the first differing byte.
Keys that still tie on those 8 bytes recurse one level deeper. Critically, a recursed group
forces a SEGMENT BREAK in the parent so the parent never interpolates across the rank-gap the
group leaves behind (that gap was the −1.29M residual outlier in the flat model).

| file | keys | residual | recursion depth | bit-packed bits/key | vs baseline 50.8 |
|---|---|---|---|---|---|
| commitment 9024-9056 | 18M | 8 bits | 1 | 20.8 | 2.44× |
| commitment 8704-8960 | 73M | 9 bits | 1 | ~21.4 | 2.37× |
| accounts (uniform) | 2.6M | 7 bits | 0 (none) | 17.4 | — |

100% correct. Uniform keys need no recursion (depth 0) and are unaffected. Recursive model
improves commitment from the flat model's 1.94× to ~2.4× — meeting the projected ~2.5–3× range.
(recsplit byte-rounds 9-bit residuals to 2 bytes; production bit-packing gives the ideal column.)

## Next steps for integration
- Bit-packed residual store (not recsplit's byte-rounded bytesPerRec) — saves the rounding waste.
- Recursive model for deep-prefix buckets.
- New accessor format behind `DataStructureVersion`; keep the existence filter for absent keys.

Prototype: `scratch/kvi_mmphf/main.go` (run `-kv <file> -eps N [-skipbuild]`).
