// Copyright 2024 The Erigon Authors
// This file is part of Erigon.
//
// Erigon is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Erigon is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Erigon. If not, see <http://www.gnu.org/licenses/>.

package seg

import (
	"math/rand"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// commitmentKVPath points to a real commitment .kv file, copied to a neutral
// directory (never a live datadir). Set via env to run the decompression
// benchmarks against production-shaped data.
func commitmentKVPath(b testing.TB) string {
	p := os.Getenv("COMMITMENT_KV")
	if p == "" {
		p = "/erigon-data/decomp_bench/commitment.kv"
	}
	if _, err := os.Stat(p); err != nil {
		b.Skipf("requires commitment .kv at %s (set COMMITMENT_KV)", p)
	}
	return p
}

// collectValueOffsets walks the whole file once and returns the offsets of the
// value words (odd positions: key, value, key, value, ...). Commitment reads
// land on a value after locating it via the accessor index, so the random-access
// benchmark below decodes only values.
func collectValueOffsets(b testing.TB, d *Decompressor) []uint64 {
	g := d.MakeGetter()
	offsets := make([]uint64, 0, 1<<20)
	idx := 0
	for g.HasNext() {
		off := g.dataP
		g.Skip()
		if idx%2 == 1 {
			offsets = append(offsets, off)
		}
		idx++
	}
	return offsets
}

// BenchmarkCommitmentDecompress measures decompression throughput of commitment
// values under the production read pattern: seek to a random value offset, then
// Next() to materialize it.
func BenchmarkCommitmentDecompress(b *testing.B) {
	path := commitmentKVPath(b)
	d, err := NewDecompressor(path)
	require.NoError(b, err)
	defer d.Close()

	d.MadvWillNeed()
	offsets := collectValueOffsets(b, d)
	require.NotEmpty(b, offsets)

	// Fixed pseudo-random permutation of value offsets (seeded for reproducibility).
	rng := rand.New(rand.NewSource(1))
	perm := rng.Perm(len(offsets))
	seq := make([]uint64, len(perm))
	for i, p := range perm {
		seq[i] = offsets[p]
	}

	g := d.MakeGetter()

	b.Run("random_next", func(b *testing.B) {
		b.ReportAllocs()
		var buf []byte
		var totalBytes int64
		i := 0
		for b.Loop() {
			g.Reset(seq[i])
			buf, _ = g.Next(buf[:0])
			totalBytes += int64(len(buf))
			i++
			if i == len(seq) {
				i = 0
			}
		}
		b.SetBytes(totalBytes / int64(b.N))
	})

	b.Run("sequential_next", func(b *testing.B) {
		b.ReportAllocs()
		var buf []byte
		var totalBytes int64
		g2 := d.MakeGetter()
		for b.Loop() {
			if !g2.HasNext() {
				g2.Reset(0)
			}
			buf, _ = g2.Next(buf[:0])
			totalBytes += int64(len(buf))
		}
		b.SetBytes(totalBytes / int64(b.N))
	})
}

// BenchmarkCommitmentUncompressed measures the uncompressed read paths that
// production actually uses for commitment/accounts/storage values (CompressKeys
// / CompressNone domains store values via the uncompressed word encoding).
func BenchmarkCommitmentUncompressed(b *testing.B) {
	path := commitmentKVPath(b)
	d, err := NewDecompressor(path)
	require.NoError(b, err)
	defer d.Close()
	d.MadvWillNeed()

	g := d.MakeGetter()
	offsets := collectValueOffsets(b, d)
	require.NotEmpty(b, offsets)
	words := make([][]byte, len(offsets))
	for i, off := range offsets {
		g.Reset(off)
		w, _ := g.NextUncompressed()
		words[i] = append([]byte(nil), w...)
	}

	b.Run("NextUncompressed", func(b *testing.B) {
		b.ReportAllocs()
		i := 0
		for b.Loop() {
			g.Reset(offsets[i])
			g.NextUncompressed()
			i++
			if i == len(offsets) {
				i = 0
			}
		}
	})

	b.Run("SkipUncompressed", func(b *testing.B) {
		b.ReportAllocs()
		i := 0
		for b.Loop() {
			g.Reset(offsets[i])
			g.SkipUncompressed()
			i++
			if i == len(offsets) {
				i = 0
			}
		}
	})

	b.Run("MatchCmpUncompressed_hit", func(b *testing.B) {
		b.ReportAllocs()
		i := 0
		for b.Loop() {
			g.Reset(offsets[i])
			g.MatchCmpUncompressed(words[i])
			i++
			if i == len(offsets) {
				i = 0
			}
		}
	})
}
