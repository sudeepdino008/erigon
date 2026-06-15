// Recursive prefix-stripping learned model for the .kvi MMPHF.
//
// The flat model used one float coordinate over the whole 64-byte key, which
// loses precision on deep-common-prefix groups (a contract's storage subtree).
// Here each node uses an EXACT uint64 coordinate = the 8 key bytes after the
// node's common prefix. Keys that tie on those 8 bytes (share prefix+8) recurse
// one level deeper (prefix += 8), drilling to where keys diverge. The coordinate
// is exact and monotonic at every level, so the residual collapses to ~eps.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/erigontech/erigon/common/log/v3"
	"github.com/erigontech/erigon/db/recsplit"
	"github.com/erigontech/erigon/db/recsplit/eliasfano32"
	"github.com/erigontech/erigon/db/seg"
)

type arena struct {
	data []byte
	off  []uint64
}

func (a *arena) n() int           { return len(a.off) - 1 }
func (a *arena) key(i int) []byte { return a.data[a.off[i]:a.off[i+1]] }

func coordAt(k []byte, p int) uint64 {
	var c uint64
	for b := 0; b < 8; b++ {
		c <<= 8
		if p+b < len(k) {
			c |= uint64(k[p+b])
		}
	}
	return c
}

type pseg struct {
	c0    uint64
	y0    float64
	slope float64
}

type node struct {
	p        int
	segs     []pseg
	children map[uint64]*node
}

// buildSegments over (coord, rank) points; coords monotonic non-decreasing.
func buildSegments(cs []uint64, ranks []int, eps float64) []pseg {
	var segs []pseg
	m := len(cs)
	i := 0
	for i < m {
		c0 := cs[i]
		y0 := float64(ranks[i])
		slopeLo, slopeHi := math.Inf(-1), math.Inf(1)
		j := i + 1
		for ; j < m; j++ {
			dx := float64(cs[j] - c0)
			yj := float64(ranks[j])
			if dx == 0 {
				if math.Abs(yj-y0) > eps {
					break
				}
				continue
			}
			lo := (yj - eps - y0) / dx
			hi := (yj + eps - y0) / dx
			if lo > slopeLo {
				slopeLo = lo
			}
			if hi < slopeHi {
				slopeHi = hi
			}
			if slopeLo > slopeHi {
				break
			}
		}
		slope := 0.0
		if !math.IsInf(slopeLo, 0) && !math.IsInf(slopeHi, 0) {
			slope = (slopeLo + slopeHi) / 2
		} else if !math.IsInf(slopeLo, 0) {
			slope = slopeLo
		} else if !math.IsInf(slopeHi, 0) {
			slope = slopeHi
		}
		segs = append(segs, pseg{c0: c0, y0: y0, slope: slope})
		i = j
	}
	return segs
}

var nNodes, nSegs, nChildren, maxDepth int

func commonPrefixLen(a, b []byte) int {
	m := len(a)
	if len(b) < m {
		m = len(b)
	}
	for i := 0; i < m; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return m
}

func buildNode(a *arena, lo, hi, depth int, eps float64) *node {
	if depth > maxDepth {
		maxDepth = depth
	}
	nNodes++
	// coordinate offset = LCP of this range (sorted => LCP(first,last)); puts
	// float precision on the first differing byte, not a constant high byte.
	p := commonPrefixLen(a.key(lo), a.key(hi-1))
	n := &node{p: p}
	var cs []uint64
	var ranks []int
	flush := func() {
		if len(cs) > 0 {
			n.segs = append(n.segs, buildSegments(cs, ranks, eps)...)
			cs, ranks = cs[:0], ranks[:0]
		}
	}
	i := lo
	for i < hi {
		c := coordAt(a.key(i), p)
		j := i + 1
		for j < hi && coordAt(a.key(j), p) == c {
			j++
		}
		if j-i > int(2*eps+1) {
			// tie on these 8 bytes too big to absorb in a residual -> recurse deeper.
			// Break the segment here so the parent never interpolates across the
			// rank-gap the recursed group leaves behind.
			flush()
			if n.children == nil {
				n.children = map[uint64]*node{}
			}
			n.children[c] = buildNode(a, i, j, depth+1, eps)
			nChildren++
		} else {
			for r := i; r < j; r++ {
				cs = append(cs, c)
				ranks = append(ranks, r)
			}
		}
		i = j
	}
	flush()
	nSegs += len(n.segs)
	return n
}

func predict(n *node, key []byte, total int) int {
	for {
		c := coordAt(key, n.p)
		if n.children != nil {
			if ch, ok := n.children[c]; ok {
				n = ch
				continue
			}
		}
		lo, hi, idx := 0, len(n.segs)-1, 0
		for lo <= hi {
			mid := (lo + hi) / 2
			if n.segs[mid].c0 <= c {
				idx = mid
				lo = mid + 1
			} else {
				hi = mid - 1
			}
		}
		s := n.segs[idx]
		p := int(math.Round(s.y0 + s.slope*float64(c-s.c0)))
		if p < 0 {
			p = 0
		}
		if p >= total {
			p = total - 1
		}
		return p
	}
}

func main() {
	kvPath := flag.String("kv", "", ".kv data file")
	eps := flag.Int("eps", 31, "model error bound")
	flag.Parse()
	if *kvPath == "" {
		fmt.Println("need -kv")
		os.Exit(1)
	}
	logger := log.New()
	ctx := context.Background()
	fmt.Printf("=== %s  eps=%d (recursive) ===\n", filepath.Base(*kvPath), *eps)

	d, err := seg.NewDecompressor(*kvPath)
	must(err)
	defer d.Close()
	g := d.MakeGetter()
	g.Reset(0)
	cnt := d.Count() / 2
	ar := &arena{data: make([]byte, 0, 1<<30), off: make([]uint64, 0, cnt+1)}
	offs := make([]uint64, 0, cnt)
	buf := make([]byte, 0, 256)
	for g.HasNext() {
		key, valPos := g.Next(buf[:0])
		ar.off = append(ar.off, uint64(len(ar.data)))
		ar.data = append(ar.data, key...)
		offs = append(offs, valPos)
		if g.HasNext() {
			g.Skip()
		}
	}
	ar.off = append(ar.off, uint64(len(ar.data)))
	n := ar.n()
	maxOffset := offs[n-1]
	fmt.Printf("keys=%d  .kv=%.2f GB\n", n, float64(fileSize(*kvPath))/1e9)

	t0 := time.Now()
	root := buildNode(ar, 0, n, 0, float64(*eps))
	// model bytes: each seg = c0(8)+y0(8)+slope(8)=24; each child edge = coord(8)+ptr(8)=16
	modelBytes := nSegs*24 + nChildren*16
	fmt.Printf("model: nodes=%d segs=%d children=%d maxDepth=%d (%.3f bits/key) build=%s\n",
		nNodes, nSegs, nChildren, maxDepth, float64(modelBytes*8)/float64(n), time.Since(t0).Round(time.Millisecond))

	minE, maxE := math.MaxInt, math.MinInt
	residuals := make([]uint64, n)
	for i := 0; i < n; i++ {
		e := i - predict(root, ar.key(i), n)
		residuals[i] = uint64(e)
		if e < minE {
			minE = e
		}
		if e > maxE {
			maxE = e
		}
	}
	for i := 0; i < n; i++ {
		residuals[i] = uint64(int(int64(residuals[i])) - minE)
	}
	residBits := int(math.Ceil(math.Log2(float64(maxE - minE + 1))))
	fmt.Printf("residual: range=[%d,%d] -> %d bits/key\n", minE, maxE, residBits)

	// build recsplit(residual) + EF(offsets) for real sizes
	tmp, _ := os.MkdirTemp("", "mmphfrec")
	defer os.RemoveAll(tmp)
	idxPath := filepath.Join(tmp, "resid.kvi")
	var salt uint32 = 0
	rs, err := recsplit.NewRecSplit(recsplit.RecSplitArgs{
		KeyCount: n, Enums: false, BucketSize: recsplit.DefaultBucketSize,
		LeafSize: recsplit.DefaultLeafSize, TmpDir: tmp, IndexFile: idxPath,
		Salt: &salt, NoFsync: true,
	}, logger)
	must(err)
	for i := 0; i < n; i++ {
		must(rs.AddKey(ar.key(i), residuals[i]))
	}
	must(rs.Build(ctx))
	rs.Close()
	residSz := fileSize(idxPath)
	ef := eliasfano32.NewEliasFano(uint64(n), maxOffset)
	for i := 0; i < n; i++ {
		ef.AddOffset(offs[i])
	}
	ef.Build()
	efBytes := len(ef.AppendBytes(nil))

	// verify
	idx, err := recsplit.OpenIndex(idxPath)
	must(err)
	defer idx.Close()
	rd := recsplit.NewIndexReader(idx)
	efr, _ := eliasfano32.ReadEliasFano(ef.AppendBytes(nil))
	bad := 0
	for i := 0; i < n; i++ {
		local, _ := rd.Lookup(ar.key(i))
		rank := predict(root, ar.key(i), n) + minE + int(local)
		if rank < 0 || rank >= n || efr.Get(uint64(rank)) != offs[i] {
			bad++
		}
	}
	fmt.Printf("verify: %d/%d correct\n", n-bad, n)

	newTotal := residSz + int64(efBytes) + int64(modelBytes)
	mphfOverhead := float64(residSz*8)/float64(n) - 8*math.Ceil(float64(maxE-minE+1)/256)
	if mphfOverhead < 0 {
		mphfOverhead = 0
	}
	efBitsPK := float64(efBytes*8) / float64(n)
	idealBits := efBitsPK + float64(modelBytes*8)/float64(n) + mphfOverhead + float64(residBits)
	baseKvi := (*kvPath)[:len(*kvPath)-3] + ".kvi"
	baseSz := fileSize(baseKvi)
	fmt.Printf("recsplit(resid)=%.0fMB EF=%.0fMB model=%.1fMB\n", mb(residSz), float64(efBytes)/1e6, float64(modelBytes)/1e6)
	if baseSz > 0 {
		fmt.Printf("baseline .kvi: %.0f MB (%.1f bits/key)\n", mb(baseSz), float64(baseSz*8)/float64(n))
	}
	fmt.Printf("new (recsplit-bound): %.0f MB (%.1f bits/key)\n", mb(newTotal), float64(newTotal*8)/float64(n))
	fmt.Printf("new (bit-packed ideal): %.1f bits/key [EF=%.1f + model=%.3f + mphf=%.1f + resid=%d]\n",
		idealBits, efBitsPK, float64(modelBytes*8)/float64(n), mphfOverhead, residBits)
	if baseSz > 0 {
		fmt.Printf("reduction: recsplit-bound %.2fx | ideal %.2fx\n",
			float64(baseSz)/float64(newTotal), float64(baseSz*8)/float64(n)/idealBits)
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
func mb(b int64) float64 { return float64(b) / 1e6 }
func fileSize(p string) int64 {
	fi, err := os.Stat(p)
	if err != nil {
		return 0
	}
	return fi.Size()
}
func must(err error) {
	if err != nil {
		panic(err)
	}
}
