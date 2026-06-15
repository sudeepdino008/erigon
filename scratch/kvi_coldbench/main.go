// Cold/warm full-Get benchmark for the learned MMPHF+EF .kvi.
//
// Cold full-Get (key+value) over N random keys with `vmtouch -e` eviction,
// comparing baseline recsplit .kvi against the new MMPHF+EF index. The
// value-read code is identical for both paths, so the latency difference is
// purely index-faulting cost.
//
// The whole index is written to disk (residual.kvi + offsets.ef + model).
// recsplit residual + EF are evicted/faulted like the baseline .kvi; the model
// is pinned resident, mirroring how .bt keeps its pivot array resident.
//
// Lean build: keys live in a single arena ([]byte + offsets), x is recomputed
// on the fly, so it scales to the big bloatnet files without an N-sized u128 array.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/erigontech/erigon/common/log/v3"
	"github.com/erigontech/erigon/common/mmap"
	"github.com/erigontech/erigon/db/recsplit"
	"github.com/erigontech/erigon/db/recsplit/eliasfano32"
	"github.com/erigontech/erigon/db/seg"
)

const K = 8

type u128 [K]uint64

const limbBase = 1.8446744073709552e19

func keyToX(k []byte) u128 {
	var x u128
	for i := 0; i < K*8; i++ {
		limb := i / 8
		x[limb] <<= 8
		if i < len(k) {
			x[limb] |= uint64(k[i])
		}
	}
	return x
}
func less(a, b u128) bool {
	for i := 0; i < K; i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}
func leq(a, b u128) bool { return !less(b, a) }
func fdelta(a, b u128) float64 {
	var d float64
	for i := 0; i < K; i++ {
		d = d*limbBase + (float64(a[i]) - float64(b[i]))
	}
	return d
}

type segment struct {
	x0    u128
	y0    float64
	slope float64
}

// key arena
type arena struct {
	data []byte
	off  []uint64 // len n+1
}

func (a *arena) n() int          { return len(a.off) - 1 }
func (a *arena) key(i int) []byte { return a.data[a.off[i]:a.off[i+1]] }
func (a *arena) x(i int) u128     { return keyToX(a.key(i)) }

func buildSegments(a *arena, eps float64) []segment {
	var segs []segment
	n := a.n()
	i := 0
	for i < n {
		x0 := a.x(i)
		y0 := float64(i)
		slopeLo, slopeHi := math.Inf(-1), math.Inf(1)
		j := i + 1
		for ; j < n; j++ {
			dx := fdelta(a.x(j), x0)
			yj := float64(j)
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
		segs = append(segs, segment{x0: x0, y0: y0, slope: slope})
		i = j
	}
	return segs
}

func predict(segs []segment, x u128, n int) int {
	lo, hi, idx := 0, len(segs)-1, 0
	for lo <= hi {
		mid := (lo + hi) / 2
		if leq(segs[mid].x0, x) {
			idx = mid
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	s := segs[idx]
	p := int(math.Round(s.y0 + s.slope*fdelta(x, s.x0)))
	if p < 0 {
		p = 0
	}
	if p >= n {
		p = n - 1
	}
	return p
}

const segBytes = K*8 + 16 // x0 + y0 + slope

func writeModel(path string, segs []segment) int64 {
	buf := make([]byte, len(segs)*segBytes)
	for s, seg := range segs {
		o := s * segBytes
		for l := 0; l < K; l++ {
			binary.BigEndian.PutUint64(buf[o+l*8:], seg.x0[l])
		}
		binary.BigEndian.PutUint64(buf[o+K*8:], math.Float64bits(seg.y0))
		binary.BigEndian.PutUint64(buf[o+K*8+8:], math.Float64bits(seg.slope))
	}
	must(os.WriteFile(path, buf, 0o644))
	return int64(len(buf))
}

func readModel(path string) []segment {
	buf, err := os.ReadFile(path)
	must(err)
	segs := make([]segment, len(buf)/segBytes)
	for s := range segs {
		o := s * segBytes
		for l := 0; l < K; l++ {
			segs[s].x0[l] = binary.BigEndian.Uint64(buf[o+l*8:])
		}
		segs[s].y0 = math.Float64frombits(binary.BigEndian.Uint64(buf[o+K*8:]))
		segs[s].slope = math.Float64frombits(binary.BigEndian.Uint64(buf[o+K*8+8:]))
	}
	return segs
}

func main() {
	kvPath := flag.String("kv", "", ".kv data file")
	eps := flag.Int("eps", 31, "model error bound")
	out := flag.String("out", "", "artifact dir (default: tmp)")
	nkeys := flag.Int("nkeys", 15000, "random keys to time")
	flag.Parse()
	if *kvPath == "" {
		fmt.Println("need -kv")
		os.Exit(1)
	}
	logger := log.New()
	ctx := context.Background()
	if *out == "" {
		*out, _ = os.MkdirTemp("", "coldbench")
	}
	os.MkdirAll(*out, 0o755)
	base := filepath.Base(*kvPath)
	fmt.Printf("=== %s  eps=%d  nkeys=%d ===\n", base, *eps, *nkeys)

	// --- extract into arena ---
	d, err := seg.NewDecompressor(*kvPath)
	must(err)
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
	d.Close() // drop the map so vmtouch can evict .kv
	fmt.Printf("keys=%d  .kv=%.2f GB  keyArena=%.1f GB\n", n, float64(fileSize(*kvPath))/1e9, float64(len(ar.data))/1e9)

	// --- model ---
	segs := buildSegments(ar, float64(*eps))
	minE, maxE := math.MaxInt, math.MinInt
	residuals := make([]uint32, n)
	for i := 0; i < n; i++ {
		e := i - predict(segs, ar.x(i), n)
		residuals[i] = uint32(e) // bias applied after minE known
		if e < minE {
			minE = e
		}
		if e > maxE {
			maxE = e
		}
	}
	for i := 0; i < n; i++ {
		residuals[i] = uint32(int(int32(residuals[i])) - minE)
	}
	residBits := int(math.Ceil(math.Log2(float64(maxE - minE + 1))))
	fmt.Printf("model: segs=%d  resid range=[%d,%d] (%d bits/key)\n", len(segs), minE, maxE, residBits)

	// --- artifacts on disk ---
	residPath := filepath.Join(*out, base+".mmphf.kvi")
	efPath := filepath.Join(*out, base+".mmphf.ef")
	modelPath := filepath.Join(*out, base+".mmphf.model")
	var salt uint32 = 0
	rs, err := recsplit.NewRecSplit(recsplit.RecSplitArgs{
		KeyCount: n, Enums: false, BucketSize: recsplit.DefaultBucketSize,
		LeafSize: recsplit.DefaultLeafSize, TmpDir: *out, IndexFile: residPath,
		Salt: &salt, NoFsync: true,
	}, logger)
	must(err)
	for i := 0; i < n; i++ {
		must(rs.AddKey(ar.key(i), uint64(residuals[i])))
	}
	must(rs.Build(ctx))
	rs.Close()
	ef := eliasfano32.NewEliasFano(uint64(n), maxOffset)
	for i := 0; i < n; i++ {
		ef.AddOffset(offs[i])
	}
	ef.Build()
	must(os.WriteFile(efPath, ef.AppendBytes(nil), 0o644))
	modelSz := writeModel(modelPath, segs)

	// A/B: plain enums=true (full rank stored permuted + internal offset EF), no model
	enumsPath := filepath.Join(*out, base+".enums.kvi")
	var salt2 uint32 = 0
	rsE, err := recsplit.NewRecSplit(recsplit.RecSplitArgs{
		KeyCount: n, Enums: true, BucketSize: recsplit.DefaultBucketSize,
		LeafSize: recsplit.DefaultLeafSize, TmpDir: *out, IndexFile: enumsPath,
		Salt: &salt2, NoFsync: true,
	}, logger)
	must(err)
	for i := 0; i < n; i++ {
		must(rsE.AddKey(ar.key(i), offs[i]))
	}
	must(rsE.Build(ctx))
	rsE.Close()
	enumsSz := fileSize(enumsPath)

	residSz, efSz := fileSize(residPath), fileSize(efPath)
	newTotal := residSz + efSz + modelSz
	baseKvi := (*kvPath)[:len(*kvPath)-3] + ".kvi"
	baseSz := fileSize(baseKvi)
	fmt.Printf("index sizes: new=%.0f MB (resid %.0f + ef %.0f + model %.0f) | enums=true=%.0f MB | baseline .kvi=%.0f MB\n",
		mb(newTotal), mb(residSz), mb(efSz), mb(modelSz), mb(enumsSz), mb(baseSz))
	fmt.Printf("  new is %.2fx smaller than baseline, %.2fx smaller than enums=true (model saves %.0f MB vs enums)\n",
		float64(baseSz)/float64(newTotal), float64(enumsSz)/float64(newTotal), mb(enumsSz-newTotal))

	// free build-time arena residuals (keep arena keys + offs for queries)
	residuals = nil

	rng := rand.New(rand.NewSource(1))
	idxs := make([]int, *nkeys)
	for i := range idxs {
		idxs[i] = rng.Intn(n)
	}
	evict := func(files ...string) { _ = exec.Command("vmtouch", append([]string{"-e"}, files...)...).Run() }
	load := func(files ...string) { _ = exec.Command("vmtouch", append([]string{"-t"}, files...)...).Run() }

	runNew := func() float64 {
		mdl := readModel(modelPath) // pinned-resident, like .bt pivots
		idx, err := recsplit.OpenIndex(residPath)
		must(err)
		rd := recsplit.NewIndexReader(idx)
		efr, efHandle := openEF(efPath)
		dv, err := seg.NewDecompressor(*kvPath)
		must(err)
		vg := dv.MakeGetter()
		vbuf := make([]byte, 0, 4096)
		var sink int
		t0 := time.Now()
		for _, i := range idxs {
			local, _ := rd.Lookup(ar.key(i))
			rank := predict(mdl, ar.x(i), n) + minE + int(local)
			vg.Reset(efr.Get(uint64(rank)))
			if vg.HasNext() {
				vbuf, _ = vg.Next(vbuf[:0])
			}
			sink += len(vbuf)
		}
		us := usOp(time.Since(t0), *nkeys)
		idx.Close()
		closeEF(efHandle)
		dv.Close()
		_ = sink
		return us
	}
	runBase := func() float64 {
		idx, err := recsplit.OpenIndex(baseKvi)
		must(err)
		rd := recsplit.NewIndexReader(idx)
		dv, err := seg.NewDecompressor(*kvPath)
		must(err)
		vg := dv.MakeGetter()
		vbuf := make([]byte, 0, 4096)
		var sink int
		t0 := time.Now()
		for _, i := range idxs {
			off, _ := rd.Lookup(ar.key(i))
			vg.Reset(off)
			if vg.HasNext() {
				vbuf, _ = vg.Next(vbuf[:0])
			}
			sink += len(vbuf)
		}
		us := usOp(time.Since(t0), *nkeys)
		idx.Close()
		dv.Close()
		_ = sink
		return us
	}

	runEnums := func() float64 {
		idx, err := recsplit.OpenIndex(enumsPath)
		must(err)
		rd := recsplit.NewIndexReader(idx)
		dv, err := seg.NewDecompressor(*kvPath)
		must(err)
		vg := dv.MakeGetter()
		vbuf := make([]byte, 0, 4096)
		var sink int
		t0 := time.Now()
		for _, i := range idxs {
			off, _ := rd.TwoLayerLookup(ar.key(i))
			vg.Reset(off)
			if vg.HasNext() {
				vbuf, _ = vg.Next(vbuf[:0])
			}
			sink += len(vbuf)
		}
		us := usOp(time.Since(t0), *nkeys)
		idx.Close()
		dv.Close()
		_ = sink
		return us
	}

	fmt.Println("\n--- COLD (index + .kv evicted; model pinned, like .bt pivots) ---")
	evict(residPath, efPath, *kvPath)
	load(modelPath)
	newCold := runNew()
	fmt.Printf("new MMPHF+EF:  %.0f µs/op\n", newCold)
	evict(enumsPath, *kvPath)
	fmt.Printf("enums=true:    %.0f µs/op\n", runEnums())
	if baseSz > 0 {
		evict(baseKvi, *kvPath)
		baseCold := runBase()
		fmt.Printf("baseline .kvi: %.0f µs/op  (new is %.2fx %s)\n", baseCold, ratio(baseCold, newCold), faster(baseCold, newCold))
	}

	fmt.Println("\n--- WARM (index resident, .kv cold) ---")
	load(residPath, efPath, modelPath)
	evict(*kvPath)
	fmt.Printf("new MMPHF+EF:  %.0f µs/op\n", runNew())
	load(enumsPath)
	evict(*kvPath)
	fmt.Printf("enums=true:    %.0f µs/op\n", runEnums())
	if baseSz > 0 {
		load(baseKvi)
		evict(*kvPath)
		fmt.Printf("baseline .kvi: %.0f µs/op\n", runBase())
	}
}

func openEF(path string) (*eliasfano32.EliasFano, []byte) {
	f, err := os.Open(path)
	must(err)
	fi, _ := f.Stat()
	m, _, err := mmap.Mmap(f, int(fi.Size()))
	must(err)
	f.Close()
	ef, _ := eliasfano32.ReadEliasFano(m)
	return ef, m
}
func closeEF(m []byte) { _ = mmap.Munmap(m, nil) }

func mb(b int64) float64 { return float64(b) / 1e6 }
func usOp(d time.Duration, n int) float64 {
	return float64(d.Microseconds()) / float64(n)
}
func ratio(a, b float64) float64 {
	if a > b {
		return a / b
	}
	return b / a
}
func faster(base, new float64) string {
	if new < base {
		return "faster"
	}
	return "slower"
}
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
