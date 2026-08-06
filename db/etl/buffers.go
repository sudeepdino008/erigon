// Copyright 2021 The Erigon Authors
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

package etl

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/c2h5oh/datasize"

	"github.com/erigontech/erigon/common/dbg"
)

const (
	//SliceBuffer - just simple slice w
	SortableSliceBuffer = iota
	//SortableAppendBuffer - map[k] [v1 v2 v3]
	SortableAppendBuffer
	// SortableOldestAppearedBuffer - buffer that keeps only the oldest entries.
	// if first v1 was added under key K, then v2; only v1 will stay
	SortableOldestAppearedBuffer

	//BufIOSize - 128 pages | default is 1 page | increasing over `64 * 4096` doesn't show speedup on SSD/NVMe, but show speedup in cloud drives
	BufIOSize = 128 * 4096

	entryLocSize = 12 // sizeof(entryLoc): offset(4) + keyLen(4) + valLen(4)
)

// writeSortedEntries writes buffer entries to w in varint-length-prefixed format.
func writeSortedEntries(w io.Writer, entries []sortableBufferEntry) error {
	var numBuf [binary.MaxVarintLen64]byte
	for _, entry := range entries {
		lk := int64(len(entry.key))
		if entry.key == nil {
			lk = -1
		}
		n := binary.PutVarint(numBuf[:], lk)
		if _, err := w.Write(numBuf[:n]); err != nil {
			return err
		}
		if _, err := w.Write(entry.key); err != nil {
			return err
		}
		lv := int64(len(entry.value))
		if entry.value == nil {
			lv = -1
		}
		n = binary.PutVarint(numBuf[:], lv)
		if _, err := w.Write(numBuf[:n]); err != nil {
			return err
		}
		if _, err := w.Write(entry.value); err != nil {
			return err
		}
	}
	return nil
}

var BufferOptimalSize = dbg.EnvDataSize("ETL_OPTIMAL", 256*datasize.MB) /*  var because we want to sometimes change it from tests or command-line flags */

// etlSmallBufRAM (BufferOptimalSize/8) bounds the flush threshold so a full
// set of domain/history/index flush collectors (~17 per batch writer) stays
// around 512 MB when all run full. Pooled buffers start empty and grow with
// the data they actually see; grown capacity survives reuse (Reset preserves
// cap), so hot collectors amortize growth while never-full ones stay small.
var etlSmallBufRAM = dbg.EnvDataSize("ETL_SMALL", BufferOptimalSize/8)
var SmallSortableBuffers = NewAllocator(&sync.Pool{
	New: func() any {
		return NewSortableBuffer(etlSmallBufRAM)
	},
})
var etlLargeBufRAM = BufferOptimalSize
var LargeSortableBuffers = NewAllocator(&sync.Pool{
	New: func() any {
		return NewSortableBuffer(etlLargeBufRAM)
	},
})

type Buffer interface {
	// Put does copy `k` and `v`
	Put(k, v []byte)
	// Get returns direct references to the internal key/value storage without copying.
	// The returned slices must not be modified by the caller.
	Get(i int) ([]byte, []byte)
	Len() int
	Reset()
	SizeLimit() int
	Prealloc(predictKeysAmount, predictDataAmount int) Buffer
	Write(io.Writer) error
	Sort()
	CheckFlushSize() bool
}

type sortableBufferEntry struct {
	key   []byte
	value []byte
}

var (
	_ Buffer = &sortableBuffer{}
	_ Buffer = &appendSortableBuffer{}
	_ Buffer = &oldestEntrySortableBuffer{}
)

// entryLoc stores the location of a key/value pair within sortableBuffer.data.
// Key occupies data[offset : offset+keyLen], value follows at data[offset+max(0,keyLen) : ...+valLen].
// keyLen/valLen of -1 indicates nil.
type entryLoc struct {
	offset int32
	keyLen int32
	valLen int32
}

// keyPrefix8 loads the first up-to-8 bytes of k as a big-endian uint64,
// zero-padded on the right, so uint64 comparison matches lexicographic order
// for that prefix (ties fall back to a full byte compare).
func keyPrefix8(k []byte) uint64 {
	if len(k) >= 8 {
		return binary.BigEndian.Uint64(k)
	}
	var p uint64
	for i := 0; i < len(k); i++ {
		p |= uint64(k[i]) << (56 - 8*i)
	}
	return p
}

func NewSortableBuffer(bufferOptimalSize datasize.ByteSize) *sortableBuffer {
	if bufferOptimalSize.Bytes() > math.MaxInt32 {
		panic(fmt.Sprintf("etl: sortableBuffer size %d exceeds MaxInt32", bufferOptimalSize.Bytes()))
	}
	return &sortableBuffer{
		optimalSize: int(bufferOptimalSize.Bytes()),
	}
}

type sortableBuffer struct {
	entries     []entryLoc
	data        []byte
	optimalSize int
	sortScratch []entryLoc
	rk          []radixKey
	rkScratch   []radixKey
}

// radixKey is the compact record radix-sorted in place of the full entryLoc,
// keeping per-pass scatter traffic small. idx points back into b.entries.
type radixKey struct {
	prefix uint64
	idx    uint32
}

// Put adds key and value to the buffer. These slices will not be accessed later,
// so no copying is necessary
func (b *sortableBuffer) Put(k, v []byte) {
	e := entryLoc{
		offset: int32(len(b.data)), //nolint:gosec
		keyLen: int32(len(k)),      //nolint:gosec
		valLen: int32(len(v)),      //nolint:gosec
	}
	if k == nil {
		e.keyLen = -1
	}
	if v == nil {
		e.valLen = -1
	}
	b.entries = append(b.entries, e)
	b.data = append(append(b.data, k...), v...)
}

func (b *sortableBuffer) Size() int { return len(b.data) + len(b.entries)*entryLocSize }

func (b *sortableBuffer) Len() int {
	return len(b.entries)
}

func (b *sortableBuffer) Get(i int) ([]byte, []byte) {
	e := &b.entries[i]
	kLen, vLen := int(e.keyLen), int(e.valLen)
	keyOffset := int(e.offset)
	valOffset := keyOffset
	if kLen > 0 {
		valOffset += kLen
	}
	var key, val []byte
	if kLen > 0 {
		key = b.data[keyOffset : keyOffset+kLen]
	} else if kLen == 0 {
		key = []byte{}
	}
	if vLen > 0 {
		val = b.data[valOffset : valOffset+vLen]
	} else if vLen == 0 {
		val = []byte{}
	}
	return key, val
}

func (b *sortableBuffer) Prealloc(predictKeysAmount, predictDataSize int) Buffer {
	if cap(b.entries) < predictKeysAmount {
		b.entries = make([]entryLoc, 0, predictKeysAmount)
	}
	if cap(b.data) < predictDataSize {
		b.data = make([]byte, 0, predictDataSize)
	}
	return b
}

func (b *sortableBuffer) Reset() {
	b.entries = b.entries[:0]
	b.data = b.data[:0]
}
func (b *sortableBuffer) SizeLimit() int { return b.optimalSize }
func (b *sortableBuffer) Sort() {
	data := b.data
	cmp := func(a, b entryLoc) int {
		aKey := data[a.offset : a.offset+max(a.keyLen, 0)]
		bKey := data[b.offset : b.offset+max(b.keyLen, 0)]
		return bytes.Compare(aKey, bKey) // equal keys count as sorted; input is already insertion order
	}
	if slices.IsSortedFunc(b.entries, cmp) {
		return
	}

	b.radixSortEntries()
}

// radixSmallRun: runs at or below this size are comparison-sorted rather than
// recursed into another radix level.
const radixSmallRun = 32

// radixSortEntries stably sorts b.entries by full key using MSD radix on 8-byte
// key chunks (b.rk carries indices; entries are gathered once at the end).
func (b *sortableBuffer) radixSortEntries() {
	n := len(b.entries)
	if n == 0 {
		return
	}
	if cap(b.rk) < n {
		b.rk = make([]radixKey, n)
		b.rkScratch = make([]radixKey, n)
	}
	rk, scratch := b.rk[:n], b.rkScratch[:n]
	for i := 0; i < n; i++ {
		rk[i] = radixKey{prefix: b.prefixAt(uint32(i), 0), idx: uint32(i)} //nolint:gosec
	}
	sorted := rk
	switch w := radixWorkers(n); {
	case w <= 0: // forced serial: no MSD partition
		b.msdSort(rk, scratch, 0)
	case n >= radixPartitionMin: // MSD partition (cache win); w==1 sorts buckets inline (serial)
		sorted = b.partitionedMSDSort(rk, scratch, w)
	default: // tiny n: partition not worth it
		b.msdSort(rk, scratch, 0)
	}

	if cap(b.sortScratch) < n {
		b.sortScratch = make([]entryLoc, n)
	}
	gathered := b.sortScratch[:n]
	for i := 0; i < n; i++ {
		gathered[i] = b.entries[sorted[i].idx]
	}
	b.entries, b.sortScratch = gathered, b.entries[:0]
}

// radixPartitionMin: above this, MSD-partition the top byte first — it localizes each
// bucket to cache and is a win even single-threaded. parallelRadixMin: above this, also
// fan the buckets across goroutines.
const radixPartitionMin = 16 * 1024
const parallelRadixMin = 128 * 1024

// radixWorkersOverride: 0=auto, -1=force serial (no MSD partition), N>0=force N workers.
var radixWorkersOverride = dbg.EnvInt("ETL_RADIX_WORKERS", 0)

// radixWorkers returns how many bucket-sort goroutines to use (1 = inline/serial).
func radixWorkers(n int) int {
	if radixWorkersOverride != 0 {
		return radixWorkersOverride
	}
	if n < parallelRadixMin {
		return 1
	}
	w := runtime.GOMAXPROCS(0)
	if w > 16 {
		w = 16
	}
	return w
}

// partitionedMSDSort partitions rk by the most-significant key byte into 256 contiguous
// buckets (landing in scratch), then sorts the buckets. Partitioning localizes each
// bucket to cache, so it helps even with a single worker. Buckets are disjoint
// sub-slices, so workers never touch shared mutable state. Returns the result slice.
func (b *sortableBuffer) partitionedMSDSort(rk, scratch []radixKey, workers int) []radixKey {
	n := len(rk)
	const shift = 56 // most-significant byte of the big-endian prefix == first key byte
	var start [257]int
	for i := 0; i < n; i++ {
		start[((rk[i].prefix>>shift)&0xff)+1]++
	}
	for d := 1; d <= 256; d++ {
		start[d] += start[d-1]
	}
	pos := start // copy; advanced during scatter
	for i := 0; i < n; i++ {
		d := (rk[i].prefix >> shift) & 0xff
		scratch[pos[d]] = rk[i]
		pos[d]++
	}

	if workers <= 1 {
		for d := 0; d < 256; d++ {
			if lo, hi := start[d], start[d+1]; hi-lo > 1 {
				b.msdSort(scratch[lo:hi], rk[lo:hi], 0)
			}
		}
		return scratch
	}

	var next atomic.Int32
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				d := int(next.Add(1)) - 1
				if d >= 256 {
					return
				}
				if lo, hi := start[d], start[d+1]; hi-lo > 1 {
					b.msdSort(scratch[lo:hi], rk[lo:hi], 0)
				}
			}
		}()
	}
	wg.Wait()
	return scratch
}

// prefixAt loads up to 8 key bytes starting at off as a big-endian uint64.
func (b *sortableBuffer) prefixAt(idx uint32, off int) uint64 {
	e := &b.entries[idx]
	kl := int(max(e.keyLen, 0))
	if off >= kl {
		return 0
	}
	return keyPrefix8(b.data[int(e.offset)+off : int(e.offset)+kl])
}

// msdSort stably sorts rk by the key bytes from byteOff onward, given all keys
// already agree on bytes [0,byteOff). scratch is a same-length companion buffer.
func (b *sortableBuffer) msdSort(rk, scratch []radixKey, byteOff int) {
	n := len(rk)
	if n <= radixSmallRun {
		b.smallSort(rk, byteOff)
		return
	}
	if byteOff > 0 {
		anyLonger := false
		for i := 0; i < n; i++ {
			if int(max(b.entries[rk[i].idx].keyLen, 0)) > byteOff {
				anyLonger = true
			}
			rk[i].prefix = b.prefixAt(rk[i].idx, byteOff)
		}
		if !anyLonger { // all keys end at/before byteOff: order by length/insertion via full compare
			b.smallSort(rk, byteOff)
			return
		}
	}
	var counts [8][256]int
	for i := 0; i < n; i++ {
		p := rk[i].prefix
		counts[0][p&0xff]++
		counts[1][(p>>8)&0xff]++
		counts[2][(p>>16)&0xff]++
		counts[3][(p>>24)&0xff]++
		counts[4][(p>>32)&0xff]++
		counts[5][(p>>40)&0xff]++
		counts[6][(p>>48)&0xff]++
		counts[7][(p>>56)&0xff]++
	}
	src, dst := rk, scratch
	for pass := 0; pass < 8; pass++ {
		shift := uint(pass * 8)
		count := &counts[pass]
		if count[(src[0].prefix>>shift)&0xff] == n {
			continue
		}
		sum := 0
		for i := 0; i < 256; i++ {
			c := count[i]
			count[i] = sum
			sum += c
		}
		for i := 0; i < n; i++ {
			byteVal := (src[i].prefix >> shift) & 0xff
			dst[count[byteVal]] = src[i]
			count[byteVal]++
		}
		src, dst = dst, src
	}
	if &src[0] != &rk[0] {
		copy(rk, src)
	}
	for i := 0; i < n; {
		j := i + 1
		for j < n && rk[j].prefix == rk[i].prefix {
			j++
		}
		if j-i > 1 {
			b.msdSort(rk[i:j], scratch[i:j], byteOff+8)
		}
		i = j
	}
}

// smallSort comparison-sorts rk by key bytes from byteOff (stable via idx, the insertion order).
func (b *sortableBuffer) smallSort(rk []radixKey, byteOff int) {
	if len(rk) <= 1 {
		return
	}
	data := b.data
	slices.SortFunc(rk, func(x, y radixKey) int {
		ex, ey := &b.entries[x.idx], &b.entries[y.idx]
		kx := data[ex.offset : int(ex.offset)+int(max(ex.keyLen, 0))]
		ky := data[ey.offset : int(ey.offset)+int(max(ey.keyLen, 0))]
		off := min(byteOff, len(kx), len(ky)) // bytes [0,off) are equal; safe to skip
		if r := bytes.Compare(kx[off:], ky[off:]); r != 0 {
			return r
		}
		return int(x.idx) - int(y.idx)
	})
}

func (b *sortableBuffer) CheckFlushSize() bool {
	return b.Size() >= b.optimalSize
}

func (b *sortableBuffer) Write(w io.Writer) error {
	var numBuf [binary.MaxVarintLen64]byte
	for i := range b.entries {
		e := &b.entries[i]
		kLen, vLen := int(e.keyLen), int(e.valLen)
		keyOffset := int(e.offset)
		valOffset := keyOffset
		if kLen > 0 {
			valOffset += kLen
		}
		// write key
		n := binary.PutVarint(numBuf[:], int64(e.keyLen))
		if _, err := w.Write(numBuf[:n]); err != nil {
			return err
		}
		if kLen > 0 {
			if _, err := w.Write(b.data[keyOffset : keyOffset+kLen]); err != nil {
				return err
			}
		}
		// write value
		n = binary.PutVarint(numBuf[:], int64(e.valLen))
		if _, err := w.Write(numBuf[:n]); err != nil {
			return err
		}
		if vLen > 0 {
			if _, err := w.Write(b.data[valOffset : valOffset+vLen]); err != nil {
				return err
			}
		}
	}
	return nil
}

func NewAppendBuffer(bufferOptimalSize datasize.ByteSize) *appendSortableBuffer {
	return &appendSortableBuffer{
		entries:     make(map[string][]byte),
		size:        0,
		optimalSize: int(bufferOptimalSize.Bytes()),
	}
}

type appendSortableBuffer struct {
	entries     map[string][]byte
	sortedBuf   []sortableBufferEntry
	size        int
	optimalSize int
}

func (b *appendSortableBuffer) Put(k, v []byte) {
	stored, ok := b.entries[string(k)]
	if !ok {
		b.size += len(k)
	}
	b.size += len(v)
	b.entries[string(k)] = append(stored, v...)
}

func (b *appendSortableBuffer) Size() int      { return b.size }
func (b *appendSortableBuffer) SizeLimit() int { return b.optimalSize }

func (b *appendSortableBuffer) Len() int {
	return len(b.entries)
}
func (b *appendSortableBuffer) Sort() {
	b.sortedBuf = b.sortedBuf[:0]
	if cap(b.sortedBuf) < len(b.entries) {
		b.sortedBuf = make([]sortableBufferEntry, 0, len(b.entries))
	}
	for key, val := range b.entries {
		b.sortedBuf = append(b.sortedBuf, sortableBufferEntry{key: []byte(key), value: val})
	}
	sort.Sort(b) // Doesn't need `sort.Stable` because this buffer type can't produce duplicated values
}

func (b *appendSortableBuffer) Less(i, j int) bool {
	return bytes.Compare(b.sortedBuf[i].key, b.sortedBuf[j].key) < 0
}

func (b *appendSortableBuffer) Swap(i, j int) {
	b.sortedBuf[i], b.sortedBuf[j] = b.sortedBuf[j], b.sortedBuf[i]
}

func (b *appendSortableBuffer) Get(i int) ([]byte, []byte) {
	return b.sortedBuf[i].key, b.sortedBuf[i].value
}
func (b *appendSortableBuffer) Reset() {
	b.sortedBuf = nil
	b.entries = make(map[string][]byte)
	b.size = 0
}
func (b *appendSortableBuffer) Prealloc(predictKeysAmount, predictDataSize int) Buffer {
	b.entries = make(map[string][]byte, predictKeysAmount) // maps have no cap(), always recreate
	if cap(b.sortedBuf) < predictKeysAmount {
		b.sortedBuf = make([]sortableBufferEntry, 0, predictKeysAmount)
	}
	return b
}

func (b *appendSortableBuffer) Write(w io.Writer) error {
	return writeSortedEntries(w, b.sortedBuf)
}

func (b *appendSortableBuffer) CheckFlushSize() bool {
	return b.size >= b.optimalSize
}

func NewOldestEntryBuffer(bufferOptimalSize datasize.ByteSize) *oldestEntrySortableBuffer {
	return &oldestEntrySortableBuffer{
		entries:     make(map[string][]byte),
		size:        0,
		optimalSize: int(bufferOptimalSize.Bytes()),
	}
}

type oldestEntrySortableBuffer struct {
	entries     map[string][]byte
	sortedBuf   []sortableBufferEntry
	size        int
	optimalSize int
}

func (b *oldestEntrySortableBuffer) Put(k, v []byte) {
	_, ok := b.entries[string(k)]
	if ok {
		// if we already had this entry, we are going to keep it and ignore new value
		return
	}

	b.size += len(k)*2 + len(v)
	b.entries[string(k)] = bytes.Clone(v)
}

func (b *oldestEntrySortableBuffer) Size() int      { return b.size }
func (b *oldestEntrySortableBuffer) SizeLimit() int { return b.optimalSize }

func (b *oldestEntrySortableBuffer) Len() int {
	return len(b.entries)
}

func (b *oldestEntrySortableBuffer) Sort() {
	b.sortedBuf = b.sortedBuf[:0]
	if cap(b.sortedBuf) < len(b.entries) {
		b.sortedBuf = make([]sortableBufferEntry, 0, len(b.entries))
	}
	for k, v := range b.entries {
		b.sortedBuf = append(b.sortedBuf, sortableBufferEntry{key: []byte(k), value: v})
	}
	sort.Sort(b) // Doesn't need `sort.Stable` because this buffer type can't produce duplicated values
}

func (b *oldestEntrySortableBuffer) Less(i, j int) bool {
	return bytes.Compare(b.sortedBuf[i].key, b.sortedBuf[j].key) < 0
}

func (b *oldestEntrySortableBuffer) Swap(i, j int) {
	b.sortedBuf[i], b.sortedBuf[j] = b.sortedBuf[j], b.sortedBuf[i]
}

func (b *oldestEntrySortableBuffer) Get(i int) ([]byte, []byte) {
	return b.sortedBuf[i].key, b.sortedBuf[i].value
}
func (b *oldestEntrySortableBuffer) Reset() {
	b.sortedBuf = nil
	b.entries = make(map[string][]byte)
	b.size = 0
}
func (b *oldestEntrySortableBuffer) Prealloc(predictKeysAmount, predictDataSize int) Buffer {
	b.entries = make(map[string][]byte, predictKeysAmount) // maps have no cap(), always recreate
	if cap(b.sortedBuf) < predictKeysAmount {
		b.sortedBuf = make([]sortableBufferEntry, 0, predictKeysAmount)
	}
	return b
}

func (b *oldestEntrySortableBuffer) Write(w io.Writer) error {
	return writeSortedEntries(w, b.sortedBuf)
}
func (b *oldestEntrySortableBuffer) CheckFlushSize() bool {
	return b.size >= b.optimalSize
}

func getBufferByType(tp int, size datasize.ByteSize) Buffer {
	switch tp {
	case SortableSliceBuffer:
		return NewSortableBuffer(size)
	case SortableAppendBuffer:
		return NewAppendBuffer(size)
	case SortableOldestAppearedBuffer:
		return NewOldestEntryBuffer(size)
	default:
		panic("unknown buffer type " + strconv.Itoa(tp))
	}
}

func getTypeByBuffer(b Buffer) int {
	switch b.(type) {
	case *sortableBuffer:
		return SortableSliceBuffer
	case *appendSortableBuffer:
		return SortableAppendBuffer
	case *oldestEntrySortableBuffer:
		return SortableOldestAppearedBuffer
	default:
		panic(fmt.Sprintf("unknown buffer type: %T ", b))
	}
}
