package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
)

// FileHeader mirrors the C struct:
// magic[4], version uint32, vocab_size uint32, merge_count uint32, reserved uint32
const magicStr = "BPE2"

type Token struct {
	Left  uint32
	Right uint32
}

type Tokenizer struct {
	Vocab  []Token           // id -> {left, right}
	Merges []uint64          // pair_t in merge order: (left<<32)|right
	Rank   map[uint64]int    // pair -> merge priority (lower = earlier/higher priority)
	PairID map[uint64]uint32 // pair -> resulting token id
}

func makePair(a, b uint32) uint64 {
	return (uint64(a) << 32) | uint64(b)
}

func pairLeft(p uint64) uint32  { return uint32(p >> 32) }
func pairRight(p uint64) uint32 { return uint32(p) }

func loadTokenizer(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)

	var magic [4]byte
	if _, err := readFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != magicStr {
		return nil, fmt.Errorf("bad magic: got %q want %q", magic, magicStr)
	}

	version, err := readU32(r)
	if err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	vocabSize, err := readU32(r)
	if err != nil {
		return nil, fmt.Errorf("read vocab_size: %w", err)
	}
	mergeCount, err := readU32(r)
	if err != nil {
		return nil, fmt.Errorf("read merge_count: %w", err)
	}
	_, err = readU32(r) // reserved
	if err != nil {
		return nil, fmt.Errorf("read reserved: %w", err)
	}
	_ = version

	vocab := make([]Token, vocabSize)
	for i := uint32(0); i < vocabSize; i++ {
		left, err := readU32(r)
		if err != nil {
			return nil, fmt.Errorf("read vocab[%d].left: %w", i, err)
		}
		right, err := readU32(r)
		if err != nil {
			return nil, fmt.Errorf("read vocab[%d].right: %w", i, err)
		}
		vocab[i] = Token{Left: left, Right: right}
	}

	merges := make([]uint64, mergeCount)
	for i := uint32(0); i < mergeCount; i++ {
		v, err := readU64(r)
		if err != nil {
			return nil, fmt.Errorf("read merge[%d]: %w", i, err)
		}
		merges[i] = v
	}

	tok := &Tokenizer{
		Vocab:  vocab,
		Merges: merges,
		Rank:   make(map[uint64]int, len(merges)),
		PairID: make(map[uint64]uint32, len(merges)),
	}
	// merge i produced token id = 256 + i (INITIAL_VOCAB=256)
	for i, p := range merges {
		tok.Rank[p] = i
		tok.PairID[p] = uint32(256 + i)
	}

	return tok, nil
}

func readFull(r *bufio.Reader, buf []byte) (int, error) {
	n := 0
	for n < len(buf) {
		m, err := r.Read(buf[n:])
		n += m
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func readU32(r *bufio.Reader) (uint32, error) {
	var buf [4]byte
	if _, err := readFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(buf[:]), nil
}

func readU64(r *bufio.Reader) (uint64, error) {
	var buf [8]byte
	if _, err := readFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

// Encode applies BPE merges greedily in trained merge order (lowest rank first),
// exactly matching the training-time merge priority.
func (t *Tokenizer) Encode(data []byte) []uint32 {
	if len(data) == 0 {
		return nil
	}

	// Start with byte-level tokens.
	tokens := make([]uint32, len(data))
	for i, b := range data {
		tokens[i] = uint32(b)
	}

	// Doubly linked list over positions, like the C trainer, so merges are O(occurrences).
	n := len(tokens)
	const none = ^uint32(0)
	prev := make([]uint32, n)
	next := make([]uint32, n)
	for i := 0; i < n; i++ {
		if i == 0 {
			prev[i] = none
		} else {
			prev[i] = uint32(i - 1)
		}
		if i+1 >= n {
			next[i] = none
		} else {
			next[i] = uint32(i + 1)
		}
	}
	alive := make([]bool, n)
	for i := range alive {
		alive[i] = true
	}
	tok := make([]uint32, n)
	copy(tok, tokens)
	head := uint32(0)

	// Simple repeated-scan approach: find the best (lowest rank) adjacent pair,
	// merge ALL its occurrences, repeat until no mergeable pair remains.
	// This matches BPE merge order semantics (global priority by training rank).
	for {
		bestRank := -1
		var bestPair uint64
		found := false

		for i := head; i != none; i = next[i] {
			j := next[i]
			if j == none {
				break
			}
			p := makePair(tok[i], tok[j])
			if r, ok := t.Rank[p]; ok {
				if !found || r < bestRank {
					bestRank = r
					bestPair = p
					found = true
				}
			}
		}

		if !found {
			break
		}

		newTok := t.PairID[bestPair]
		a := pairLeft(bestPair)
		b := pairRight(bestPair)

		i := head
		for i != none {
			j := next[i]
			if j != none && tok[i] == a && tok[j] == b {
				nn := next[j]
				tok[i] = newTok
				next[i] = nn
				if nn != none {
					prev[nn] = i
				}
				alive[j] = false
				i = next[i]
			} else {
				i = next[i]
			}
		}
	}

	out := make([]uint32, 0, n)
	for i := head; i != none; i = next[i] {
		out = append(out, tok[i])
	}
	return out
}

// Decode expands a token id back into raw bytes by recursively expanding
// merge tokens into their left/right children down to base bytes (<256).
func (t *Tokenizer) Decode(ids []uint32) []byte {
	var out []byte
	var expand func(id uint32)
	expand = func(id uint32) {
		if id < 256 {
			out = append(out, byte(id))
			return
		}
		if int(id) >= len(t.Vocab) {
			// unknown token, skip
			return
		}
		tk := t.Vocab[id]
		expand(tk.Left)
		expand(tk.Right)
	}
	for _, id := range ids {
		expand(id)
	}
	return out
}

func main() {
	binPath := flag.String("bin", "tokenizer.bin", "path to tokenizer .bin file")
	mode := flag.String("mode", "encode", "encode | decode")
	inPath := flag.String("in", "", "input file (text for encode, or ids for decode; default stdin)")
	outPath := flag.String("out", "", "output file (default stdout)")
	idsAreText := flag.Bool("ids-text", true, "for decode: input ids are whitespace/comma separated decimal numbers (else raw uint32 LE binary)")
	flag.Parse()

	tok, err := loadTokenizer(*binPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load tokenizer:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded tokenizer: vocab=%d merges=%d\n", len(tok.Vocab), len(tok.Merges))

	var in *os.File
	if *inPath != "" {
		f, err := os.Open(*inPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "open input:", err)
			os.Exit(1)
		}
		defer f.Close()
		in = f
	} else {
		in = os.Stdin
	}

	var out *os.File
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "create output:", err)
			os.Exit(1)
		}
		defer f.Close()
		out = f
	} else {
		out = os.Stdout
	}

	switch *mode {
	case "encode":
		data, err := readAll(in)
		if err != nil {
			fmt.Fprintln(os.Stderr, "read input:", err)
			os.Exit(1)
		}
		ids := tok.Encode(data)
		w := bufio.NewWriter(out)
		defer w.Flush()
		for i, id := range ids {
			if i > 0 {
				w.WriteByte(' ')
			}
			fmt.Fprintf(w, "%d", id)
		}
		w.WriteByte('\n')
		fmt.Fprintf(os.Stderr, "encoded %d bytes -> %d tokens\n", len(data), len(ids))

	case "decode":
		var ids []uint32
		if *idsAreText {
			ids, err = readIDsText(in)
		} else {
			ids, err = readIDsBinary(in)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "read ids:", err)
			os.Exit(1)
		}
		data := tok.Decode(ids)
		if _, err := out.Write(data); err != nil {
			fmt.Fprintln(os.Stderr, "write output:", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "decoded %d tokens -> %d bytes\n", len(ids), len(data))

	case "markov":
		markovDemo()
		
	default:
		fmt.Fprintln(os.Stderr, "unknown mode:", *mode, "(use encode or decode)")
		os.Exit(1)
	}
}

func readAll(f *os.File) ([]byte, error) {
	r := bufio.NewReader(f)
	var buf []byte
	chunk := make([]byte, 65536)
	for {
		n, err := r.Read(chunk)
		if n > 0 {
			buf = append(buf, chunk[:n]...)
		}
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return buf, nil // treat any read stop as EOF for simplicity
		}
	}
	return buf, nil
}

func readIDsText(f *os.File) ([]uint32, error) {
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 64*1024*1024)
	sc.Split(bufio.ScanWords)
	var ids []uint32
	for sc.Scan() {
		s := sc.Text()
		// strip trailing commas if present
		clean := make([]byte, 0, len(s))
		for i := 0; i < len(s); i++ {
			if s[i] >= '0' && s[i] <= '9' {
				clean = append(clean, s[i])
			}
		}
		if len(clean) == 0 {
			continue
		}
		var v uint64
		for _, c := range clean {
			v = v*10 + uint64(c-'0')
		}
		ids = append(ids, uint32(v))
	}
	return ids, sc.Err()
}

func readIDsBinary(f *os.File) ([]uint32, error) {
	r := bufio.NewReader(f)
	var ids []uint32
	for {
		v, err := readU32(r)
		if err != nil {
			break
		}
		ids = append(ids, v)
	}
	return ids, nil
}