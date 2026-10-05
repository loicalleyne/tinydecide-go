package tinydecide

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// encoded is token ids plus the byte range of the text each token came from.
type encoded struct {
	ids     []uint32
	offsets [][2]int // (start, end) UTF-8 byte offsets into the encoded text
}

// WordPiece is a lowercase WordPiece tokenizer, matching WordPiece.encode in
// tinydecide.js.
type WordPiece struct {
	vocab    map[string]uint32
	unk      uint32
	prefix   string
	maxChars int
}

func isCJK(cp rune) bool {
	switch {
	case cp >= 0x4E00 && cp <= 0x9FFF,
		cp >= 0x3400 && cp <= 0x4DBF,
		cp >= 0x20000 && cp <= 0x2A6DF,
		cp >= 0x2A700 && cp <= 0x2B73F,
		cp >= 0x2B740 && cp <= 0x2B81F,
		cp >= 0x2B820 && cp <= 0x2CEAF,
		cp >= 0xF900 && cp <= 0xFAFF,
		cp >= 0x2F800 && cp <= 0x2FA1F:
		return true
	}
	return false
}

func isPunct(ch rune) bool {
	c := uint32(ch)
	if (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126) {
		return true
	}
	return unicode.In(ch, unicode.P)
}

// isJSSpace reports JavaScript's \s (WhiteSpace + LineTerminator).
func isJSSpace(ch rune) bool {
	switch ch {
	case '\t', '\n', '\u000B', '\u000C', '\r', ' ', '\u00A0', '\u1680',
		'\u2028', '\u2029', '\u202F', '\u205F', '\u3000', '\uFEFF':
		return true
	}
	return ch >= '\u2000' && ch <= '\u200A'
}

// newWordPiece builds a tokenizer. vocab[i] is the piece for id i (nil for
// unused ids). Later duplicates win, as in the JS.
func newWordPiece(vocab []*string, unk, prefix string, maxChars int) (*WordPiece, *Error) {
	m := make(map[string]uint32, len(vocab))
	for i, p := range vocab {
		if p != nil {
			m[*p] = uint32(i)
		}
	}
	u, ok := m[unk]
	if !ok {
		return nil, modelErr("unknown-token piece %q is not in the vocabulary", unk)
	}
	return &WordPiece{vocab: m, unk: u, prefix: prefix, maxChars: maxChars}, nil
}

// Unk returns the id of the unknown token.
func (w *WordPiece) Unk() uint32 { return w.unk }

type chRune struct {
	ch   rune
	a, b int
}

func lowerRune(r rune) []rune {
	return []rune(strings.ToLower(string(r)))
}

// Encode tokenizes text.
func (w *WordPiece) Encode(text string) encoded {
	chars := make([]chRune, 0, len(text)+8)
	for a, ch := range text {
		b := a + runeLen(ch)
		cp := uint32(ch)
		isCtrlFmt := (unicode.In(ch, unicode.Cc) || unicode.In(ch, unicode.Cf)) &&
			ch != '\t' && ch != '\n' && ch != '\r'
		if cp == 0 || cp == 0xFFFD || isCtrlFmt {
			continue
		}
		if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || unicode.In(ch, unicode.Zs) {
			chars = append(chars, chRune{' ', a, b})
			continue
		}
		if isCJK(ch) {
			chars = append(chars, chRune{' ', a, a}, chRune{ch, a, b}, chRune{' ', b, b})
			continue
		}
		for _, d := range norm.NFD.String(string(ch)) {
			if unicode.In(d, unicode.Mn) {
				continue
			}
			for _, l := range lowerRune(d) {
				chars = append(chars, chRune{l, a, b})
			}
		}
	}

	var words [][]chRune
	var cur []chRune
	for _, c := range chars {
		if c.ch == ' ' || isJSSpace(c.ch) {
			if len(cur) > 0 {
				words = append(words, cur)
				cur = nil
			}
			continue
		}
		if isPunct(c.ch) {
			if len(cur) > 0 {
				words = append(words, cur)
				cur = nil
			}
			words = append(words, []chRune{c})
			continue
		}
		cur = append(cur, c)
	}
	if len(cur) > 0 {
		words = append(words, cur)
	}

	var out encoded
	var sb strings.Builder
	var pieces [][3]int // id, a, b
	for _, word := range words {
		whole := [2]int{word[0].a, word[len(word)-1].b}
		if len(word) > w.maxChars {
			out.ids = append(out.ids, w.unk)
			out.offsets = append(out.offsets, whole)
			continue
		}
		pieces = pieces[:0]
		start := 0
		bad := false
		for start < len(word) {
			end := len(word)
			got := int64(-1)
			for start < end {
				sb.Reset()
				if start > 0 {
					sb.WriteString(w.prefix)
				}
				for _, c := range word[start:end] {
					sb.WriteRune(c.ch)
				}
				if id, ok := w.vocab[sb.String()]; ok {
					got = int64(id)
					break
				}
				end--
			}
			if got < 0 {
				bad = true
				break
			}
			pieces = append(pieces, [3]int{int(got), word[start].a, word[end-1].b})
			start = end
		}
		if bad {
			out.ids = append(out.ids, w.unk)
			out.offsets = append(out.offsets, whole)
		} else {
			for _, p := range pieces {
				out.ids = append(out.ids, uint32(p[0]))
				out.offsets = append(out.offsets, [2]int{p[1], p[2]})
			}
		}
	}
	return out
}

func runeLen(r rune) int {
	switch {
	case r < 0x80:
		return 1
	case r < 0x800:
		return 2
	case r < 0x10000:
		return 3
	default:
		return 4
	}
}
