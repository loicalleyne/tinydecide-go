package tinydecide

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	kState   = 0
	kQText   = 1
	kAns     = 2
	kOpt     = 3
	kLv      = 4
	headSeed = 0
)

var kBuckets = [4]int{1, 2, 4, 8}

func bucket(c int) int {
	n := 0
	for _, e := range kBuckets {
		if c > e {
			n++
		}
	}
	return n
}

// QType is one of the four answer types.
type QType int

const (
	// Choice: pick one of 2 to 32 options.
	Choice QType = iota
	// Noul: is a statement about the message true?
	Noul
	// Score: place the message on ordered levels, lowest first.
	Score
	// Span: extract a piece of the message.
	Span
)

// String returns the lowercase name of the type.
func (q QType) String() string {
	switch q {
	case Choice:
		return "choice"
	case Noul:
		return "noul"
	case Score:
		return "score"
	case Span:
		return "span"
	}
	return "unknown"
}

// MarshalJSON writes the lowercase name.
func (q QType) MarshalJSON() ([]byte, error) {
	return json.Marshal(q.String())
}

// UnmarshalJSON reads the lowercase name.
func (q *QType) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case "choice":
		*q = Choice
	case "noul":
		*q = Noul
	case "score":
		*q = Score
	case "span":
		*q = Span
	default:
		return &Error{Kind: ErrJSON, Msg: "unknown question type " + s}
	}
	return nil
}

// Question is one question, written in plain language at call time.
type Question struct {
	Kind QType  `json:"type"`
	Text string `json:"text"`
	// Options (choice) or levels, lowest first (score). Empty for noul and span.
	Options []string `json:"options,omitempty"`
}

// NewChoice builds a choice question.
func NewChoice(text string, options []string) Question {
	return Question{Kind: Choice, Text: text, Options: append([]string(nil), options...)}
}

// NewNoul builds a noul question.
func NewNoul(text string) Question {
	return Question{Kind: Noul, Text: text}
}

// NewScore builds a score question (levels lowest first).
func NewScore(text string, levels []string) Question {
	return Question{Kind: Score, Text: text, Options: append([]string(nil), levels...)}
}

// NewSpan builds a span question.
func NewSpan(text string) Question {
	return Question{Kind: Span, Text: text}
}

// Protos holds corrections for one question (see MakeProtos).
type Protos struct {
	// Vec has one prototype per option (2 for noul: false, true),
	// n_options * qvec_len values.
	Vec []float32 `json:"vec"`
	// Cnt is the number of examples behind each prototype; options with 0 add nothing.
	Cnt []int `json:"cnt"`
	// Center is the mean vector the cosines are centred on. Nil falls back to
	// the older uncentred rule, as tinydecide.js does.
	Center []float32 `json:"center"`
	// Lam is the trust weight. Nil means 1.
	Lam *float64 `json:"lam"`
}

// Answer is one answer. Which fields are set depends on Kind.
type Answer struct {
	Kind QType `json:"kind"`
	// Probs: choice/score, one probability per option.
	Probs []float64 `json:"probs"`
	// Pick: choice/score, index of the most likely option (-1 if unset).
	Pick int `json:"pick"`
	// Confidence: choice/score, 1 - normalised entropy (NaN if unset).
	Confidence float64 `json:"confidence"`
	// ScoreVal: score, expected level from 0 (first) to 1 (last) (NaN if unset).
	ScoreVal float64 `json:"score"`
	// P: noul, probability the statement is true (NaN if unset).
	P float64 `json:"p"`
	// Qvec: choice/score/noul, vector to store with a correction.
	Qvec []float32 `json:"qvec"`
	// Proj is true when Qvec is the prototype projection.
	Proj bool `json:"proj"`
	// Z0: choice/score/noul, zero-shot logits over temperature (noul: [0, z]).
	Z0 []float64 `json:"z0"`
	// PPresent: span, probability the message contains what was asked for.
	PPresent float64 `json:"p_present"`
	// PSpan: span, probability of the chosen start and end tokens.
	PSpan float64 `json:"p_span"`
	// Tok: span, first and last state token of the span (0-based). nil if unset.
	Tok *[2]int `json:"tok"`
	// Text: span, the extracted text, trimmed.
	Text string `json:"text"`
	// Char: span, (start, end) UTF-8 byte offsets into the state string. nil if unset.
	Char *[2]int `json:"char"`
}

func emptyAnswer(kind QType) Answer {
	return Answer{
		Kind:       kind,
		Pick:       -1,
		Confidence: math.NaN(),
		ScoreVal:   math.NaN(),
		P:          math.NaN(),
		PPresent:   math.NaN(),
		PSpan:      math.NaN(),
	}
}

// Tokens is the token count breakdown.
type Tokens struct {
	State     int `json:"state"`
	Questions int `json:"questions"`
	Total     int `json:"total"`
}

// Response is the result of answering.
type Response struct {
	Answers []Answer `json:"answers"`
	IDs     []uint32 `json:"ids"`
	Tokens  Tokens   `json:"tokens"`
	// Truncated reports the message was longer than the model reads and was cut.
	Truncated bool    `json:"truncated"`
	MS        float64 `json:"ms"`
}

// ------------------------------------------------------------------ math

func dot(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	a, b = a[:n], b[:n]
	var acc [16]float32
	i := 0
	for ; i+16 <= n; i += 16 {
		for j := 0; j < 16; j++ {
			acc[j] += a[i+j] * b[i+j]
		}
	}
	var s float32
	for _, v := range acc {
		s += v
	}
	for ; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

func dot64(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float64
	for i := 0; i < n; i++ {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func dotd(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float64
	for i := 0; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

func norm64(a []float32) []float64 {
	n := math.Sqrt(dot64(a, a))
	if n == 0 || math.IsNaN(n) {
		n = 1e-12
	}
	out := make([]float64, len(a))
	for i, z := range a {
		out[i] = float64(z) / n
	}
	return out
}

type lin struct {
	w    []float32
	b    []float32 // nil for no bias
	rows int
	cols int
}

// apply computes y[t] = W x[t] + b for t rows of x.
func (l *lin) apply(x []float32, t int) []float32 {
	din, dout := l.cols, l.rows
	y := make([]float32, t*dout)
	for j := 0; j < dout; j++ {
		row := l.w[j*din : (j+1)*din]
		var b float32
		if l.b != nil {
			b = l.b[j]
		}
		for tt := 0; tt < t; tt++ {
			y[tt*dout+j] = b + dot(row, x[tt*din:(tt+1)*din])
		}
	}
	return y
}

func (l *lin) mv(v []float32) []float32 {
	return l.apply(v, 1)
}

func layernorm(x []float32, t, d int, w []float32, b []float32, eps float64) []float32 {
	y := make([]float32, t*d)
	for tt := 0; tt < t; tt++ {
		r := x[tt*d : (tt+1)*d]
		var m float64
		for _, z := range r {
			m += float64(z)
		}
		m /= float64(d)
		var v float64
		for _, z := range r {
			diff := float64(z) - m
			v += diff * diff
		}
		v /= float64(d)
		inv := 1.0 / math.Sqrt(v+eps)
		for i := 0; i < d; i++ {
			var bi float64
			if b != nil {
				bi = float64(b[i])
			}
			y[tt*d+i] = float32((float64(r[i])-m)*inv*float64(w[i]) + bi)
		}
	}
	return y
}

// erf is the erf used by tinydecide.js: a Taylor series below 0.5, the
// Numerical Recipes erfc fit above.
func erf(x float64) float64 {
	var s float64
	switch {
	case x > 0:
		s = 1
	case x < 0:
		s = -1
	default:
		return x
	}
	a := math.Abs(x)
	if a < 0.5 {
		sum, term, a2 := a, a, a*a
		for n := 1; n < 12; n++ {
			term *= -a2 / float64(n)
			sum += term / float64(2*n+1)
		}
		return s * 2.0 / math.Sqrt(math.Pi) * sum
	}
	t := 1.0 / (1.0 + 0.5*a)
	y := t * math.Exp(-a*a-1.26551223+
		t*(1.00002368+
			t*(0.37409196+
				t*(0.09678418+
					t*(-0.18628806+
						t*(0.27886807+
							t*(-1.13520398+
								t*(1.48851587+
									t*(-0.82215223+
										t*0.17087277)))))))))
	return s * (1.0 - y)
}

func gelu(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1.0 + erf(xf/math.Sqrt2)))
}

// lsm is the log-softmax of a / tt.
func lsm(a []float64, tt float64) []float64 {
	m := math.Inf(-1)
	for _, z := range a {
		if z > m {
			m = z
		}
	}
	var sum float64
	for _, z := range a {
		sum += math.Exp((z - m) / tt)
	}
	l := math.Log(sum)
	out := make([]float64, len(a))
	for i, z := range a {
		out[i] = (z-m)/tt - l
	}
	return out
}

func jsTrim(s string) string {
	return strings.TrimFunc(s, isJSSpace)
}

// ------------------------------------------------------------------ model

type block struct {
	q, k, v, o lin
	ln1w, ln1b []float32
	fc, fc2    lin
	ln2w, ln2b []float32
}

type wordTable struct {
	full    []float32 // set for full table
	a, b    []float32 // set for low-rank
	r       int
	lowRank bool
}

type heads struct {
	normW, normB []float32
	scale        []float32
	a            [2]lin
	o            [2]lin
	p            *lin
	noulW        []float32
	noulB        float32
	noulQ        lin
	sq, sk       lin
	snullW       []float32
	snullB       float32
	eq, es, ek   lin
}

type specials struct {
	state uint32
	types [4]uint32
	sep   uint32
	o     uint32
	lv    uint32
	ans   uint32
}

type qEnc struct {
	kind QType
	ans  int
	opt  []int
}

type enc struct {
	ids       []uint32
	pos       []int
	blk       []int
	kind      []int
	stIdx     []int
	stOff     [][2]int
	qs        []qEnc
	truncated bool
}

// TinyDecide is a loaded TinyDecide model.
type TinyDecide struct {
	meta   Meta
	tok    *WordPiece
	sp     specials
	word   wordTable
	pos    []float32
	type0  []float32
	elnW   []float32
	elnB   []float32
	proj   lin
	blocks []block
	heads  heads
}

func takeVec(w *weights, n string, length int) ([]float32, *Error) {
	t, err := w.take(n, []*int{ip(length)})
	if err != nil {
		return nil, err
	}
	return t.data, nil
}

func takeLin(w *weights, n string, rows *int, cols int, bias string) (lin, *Error) {
	t, err := w.take(n, []*int{rows, ip(cols)})
	if err != nil {
		return lin{}, err
	}
	r := t.shape[0]
	var b []float32
	if bias != "" {
		b, err = takeVec(w, bias, r)
		if err != nil {
			return lin{}, err
		}
	}
	return lin{w: t.data, b: b, rows: r, cols: cols}, nil
}

// Load reads meta.json and model.bin from a directory.
//
// When the binary is built with the "embed" build tag, an embedded model is
// baked in at compile time. In that case the files in dir take precedence when
// both meta.json and model.bin are present there; otherwise the embedded model
// is used. Without the "embed" tag, dir is always required.
func Load(dir string) (*TinyDecide, error) {
	embMeta, embBin, haveEmbed := embeddedModel()

	metaBytes, metaErr := os.ReadFile(filepath.Join(dir, "meta.json"))
	bin, binErr := os.ReadFile(filepath.Join(dir, "model.bin"))

	if metaErr != nil || binErr != nil {
		if haveEmbed {
			return FromBytes(embMeta, embBin)
		}
		if metaErr != nil {
			return nil, ioErr(metaErr)
		}
		return nil, ioErr(binErr)
	}
	return FromBytes(metaBytes, bin)
}

// FromBytes builds a model from the text of meta.json and the bytes of model.bin.
func FromBytes(metaJSON []byte, bin []byte) (*TinyDecide, error) {
	var meta Meta
	if err := json.Unmarshal(metaJSON, &meta); err != nil {
		return nil, jsonErr(err)
	}
	c := meta.Cfg
	if c.Arch != "electra" {
		return nil, modelErr("unsupported architecture %s", c.Arch)
	}
	if meta.Tokenizer.Kind != "wordpiece" {
		return nil, modelErr("this engine build ships the WordPiece tokenizer only")
	}
	if c.Heads == 0 || c.D%c.Heads != 0 {
		return nil, modelErr("cfg.d must be a multiple of cfg.heads")
	}
	if len(meta.Temp) < 4 || len(meta.Beta) < 5 {
		return nil, modelErr("meta.temp needs 4 values and meta.beta 5")
	}
	tk := meta.Tokenizer
	tok, mErr := newWordPiece(tk.Vocab, tk.Unk, tk.Prefix, tk.MaxChars)
	if mErr != nil {
		return nil, mErr
	}
	vocabLen := len(tk.Vocab)
	meta.Tokenizer.Vocab = nil

	w, mErr := newWeights(&meta, bin)
	if mErr != nil {
		return nil, mErr
	}
	d, e, dh := c.D, c.EmbDim, c.DHHead

	var word wordTable
	var vocabRows int
	if w.has("m.word.a.weight") {
		a, err := w.take("m.word.a.weight", []*int{nil, nil})
		if err != nil {
			return nil, err
		}
		r := a.shape[1]
		b, err := w.take("m.word.b.weight", []*int{ip(e), ip(r)})
		if err != nil {
			return nil, err
		}
		word = wordTable{a: a.data, b: b.data, r: r, lowRank: true}
		vocabRows = a.shape[0]
	} else {
		t, err := w.take("m.word.weight", []*int{nil, ip(e)})
		if err != nil {
			return nil, err
		}
		word = wordTable{full: t.data}
		vocabRows = t.shape[0]
	}

	posemb, err := w.take("m.posemb.weight", []*int{nil, ip(e)})
	if err != nil {
		return nil, err
	}
	posRows := posemb.shape[0]
	f := meta.Format
	if f.TSMax == 0 || f.TSMax > posRows || f.PQ+f.QMax > posRows {
		return nil, modelErr("format.ts_max / p_q / q_max do not fit the position table")
	}
	spc := func(n string) (uint32, *Error) {
		id, ok := meta.Specials[n]
		if !ok {
			return 0, modelErr("missing special token %s", n)
		}
		if int(id) >= vocabRows {
			return 0, modelErr("special token %s is outside the embedding table", n)
		}
		return id, nil
	}
	var sp specials
	for _, pair := range []struct {
		dst *uint32
		n   string
	}{
		{&sp.state, "<|state|>"}, {&sp.sep, "<|sep|>"}, {&sp.o, "<|o|>"},
		{&sp.lv, "<|lv|>"}, {&sp.ans, "<|ans|>"},
	} {
		id, e := spc(pair.n)
		if e != nil {
			return nil, e
		}
		*pair.dst = id
	}
	for i, n := range []string{"<|choice|>", "<|noul|>", "<|score|>", "<|span|>"} {
		id, e := spc(n)
		if e != nil {
			return nil, e
		}
		sp.types[i] = id
	}
	if int(tok.Unk()) >= vocabRows {
		return nil, modelErr("the unknown token is outside the embedding table")
	}
	if vocabLen > vocabRows {
		return nil, modelErr("the vocabulary is larger than the embedding table")
	}

	type0, err := takeVec(w, "m.type0", e)
	if err != nil {
		return nil, err
	}
	elnW, err := takeVec(w, "m.eln.weight", e)
	if err != nil {
		return nil, err
	}
	elnB, err := takeVec(w, "m.eln.bias", e)
	if err != nil {
		return nil, err
	}
	proj, err := takeLin(w, "m.proj.weight", ip(d), e, "m.proj.bias")
	if err != nil {
		return nil, err
	}
	blocks := make([]block, 0, c.Layers)
	for l := 0; l < c.Layers; l++ {
		p := "m.blocks." + itoa(l) + "."
		nm := func(s string) string { return p + s }
		var bl block
		var e2 *Error
		if bl.q, e2 = takeLin(w, nm("q.weight"), ip(d), d, nm("q.bias")); e2 != nil {
			return nil, e2
		}
		if bl.k, e2 = takeLin(w, nm("k.weight"), ip(d), d, nm("k.bias")); e2 != nil {
			return nil, e2
		}
		if bl.v, e2 = takeLin(w, nm("v.weight"), ip(d), d, nm("v.bias")); e2 != nil {
			return nil, e2
		}
		if bl.o, e2 = takeLin(w, nm("o.weight"), ip(d), d, nm("o.bias")); e2 != nil {
			return nil, e2
		}
		if bl.ln1w, e2 = takeVec(w, nm("ln1.weight"), d); e2 != nil {
			return nil, e2
		}
		if bl.ln1b, e2 = takeVec(w, nm("ln1.bias"), d); e2 != nil {
			return nil, e2
		}
		if bl.fc, e2 = takeLin(w, nm("fc.weight"), nil, d, nm("fc.bias")); e2 != nil {
			return nil, e2
		}
		if bl.fc2, e2 = takeLin(w, nm("fc2.weight"), ip(d), bl.fc.rows, nm("fc2.bias")); e2 != nil {
			return nil, e2
		}
		if bl.ln2w, e2 = takeVec(w, nm("ln2.weight"), d); e2 != nil {
			return nil, e2
		}
		if bl.ln2b, e2 = takeVec(w, nm("ln2.bias"), d); e2 != nil {
			return nil, e2
		}
		blocks = append(blocks, bl)
	}
	hl := func(n string) (lin, *Error) { return takeLin(w, n, ip(dh), d, "") }
	var hd heads
	if w.has("h.P.weight") {
		pl, e2 := hl("h.P.weight")
		if e2 != nil {
			return nil, e2
		}
		hd.p = &pl
	}
	var e2 *Error
	if hd.normW, e2 = takeVec(w, "h.norm.weight", d); e2 != nil {
		return nil, e2
	}
	if hd.normB, e2 = takeVec(w, "h.norm.bias", d); e2 != nil {
		return nil, e2
	}
	if hd.scale, e2 = takeVec(w, "h.scale", 2); e2 != nil {
		return nil, e2
	}
	if hd.a[0], e2 = hl("h.A.0.weight"); e2 != nil {
		return nil, e2
	}
	if hd.a[1], e2 = hl("h.A.1.weight"); e2 != nil {
		return nil, e2
	}
	if hd.o[0], e2 = hl("h.O.0.weight"); e2 != nil {
		return nil, e2
	}
	if hd.o[1], e2 = hl("h.O.1.weight"); e2 != nil {
		return nil, e2
	}
	noul, e2 := w.take("h.noul.weight", []*int{ip(1), ip(d)})
	if e2 != nil {
		return nil, e2
	}
	hd.noulW = noul.data
	noulB, e2 := takeVec(w, "h.noul.bias", 1)
	if e2 != nil {
		return nil, e2
	}
	hd.noulB = noulB[0]
	if hd.noulQ, e2 = hl("h.noul_q.weight"); e2 != nil {
		return nil, e2
	}
	if hd.sq, e2 = hl("h.sq.weight"); e2 != nil {
		return nil, e2
	}
	if hd.sk, e2 = hl("h.sk.weight"); e2 != nil {
		return nil, e2
	}
	snull, e2 := w.take("h.snull.weight", []*int{ip(1), ip(d)})
	if e2 != nil {
		return nil, e2
	}
	hd.snullW = snull.data
	snullB, e2 := takeVec(w, "h.snull.bias", 1)
	if e2 != nil {
		return nil, e2
	}
	hd.snullB = snullB[0]
	if hd.eq, e2 = hl("h.eq.weight"); e2 != nil {
		return nil, e2
	}
	if hd.es, e2 = hl("h.es.weight"); e2 != nil {
		return nil, e2
	}
	if hd.ek, e2 = hl("h.ek.weight"); e2 != nil {
		return nil, e2
	}

	return &TinyDecide{
		meta: meta, tok: tok, sp: sp, word: word, pos: posemb.data,
		type0: type0, elnW: elnW, elnB: elnB, proj: proj, blocks: blocks, heads: hd,
	}, nil
}

// Meta returns the model's metadata; the vocabulary is dropped after loading.
func (m *TinyDecide) Meta() *Meta { return &m.meta }

// Beta returns meta.beta, the prototype weights by example count.
func (m *TinyDecide) Beta() []float64 { return m.meta.Beta }

// Tokenizer returns the model's tokenizer.
func (m *TinyDecide) Tokenizer() *WordPiece { return m.tok }

// Answer answers every question about state in one encoder pass.
func (m *TinyDecide) Answer(state string, questions []Question) (*Response, error) {
	return m.AnswerWith(state, questions, nil)
}

// AnswerWith is like Answer, with corrections: protos[i] belongs to
// questions[i] (missing or nil entries mean none).
func (m *TinyDecide) AnswerWith(state string, questions []Question, protos []*Protos) (*Response, error) {
	t0 := time.Now()
	e, err := m.encodeRequest(state, questions)
	if err != nil {
		return nil, err
	}
	dh := m.meta.Cfg.DHHead
	for qi, q := range questions {
		p := protoAt(protos, qi)
		if p == nil {
			continue
		}
		k := len(q.Options)
		if q.Kind == Noul {
			k = 2
		}
		for i := 0; i < k; i++ {
			if cntAt(p.Cnt, i) > 0 && len(p.Vec) < (i+1)*dh {
				return nil, requestErr("Corrections for question %d have too few vector values.", qi+1)
			}
		}
		if p.Center != nil && len(p.Center) < dh {
			return nil, requestErr("Corrections for question %d have a short center.", qi+1)
		}
	}
	x := m.hidden(e)
	answers := m.computeHeads(e, x, state, protos)
	t := len(e.ids)
	s := len(e.stIdx)
	return &Response{
		Answers:   answers,
		Tokens:    Tokens{State: s + 1, Questions: t - s - 1, Total: t},
		Truncated: e.truncated,
		MS:        float64(time.Since(t0).Nanoseconds()) / 1e6,
		IDs:       e.ids,
	}, nil
}

func protoAt(protos []*Protos, i int) *Protos {
	if i < len(protos) {
		return protos[i]
	}
	return nil
}

func cntAt(cnt []int, i int) int {
	if i < len(cnt) {
		return cnt[i]
	}
	return 0
}

func (m *TinyDecide) encodeRequest(state string, questions []Question) (*enc, *Error) {
	f := m.meta.Format
	st := m.tok.Encode(state)
	n := len(st.ids)
	if f.TSMax-1 < n {
		n = f.TSMax - 1
	}
	ids := make([]uint32, 0, n+1+len(questions)*16)
	ids = append(ids, m.sp.state)
	ids = append(ids, st.ids[:n]...)
	pos := make([]int, len(ids))
	for i := range pos {
		pos[i] = i
	}
	blk := make([]int, len(ids))
	kind := make([]int, len(ids))
	for i := range kind {
		kind[i] = kState
	}
	stIdx := make([]int, 0, len(ids)-1)
	for i := 1; i < len(ids); i++ {
		stIdx = append(stIdx, i)
	}
	stOff := append([][2]int(nil), st.offsets[:n]...)
	qs := make([]qEnc, 0, len(questions))
	for qi, q := range questions {
		k := qi + 1
		kMax := -1
		if f.KMax != nil {
			kMax = *f.KMax
		}
		nOpt := len(q.Options)
		if (q.Kind == Choice || q.Kind == Score) && (nOpt < 2 || (kMax >= 0 && nOpt > kMax)) {
			max := "any"
			if f.KMax != nil {
				max = itoa(*f.KMax)
			}
			return nil, requestErr("Question %d needs 2 to %s options, not %d.", k, max, nOpt)
		}
		var ti int
		switch q.Kind {
		case Choice:
			ti = 0
		case Noul:
			ti = 1
		case Score:
			ti = 2
		case Span:
			ti = 3
		}
		bIDs := []uint32{m.sp.types[ti]}
		bKind := []int{kQText}
		t := m.tok.Encode(q.Text).ids
		for range t {
			bKind = append(bKind, kQText)
		}
		bIDs = append(bIDs, t...)
		var optLocal []int
		if len(q.Options) > 0 {
			bIDs = append(bIDs, m.sp.sep)
			bKind = append(bKind, kQText)
			mk, mkind := m.sp.o, kOpt
			if q.Kind != Choice {
				mk, mkind = m.sp.lv, kLv
			}
			for _, o := range q.Options {
				oi := m.tok.Encode(o).ids
				for range oi {
					bKind = append(bKind, kQText)
				}
				bIDs = append(bIDs, oi...)
				optLocal = append(optLocal, len(bIDs))
				bIDs = append(bIDs, mk)
				bKind = append(bKind, mkind)
			}
		}
		ansLocal := len(bIDs)
		bIDs = append(bIDs, m.sp.ans)
		bKind = append(bKind, kAns)
		if len(bIDs) > f.QMax {
			return nil, requestErr("Question %d is too long (%d tokens, max %d).", k, len(bIDs), f.QMax)
		}
		base := len(ids)
		for j := range bIDs {
			ids = append(ids, bIDs[j])
			pos = append(pos, f.PQ+j)
			blk = append(blk, k)
			kind = append(kind, bKind[j])
		}
		opt := make([]int, len(optLocal))
		for i, j := range optLocal {
			opt[i] = base + j
		}
		qs = append(qs, qEnc{kind: q.Kind, ans: base + ansLocal, opt: opt})
	}
	return &enc{
		ids: ids, pos: pos, blk: blk, kind: kind, stIdx: stIdx,
		stOff: stOff, qs: qs, truncated: len(st.ids) > n,
	}, nil
}

func (m *TinyDecide) continues(kind []int) []bool {
	out := make([]bool, len(kind))
	fusion := ""
	if m.meta.Cfg.Fusion != nil {
		fusion = *m.meta.Cfg.Fusion
	}
	switch fusion {
	case "all":
		for i := range out {
			out[i] = true
		}
	case "markers":
		for i, k := range kind {
			out[i] = k == kState || k == kAns || k == kOpt || k == kLv
		}
	default:
		for i, k := range kind {
			out[i] = k == kState || k == kAns
		}
	}
	return out
}

func allowedSets(e *enc, cont []bool, fusionLayer bool) [][]int {
	t := len(e.ids)
	nb := 0
	for _, b := range e.blk {
		if b+1 > nb {
			nb = b + 1
		}
	}
	byBlk := make([][]int, nb)
	for j := 0; j < t; j++ {
		if fusionLayer && !cont[j] {
			continue
		}
		byBlk[e.blk[j]] = append(byBlk[e.blk[j]], j)
	}
	sets := make([][]int, 0, t)
	for i := 0; i < t; i++ {
		if fusionLayer && !cont[i] {
			sets = append(sets, []int{i})
			continue
		}
		var own []int
		if len(byBlk[e.blk[i]]) == 0 {
			own = []int{i}
		} else {
			own = byBlk[e.blk[i]]
		}
		if !fusionLayer || e.blk[i] == 0 {
			sets = append(sets, own)
		} else {
			s := append([]int(nil), byBlk[0]...)
			s = append(s, own...)
			sets = append(sets, s)
		}
	}
	return sets
}

func (m *TinyDecide) attend(q, k, v []float32, t int, sets [][]int) []float32 {
	d := m.meta.Cfg.D
	hds := m.meta.Cfg.Heads
	dh := d / hds
	scale := float32(1.0 / math.Sqrt(float64(dh)))
	out := make([]float32, t*d)
	sc := make([]float64, t)
	for i := 0; i < t && i < len(sets); i++ {
		keys := sets[i]
		for h := 0; h < hds; h++ {
			qo := i*d + h*dh
			qr := q[qo : qo+dh]
			mx := math.Inf(-1)
			for a, kj := range keys {
				ko := kj*d + h*dh
				s := float64(dot(qr, k[ko:ko+dh]) * scale)
				sc[a] = s
				if s > mx {
					mx = s
				}
			}
			var z float64
			for a := 0; a < len(keys); a++ {
				sc[a] = math.Exp(sc[a] - mx)
				z += sc[a]
			}
			o := out[qo : qo+dh]
			for a, kj := range keys {
				p := float32(sc[a] / z)
				vo := kj*d + h*dh
				vr := v[vo : vo+dh]
				for c := range o {
					o[c] += p * vr[c]
				}
			}
		}
	}
	return out
}

func (m *TinyDecide) hidden(e *enc) []float32 {
	c := m.meta.Cfg
	t, d, ed := len(e.ids), c.D, c.EmbDim
	lA := 0
	if c.LA != nil {
		lA = *c.LA
	}
	cont := m.continues(e.kind)
	setsF := allowedSets(e, cont, true)
	var setsC [][]int
	if lA > 0 {
		setsC = allowedSets(e, cont, false)
	}

	raw := make([]float32, t*ed)
	for tt := 0; tt < t; tt++ {
		id := int(e.ids[tt])
		po := e.pos[tt] * ed
		for ch := 0; ch < ed; ch++ {
			var wv float64
			if m.word.lowRank {
				r := m.word.r
				wv = dot64(m.word.a[id*r:(id+1)*r], m.word.b[ch*r:(ch+1)*r])
			} else {
				wv = float64(m.word.full[id*ed+ch])
			}
			raw[tt*ed+ch] = float32(wv + float64(m.pos[po+ch]) + float64(m.type0[ch]))
		}
	}
	nrm := layernorm(raw, t, ed, m.elnW, m.elnB, c.LNEps)
	x := m.proj.apply(nrm, t)
	for l := range m.blocks {
		b := &m.blocks[l]
		sets := setsF
		if l < lA {
			sets = setsC
		}
		q := b.q.apply(x, t)
		k := b.k.apply(x, t)
		v := b.v.apply(x, t)
		y := m.attend(q, k, v, t, sets)
		o := b.o.apply(y, t)
		for i := range o {
			o[i] += x[i]
		}
		x = layernorm(o, t, d, b.ln1w, b.ln1b, c.LNEps)
		fv := b.fc.apply(x, t)
		for i := range fv {
			fv[i] = gelu(fv[i])
		}
		f2 := b.fc2.apply(fv, t)
		for i := range f2 {
			f2[i] += x[i]
		}
		x = layernorm(f2, t, d, b.ln2w, b.ln2b, c.LNEps)
	}
	return x
}

func (m *TinyDecide) computeHeads(e *enc, x []float32, state string, protos []*Protos) []Answer {
	c := m.meta.Cfg
	d, dh, t := c.D, c.DHHead, len(e.ids)
	h := &m.heads
	hq := layernorm(x, t, d, h.normW, h.normB, 1e-5)
	row := func(i int) []float32 { return hq[i*d : (i+1)*d] }
	temp := m.meta.Temp
	beta := m.meta.Beta
	sqrtDh := math.Sqrt(float64(dh))
	hasP := h.p != nil
	out := make([]Answer, 0, len(e.qs))

	for qi, q := range e.qs {
		hAns := row(q.ans)
		pr := protoAt(protos, qi)
		var pv []float32
		if h.p != nil {
			pv = h.p.mv(hAns)
		}
		cnt := func(i int) int {
			if pr == nil {
				return 0
			}
			return cntAt(pr.Cnt, i)
		}
		pvec := func(i int) []float32 { return pr.Vec[i*dh : (i+1)*dh] }
		// centred-cosine prototype term; ok=false means fall back to the older rule.
		protoTerm := func(i int) (float64, bool) {
			if pr == nil {
				return 0, true
			}
			if cnt(i) <= 0 {
				return 0, true
			}
			if pv != nil && pr.Center != nil {
				v := pvec(i)
				a := make([]float32, dh)
				b := make([]float32, dh)
				for j := 0; j < dh; j++ {
					a[j] = pv[j] - pr.Center[j]
					b[j] = v[j] - pr.Center[j]
				}
				na := math.Sqrt(dot64(a, a))
				nb := math.Sqrt(dot64(b, b))
				if na == 0 {
					na = 1e-12
				}
				if nb == 0 {
					nb = 1e-12
				}
				return beta[bucket(cnt(i))] * dot64(a, b) / (na * nb), true
			}
			return 0, false
		}
		lam := 1.0
		if pr != nil && pr.Lam != nil {
			lam = *pr.Lam
		}

		switch q.kind {
		case Choice, Score:
			sel := 0
			if q.kind == Score {
				sel = 1
			}
			ti := 0
			if sel == 1 {
				ti = 2
			}
			tt := temp[ti]
			qa := h.a[sel].mv(hAns)
			s := math.Exp(float64(h.scale[sel]))
			qv := norm64(qa)
			z0 := make([]float64, 0, len(q.opt))
			logits := make([]float64, len(q.opt))
			for i, oi := range q.opt {
				ov := h.o[sel].mv(row(oi))
				z := s * dot64(qa, ov) / sqrtDh
				z0 = append(z0, z/tt)
				if tv, ok := protoTerm(i); ok {
					z += lam * tv
				} else if cnt(i) > 0 {
					z += beta[bucket(cnt(i))] * dotd(qv, norm64(pvec(i)))
				}
				logits[i] = z
			}
			mmax := math.Inf(-1)
			for _, z := range logits {
				if z > mmax {
					mmax = z
				}
			}
			ex := make([]float64, len(logits))
			var zs float64
			for i, z := range logits {
				ex[i] = math.Exp((z - mmax) / tt)
				zs += ex[i]
			}
			probs := make([]float64, len(ex))
			for i, ev := range ex {
				probs[i] = ev / zs
			}
			k := len(probs)
			var hh float64
			for _, p := range probs {
				if p > 0 {
					hh += p * math.Log(p)
				}
			}
			hh = -hh
			pick := 0
			for i, p := range probs {
				if p > probs[pick] {
					pick = i
				}
			}
			a := emptyAnswer(q.kind)
			if k > 1 {
				a.Confidence = 1.0 - hh/math.Log(float64(k))
			} else {
				a.Confidence = 1.0
			}
			if q.kind == Score {
				if k > 1 {
					var sv float64
					for i, p := range probs {
						sv += p * float64(i) / float64(k-1)
					}
					a.ScoreVal = sv
				} else {
					a.ScoreVal = 0.0
				}
			}
			a.Pick = pick
			a.Probs = probs
			if pv != nil {
				a.Qvec = append([]float32(nil), pv...)
			} else {
				a.Qvec = f64ToF32(qv)
			}
			a.Proj = hasP
			a.Z0 = z0
			out = append(out, a)
		case Noul:
			z := dot64(h.noulW, hAns) + float64(h.noulB)
			z0 := z / temp[1]
			nv := norm64(h.noulQ.mv(hAns))
			if pr != nil {
				term := make([]float64, 2)
				for i := 0; i < 2; i++ {
					if tv, ok := protoTerm(i); ok {
						term[i] = lam * tv
					} else if cnt(i) > 0 {
						term[i] = beta[bucket(cnt(i))] * dotd(nv, norm64(pvec(i)))
					}
				}
				z += term[1] - term[0]
			}
			a := emptyAnswer(Noul)
			a.P = 1.0 / (1.0 + math.Exp(-z/temp[1]))
			if pv != nil {
				a.Qvec = append([]float32(nil), pv...)
			} else {
				a.Qvec = f64ToF32(nv)
			}
			a.Proj = hasP
			a.Z0 = []float64{0.0, z0}
			out = append(out, a)
		case Span:
			s := len(e.stIdx)
			tt := temp[3]
			qsv := h.sq.mv(hAns)
			start := make([]float64, 0, s+1)
			start = append(start, dot64(h.snullW, hAns)+float64(h.snullB))
			for _, i := range e.stIdx {
				start = append(start, dot64(qsv, h.sk.mv(row(i)))/sqrtDh)
			}
			ls := lsm(start, tt)
			best := 0
			for ti := 1; ti < s; ti++ {
				if ls[1+ti] > ls[1+best] {
					best = ti
				}
			}
			eq := h.eq.mv(hAns)
			bestIdx := 0
			if best < len(e.stIdx) {
				bestIdx = e.stIdx[best]
			}
			es := h.es.mv(row(bestIdx))
			qe := make([]float32, len(eq))
			for i := range eq {
				qe[i] = eq[i] + es[i]
			}
			spanMax := m.meta.Format.SpanMax
			ev := make([]float64, len(e.stIdx))
			for ti, i := range e.stIdx {
				if ti >= best && ti < best+spanMax {
					ev[ti] = dot64(qe, h.ek.mv(row(i))) / sqrtDh
				} else {
					ev[ti] = -1e4
				}
			}
			var le []float64
			if s > 0 {
				le = lsm(ev, tt)
			}
			bestE := best
			for ti := 0; ti < s; ti++ {
				if le[ti] > le[bestE] {
					bestE = ti
				}
			}
			a := emptyAnswer(Span)
			a.PPresent = 1.0 - math.Exp(ls[0])
			a.Tok = &[2]int{best, bestE}
			if s > 0 {
				a.PSpan = math.Exp(ls[1+best] + le[bestE])
			} else {
				a.PSpan = 0.0
			}
			text := ""
			if s > 0 && best < len(e.stOff) && bestE < len(e.stOff) {
				a0 := e.stOff[best][0]
				b1 := e.stOff[bestE][1]
				a.Char = &[2]int{a0, b1}
				if a0 >= 0 && b1 <= len(state) && a0 <= b1 {
					text = jsTrim(state[a0:b1])
				}
			}
			a.Text = text
			out = append(out, a)
		}
	}
	return out
}

func f64ToF32(a []float64) []float32 {
	out := make([]float32, len(a))
	for i, v := range a {
		out[i] = float32(v)
	}
	return out
}
