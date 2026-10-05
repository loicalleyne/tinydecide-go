package tinydecide

import (
	"encoding/binary"
	"math"
)

// Meta is the parsed meta.json.
type Meta struct {
	Name      string            `json:"name"`
	Cfg       Cfg               `json:"cfg"`
	Tensors   []TensorMeta      `json:"tensors"`
	Specials  map[string]uint32 `json:"specials"`
	Format    Format            `json:"format"`
	Temp      []float64         `json:"temp"`
	Beta      []float64         `json:"beta"`
	Tokenizer TokMeta           `json:"tokenizer"`
}

// Cfg is the model architecture configuration.
type Cfg struct {
	Arch   string  `json:"arch"`
	D      int     `json:"d"`
	Layers int     `json:"layers"`
	LA     *int    `json:"L_a"`
	Fusion *string `json:"fusion"`
	DHHead int     `json:"dh_head"`
	Heads  int     `json:"heads"`
	LNEps  float64 `json:"ln_eps"`
	EmbDim int     `json:"emb_dim"`
}

// Format describes the request token layout limits.
type Format struct {
	TSMax   int  `json:"ts_max"`
	PQ      int  `json:"p_q"`
	QMax    int  `json:"q_max"`
	KMax    *int `json:"k_max"`
	SpanMax int  `json:"span_max"`
}

// TokMeta is the tokenizer metadata.
type TokMeta struct {
	Kind     string    `json:"kind"`
	Vocab    []*string `json:"vocab"`
	Unk      string    `json:"unk"`
	Prefix   string    `json:"prefix"`
	MaxChars int       `json:"max_chars"`
}

// TensorMeta describes one tensor in model.bin.
type TensorMeta struct {
	Name        string `json:"name"`
	Shape       []int  `json:"shape"`
	Dtype       string `json:"dtype"`
	Offset      int    `json:"offset"`
	ScaleOffset *int   `json:"scale_offset"`
}

// tensor is a dequantised tensor.
type tensor struct {
	data  []float32
	shape []int
}

func sliceBytes(buf []byte, off, length int, name string) ([]byte, *Error) {
	if off < 0 || length < 0 || off+length > len(buf) {
		return nil, modelErr("tensor %s runs past the end of model.bin", name)
	}
	return buf[off : off+length], nil
}

func f32At(b []byte, i int) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(b[4*i : 4*i+4]))
}

func dequant(e *TensorMeta, buf []byte) (*tensor, *Error) {
	name := e.Name
	n := 1
	for _, s := range e.Shape {
		n *= s
	}
	if n < 0 || n/2 > len(buf) {
		return nil, modelErr("tensor %s is larger than model.bin", name)
	}
	needScale := func() (int, *Error) {
		if e.ScaleOffset == nil {
			return 0, modelErr("tensor %s has no scale_offset", name)
		}
		return *e.ScaleOffset, nil
	}
	var data []float32
	switch e.Dtype {
	case "f32":
		b, err := sliceBytes(buf, e.Offset, 4*n, name)
		if err != nil {
			return nil, err
		}
		data = make([]float32, n)
		for i := 0; i < n; i++ {
			data[i] = f32At(b, i)
		}
	case "q4":
		if len(e.Shape) != 2 || e.Shape[1]%32 != 0 {
			return nil, modelErr("q4 tensor %s must be 2-D with columns a multiple of 32", name)
		}
		rows, cols := e.Shape[0], e.Shape[1]
		nb := cols / 32
		nib, err := sliceBytes(buf, e.Offset, rows*nb*16, name)
		if err != nil {
			return nil, err
		}
		so, err := needScale()
		if err != nil {
			return nil, err
		}
		sc, err := sliceBytes(buf, so, rows*nb*2, name)
		if err != nil {
			return nil, err
		}
		data = make([]float32, n)
		for r := 0; r < rows; r++ {
			for b := 0; b < nb; b++ {
				i := r*nb + b
				bits := uint32(binary.LittleEndian.Uint16(sc[2*i:2*i+2])) << 16
				d := math.Float32frombits(bits)
				no, wo := i*16, r*cols+b*32
				for k := 0; k < 16; k++ {
					byteVal := nib[no+k]
					data[wo+k] = float32(int32(byteVal&15)-8) * d
					data[wo+k+16] = float32(int32(byteVal>>4)-8) * d
				}
			}
		}
	case "int8":
		if len(e.Shape) != 2 {
			return nil, modelErr("int8 tensor %s must be 2-D", name)
		}
		rows, cols := e.Shape[0], e.Shape[1]
		q, err := sliceBytes(buf, e.Offset, n, name)
		if err != nil {
			return nil, err
		}
		so, err := needScale()
		if err != nil {
			return nil, err
		}
		s, err := sliceBytes(buf, so, 4*rows, name)
		if err != nil {
			return nil, err
		}
		data = make([]float32, n)
		for r := 0; r < rows; r++ {
			scl := f32At(s, r)
			for c := 0; c < cols; c++ {
				data[r*cols+c] = float32(int8(q[r*cols+c])) * scl
			}
		}
	default:
		return nil, modelErr("tensor %s has unknown dtype %s", name, e.Dtype)
	}
	return &tensor{data: data, shape: append([]int(nil), e.Shape...)}, nil
}

// weights holds dequantised tensors keyed by name.
type weights struct {
	t map[string]*tensor
}

func newWeights(meta *Meta, buf []byte) (*weights, *Error) {
	t := make(map[string]*tensor, len(meta.Tensors))
	for i := range meta.Tensors {
		e := &meta.Tensors[i]
		td, err := dequant(e, buf)
		if err != nil {
			return nil, err
		}
		t[e.Name] = td
	}
	return &weights{t: t}, nil
}

func (w *weights) has(n string) bool {
	_, ok := w.t[n]
	return ok
}

// take moves a tensor out, checking its shape (nil entries match anything).
func (w *weights) take(n string, shape []*int) (*tensor, *Error) {
	x, ok := w.t[n]
	if !ok {
		return nil, modelErr("missing tensor %s", n)
	}
	delete(w.t, n)
	ok = len(x.shape) == len(shape)
	if ok {
		for i, b := range shape {
			if b != nil && x.shape[i] != *b {
				ok = false
				break
			}
		}
	}
	if !ok {
		return nil, modelErr("tensor %s has shape %v, expected %v", n, x.shape, shapeStr(shape))
	}
	return x, nil
}

func shapeStr(shape []*int) []any {
	out := make([]any, len(shape))
	for i, s := range shape {
		if s == nil {
			out[i] = "?"
		} else {
			out[i] = *s
		}
	}
	return out
}

func ip(v int) *int { return &v }
