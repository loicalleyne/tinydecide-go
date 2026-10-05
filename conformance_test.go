package tinydecide

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

const tol = 1e-3

func readJSON(t *testing.T, p string) map[string]json.RawMessage {
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("%s: %v", p, err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("%s: %v", p, err)
	}
	return m
}

func upd(m *float64, a, b float64) {
	d := math.Abs(a - b)
	if math.IsNaN(d) {
		*m = math.Inf(1)
	} else if d > *m {
		*m = d
	}
}

type expAnswer struct {
	Type       string    `json:"type"`
	Probs      []float64 `json:"probs"`
	Confidence float64   `json:"confidence"`
	Score      float64   `json:"score"`
	Pick       *int      `json:"pick"`
	P          float64   `json:"p"`
	Qvec       []float64 `json:"qvec"`
	Z0         []float64 `json:"z0"`
	PPresent   float64   `json:"p_present"`
	PSpan      float64   `json:"p_span"`
	Tok        [2]int    `json:"tok"`
	Text       string    `json:"text"`
}

func TestConformance(t *testing.T) {
	model, err := Load("testdata/model")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cases := readJSON(t, filepath.Join("testdata", "conformance", "cases.json"))
	exp := readJSON(t, filepath.Join("testdata", "conformance", "expected.S768.json"))

	var maxDp, maxDq, maxDz, maxDproto float64

	// tokenizer
	var tokStrings []string
	var tokExp [][]uint32
	json.Unmarshal(cases["tokenizer"], &tokStrings)
	json.Unmarshal(exp["tokenizer"], &tokExp)
	if len(tokStrings) != len(tokExp) {
		t.Fatalf("tokenizer case count: %d vs %d", len(tokStrings), len(tokExp))
	}
	for i, s := range tokStrings {
		got := model.Tokenizer().Encode(s).ids
		if !equalU32(got, tokExp[i]) {
			t.Errorf("tokenizer differs for %q:\n  exp  %v\n  got  %v", s, tokExp[i], got)
		}
	}

	// requests
	var reqCases []struct {
		Name      string     `json:"name"`
		State     string     `json:"state"`
		Questions []Question `json:"questions"`
	}
	var reqExp []struct {
		Name      string      `json:"name"`
		Error     *string     `json:"error"`
		IDs       []uint32    `json:"ids"`
		Truncated bool        `json:"truncated"`
		Answers   []expAnswer `json:"answers"`
	}
	json.Unmarshal(cases["requests"], &reqCases)
	json.Unmarshal(exp["requests"], &reqExp)
	if len(reqCases) != len(reqExp) {
		t.Fatalf("request count: %d vs %d", len(reqCases), len(reqExp))
	}
	for i, c := range reqCases {
		e := reqExp[i]
		r, rerr := model.Answer(c.State, c.Questions)
		if e.Error != nil {
			if rerr == nil {
				t.Errorf("%s: expected error %q, got none", c.Name, *e.Error)
			} else if rerr.Error() != *e.Error {
				t.Errorf("%s: error %q, want %q", c.Name, rerr.Error(), *e.Error)
			}
			continue
		}
		if rerr != nil {
			t.Errorf("%s: unexpected error %v", c.Name, rerr)
			continue
		}
		if !equalU32(r.IDs, e.IDs) {
			t.Errorf("%s: ids differ", c.Name)
			continue
		}
		if r.Truncated != e.Truncated {
			t.Errorf("%s: truncated %v want %v", c.Name, r.Truncated, e.Truncated)
		}
		if len(r.Answers) != len(e.Answers) {
			t.Errorf("%s: answer count", c.Name)
			continue
		}
		for j := range e.Answers {
			checkAnswer(t, c.Name, &e.Answers[j], &r.Answers[j], &maxDp, &maxDq, &maxDz)
		}
	}

	// corrections
	beta := model.Beta()
	var protoCases []struct {
		Name     string   `json:"name"`
		State    string   `json:"state"`
		Question Question `json:"question"`
		NoCenter bool     `json:"no_center"`
	}
	var protoExp []struct {
		Name     string      `json:"name"`
		Examples [][]Example `json:"examples"`
		IDs      []uint32    `json:"ids"`
		Answers  []expAnswer `json:"answers"`
		Protos   struct {
			Cnt    []int     `json:"cnt"`
			Vec    []float64 `json:"vec"`
			Center []float64 `json:"center"`
			Lam    *float64  `json:"lam"`
		} `json:"protos"`
	}
	json.Unmarshal(cases["protos"], &protoCases)
	json.Unmarshal(exp["protos"], &protoExp)
	if len(protoCases) != len(protoExp) {
		t.Fatalf("protos case count")
	}
	for i, c := range protoCases {
		e := protoExp[i]
		p := MakeProtos(c.Question.Kind, e.Examples, beta, nil)
		if p == nil {
			t.Errorf("%s: nil protos", c.Name)
			continue
		}
		if c.NoCenter {
			p.Center = nil
		}
		if !equalInt(p.Cnt, e.Protos.Cnt) || len(p.Vec) != len(e.Protos.Vec) {
			t.Errorf("%s: protos cnt/vec shape", c.Name)
		}
		for k := range e.Protos.Vec {
			if k < len(p.Vec) {
				upd(&maxDproto, e.Protos.Vec[k], float64(p.Vec[k]))
			}
		}
		if p.Center != nil {
			for k := range e.Protos.Center {
				if k < len(p.Center) {
					upd(&maxDproto, e.Protos.Center[k], float64(p.Center[k]))
				}
			}
		} else if e.Protos.Center != nil {
			t.Errorf("%s: center expected", c.Name)
		}
		if (e.Protos.Lam == nil) != (p.Lam == nil) || (e.Protos.Lam != nil && *e.Protos.Lam != *p.Lam) {
			t.Errorf("%s: lam differs exp %v got %v", c.Name, e.Protos.Lam, p.Lam)
		}
		r, rerr := model.AnswerWith(c.State, []Question{c.Question}, []*Protos{p})
		if rerr != nil {
			t.Errorf("%s: %v", c.Name, rerr)
			continue
		}
		if !equalU32(r.IDs, e.IDs) {
			t.Errorf("%s: ids differ", c.Name)
		}
		for j := range e.Answers {
			checkAnswer(t, c.Name, &e.Answers[j], &r.Answers[j], &maxDp, &maxDq, &maxDz)
		}
	}

	t.Logf("max|dp|=%.2e max|dqvec|=%.2e max|dz0|=%.2e max|dproto|=%.2e", maxDp, maxDq, maxDz, maxDproto)
	if maxDp > tol {
		t.Errorf("max|dp| %.3e exceeds tol", maxDp)
	}
	if maxDq > tol {
		t.Errorf("max|dqvec| %.3e exceeds tol", maxDq)
	}
	if maxDproto > 1e-5 {
		t.Errorf("max|dproto| %.3e exceeds tol", maxDproto)
	}
}

func checkAnswer(t *testing.T, name string, e *expAnswer, a *Answer, maxDp, maxDq, maxDz *float64) {
	if e.Type != a.Kind.String() {
		t.Errorf("%s: answer type %s want %s", name, a.Kind, e.Type)
		return
	}
	switch a.Kind {
	case Choice, Score:
		if len(e.Probs) != len(a.Probs) {
			t.Errorf("%s: probs len", name)
			return
		}
		for i := range e.Probs {
			upd(maxDp, e.Probs[i], a.Probs[i])
		}
		upd(maxDp, e.Confidence, a.Confidence)
		if a.Kind == Score {
			upd(maxDp, e.Score, a.ScoreVal)
		}
		if e.Pick != nil && *e.Pick != a.Pick {
			t.Errorf("%s: pick %d want %d", name, a.Pick, *e.Pick)
		}
	case Noul:
		upd(maxDp, e.P, a.P)
	case Span:
		upd(maxDp, e.PPresent, a.PPresent)
		upd(maxDp, e.PSpan, a.PSpan)
		if a.Tok == nil || *a.Tok != e.Tok || a.Text != e.Text {
			t.Errorf("%s: span exp %v %q got %v %q", name, e.Tok, e.Text, a.Tok, a.Text)
		}
	}
	if a.Kind != Span {
		for i := range e.Qvec {
			if i < len(a.Qvec) {
				upd(maxDq, e.Qvec[i], float64(a.Qvec[i]))
			}
		}
		for i := range e.Z0 {
			if i < len(a.Z0) {
				upd(maxDz, e.Z0[i], a.Z0[i])
			}
		}
	}
}

func equalU32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalInt(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestBadInputIsError(t *testing.T) {
	model, err := Load("testdata/model")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if _, err := model.Answer("hi", []Question{NewChoice("Pick one.", nil)}); err == nil {
		t.Error("expected error for empty options")
	}
	p := &Protos{Vec: make([]float32, 3), Cnt: []int{1, 1}}
	if _, err := model.AnswerWith("hi", []Question{NewNoul("It is a test.")}, []*Protos{p}); err == nil {
		t.Error("expected error for short proto vec")
	}
	if _, err := FromBytes([]byte("{}"), nil); err == nil {
		t.Error("expected error for empty meta")
	}
	if _, err := Load("/nonexistent"); err == nil {
		t.Error("expected error for missing dir")
	}
}
