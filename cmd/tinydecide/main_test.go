package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	tinydecide "github.com/loicalleyne/tinydecide-go"
)

func TestAssembleRequest_PipedTextAndFlags(t *testing.T) {
	// Piped message on stdin, questions supplied as CLI flags.
	req, err := assembleRequest(
		"Book a table for 4 on Friday at 7:30\n", // stdin
		"",                                       // --state
		stringList{"Which app?=reminders,calendar,restaurants"},
		stringList{"The message is urgent."},
		stringList{"How positive?=negative,neutral,positive"},
		stringList{"Extract the time."},
	)
	if err != nil {
		t.Fatalf("assembleRequest: %v", err)
	}
	if req.State != "Book a table for 4 on Friday at 7:30" {
		t.Errorf("state = %q, want trimmed piped text", req.State)
	}
	if len(req.Questions) != 4 {
		t.Fatalf("got %d questions, want 4", len(req.Questions))
	}
	// Order: choices, then scores, then nouls, then spans.
	want := []struct {
		kind tinydecide.QType
		text string
		opts []string
	}{
		{tinydecide.Choice, "Which app?", []string{"reminders", "calendar", "restaurants"}},
		{tinydecide.Score, "How positive?", []string{"negative", "neutral", "positive"}},
		{tinydecide.Noul, "The message is urgent.", nil},
		{tinydecide.Span, "Extract the time.", nil},
	}
	for i, w := range want {
		q := req.Questions[i]
		if q.Kind != w.kind || q.Text != w.text {
			t.Errorf("q%d = {%v %q}, want {%v %q}", i, q.Kind, q.Text, w.kind, w.text)
		}
		if !reflect.DeepEqual(q.Options, w.opts) {
			t.Errorf("q%d options = %v, want %v", i, q.Options, w.opts)
		}
	}
}

func TestAssembleRequest_StateFlagOverridesStdin(t *testing.T) {
	// CLI argument input: --state wins over any piped text.
	req, err := assembleRequest(
		"piped text",
		"The server is on fire, help now!",
		nil,
		stringList{"Is this urgent?"},
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("assembleRequest: %v", err)
	}
	if req.State != "The server is on fire, help now!" {
		t.Errorf("state = %q, want --state value", req.State)
	}
	if len(req.Questions) != 1 || req.Questions[0].Kind != tinydecide.Noul {
		t.Fatalf("questions = %+v, want one noul", req.Questions)
	}
}

func TestAssembleRequest_JSONRequestOnStdin(t *testing.T) {
	stdin := `{"state":"hello world","questions":[{"type":"noul","text":"Is it a greeting?"}]}`
	req, err := assembleRequest(stdin, "", nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("assembleRequest: %v", err)
	}
	if req.State != "hello world" {
		t.Errorf("state = %q", req.State)
	}
	if len(req.Questions) != 1 || req.Questions[0].Text != "Is it a greeting?" {
		t.Fatalf("questions = %+v", req.Questions)
	}
}

func TestAssembleRequest_JSONRequestMergesFlags(t *testing.T) {
	stdin := `{"state":"hello","questions":[{"type":"noul","text":"from json"}]}`
	req, err := assembleRequest(stdin, "", nil, nil, nil, stringList{"from flag"})
	if err != nil {
		t.Fatalf("assembleRequest: %v", err)
	}
	if len(req.Questions) != 2 {
		t.Fatalf("got %d questions, want 2 (json + flag)", len(req.Questions))
	}
	if req.Questions[1].Kind != tinydecide.Span || req.Questions[1].Text != "from flag" {
		t.Errorf("merged question = %+v, want span %q", req.Questions[1], "from flag")
	}
}

func TestAssembleRequest_BadJSON(t *testing.T) {
	if _, err := assembleRequest("{not json", "", nil, nil, nil, nil); err == nil {
		t.Fatal("expected an error for malformed JSON request")
	}
}

func TestResolveModelDir_FlagWins(t *testing.T) {
	t.Setenv("TINYDECIDE_MODEL", "/env/path")
	if got := resolveModelDir("/flag/path"); got != "/flag/path" {
		t.Errorf("got %q, want the --model value", got)
	}
}

func TestResolveModelDir_Env(t *testing.T) {
	t.Setenv("TINYDECIDE_MODEL", "/env/path")
	if got := resolveModelDir(""); got != "/env/path" {
		t.Errorf("got %q, want $TINYDECIDE_MODEL", got)
	}
}

func TestResolveModelDir_HomeDir(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, ".tinydecide")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"meta.json", "model.bin"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("TINYDECIDE_MODEL", "")
	t.Setenv("HOME", home)        // unix
	t.Setenv("USERPROFILE", home) // windows
	if got := resolveModelDir(""); got != dir {
		t.Errorf("got %q, want %q", got, dir)
	}
}

func TestResolveModelDir_FallsBackToDot(t *testing.T) {
	// No flag, no env, and ~/.tinydecide missing its files -> "." so Load can
	// fall back to an embedded model.
	home := t.TempDir() // empty, no .tinydecide
	t.Setenv("TINYDECIDE_MODEL", "")
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	if got := resolveModelDir(""); got != "." {
		t.Errorf("got %q, want \".\"", got)
	}
}

func TestResolveModelDir_HomeDirIgnoredWhenIncomplete(t *testing.T) {
	// ~/.tinydecide exists but lacks model.bin -> not used.
	home := t.TempDir()
	dir := filepath.Join(home, ".tinydecide")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TINYDECIDE_MODEL", "")
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	if got := resolveModelDir(""); got != "." {
		t.Errorf("got %q, want \".\" (incomplete dir ignored)", got)
	}
}

func TestSplitOptions(t *testing.T) {
	text, opts, err := splitOptions("How positive?= negative , neutral ,positive ", "score")
	if err != nil {
		t.Fatalf("splitOptions: %v", err)
	}
	if text != "How positive?" {
		t.Errorf("text = %q", text)
	}
	if want := []string{"negative", "neutral", "positive"}; !reflect.DeepEqual(opts, want) {
		t.Errorf("opts = %v, want %v", opts, want)
	}
}

func TestSplitOptions_MissingOptions(t *testing.T) {
	if _, _, err := splitOptions("no equals sign", "choice"); err == nil {
		t.Fatal("expected an error when options are missing")
	}
}

func TestSanitizeJSON_NaNBecomesNull(t *testing.T) {
	// A span-style answer with several unset NaN fields must encode as valid
	// JSON with those fields null.
	r := &tinydecide.Response{
		Answers: []tinydecide.Answer{{
			Kind:       tinydecide.Noul,
			P:          0.5,
			Confidence: math.NaN(),
			ScoreVal:   math.Inf(1),
			Pick:       -1,
		}},
	}
	clean, err := sanitizeJSON(r)
	if err != nil {
		t.Fatalf("sanitizeJSON: %v", err)
	}
	b, err := json.Marshal(clean)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ans := got["answers"].([]any)[0].(map[string]any)
	if ans["confidence"] != nil {
		t.Errorf("confidence = %v, want null", ans["confidence"])
	}
	if ans["score"] != nil {
		t.Errorf("score = %v, want null", ans["score"])
	}
	if ans["p"] != 0.5 {
		t.Errorf("p = %v, want 0.5", ans["p"])
	}
}

func TestDropVectors(t *testing.T) {
	clean, err := sanitizeJSON(&tinydecide.Response{
		IDs: []uint32{1, 2, 3},
		Answers: []tinydecide.Answer{{
			Kind: tinydecide.Choice,
			Qvec: []float32{1, 2},
			Z0:   []float64{3, 4},
			Proj: true,
		}},
	})
	if err != nil {
		t.Fatalf("sanitizeJSON: %v", err)
	}
	dropVectors(clean)
	m := clean.(map[string]any)
	if _, ok := m["ids"]; ok {
		t.Error("ids should have been dropped")
	}
	ans := m["answers"].([]any)[0].(map[string]any)
	for _, k := range []string{"qvec", "z0", "proj"} {
		if _, ok := ans[k]; ok {
			t.Errorf("%s should have been dropped", k)
		}
	}
}
