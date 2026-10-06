// Command tinydecide answers questions about one message using the TinyDecide
// model and prints the result as JSON.
//
// The message (the "state") can be piped on stdin or given with --state. Stdin
// is auto-detected: a JSON object is read as a full request
//
//	{"state": "...", "questions": [{"type": "choice", "text": "...", "options": ["a","b"]}, ...]}
//
// and anything else is treated as the raw message text, with the questions
// supplied by the repeatable --choice/--noul/--score/--span flags.
//
// Usage:
//
//	echo "Book a table for 4 on Friday at 7:30" | tinydecide \
//	    --model path/to/TinyDecide \
//	    --choice "Which app?=reminders,calendar,restaurants" \
//	    --noul   "The message is urgent." \
//	    --score  "How positive?=negative,neutral,positive" \
//	    --span   "Extract the time."
//
//	echo '{"state":"...","questions":[{"type":"noul","text":"Is it urgent?"}]}' | tinydecide --model .
//
// The state is tokenised before answering. By default the limit is the model's
// own ts_max; --max-tokens overrides it. With --strict the command exits
// non-zero when the state exceeds the limit instead of letting the model
// silently truncate it.
//
// The model directory is resolved in this order: --model, then
// $TINYDECIDE_MODEL, then ~/.tinydecide, then a model embedded at build time
// (go build -tags embed ./cmd/tinydecide), then the current directory. This
// lets an installed binary on $PATH run without a --model argument.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	tinydecide "github.com/loicalleyne/tinydecide-go"
)

// stringList collects a repeatable flag's values.
type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ", ") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

// request is the JSON shape accepted on stdin and used internally.
type request struct {
	State     string                `json:"state"`
	Questions []tinydecide.Question `json:"questions"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "tinydecide: "+err.Error())
		os.Exit(1)
	}
}

func run() error {
	var (
		model     = flag.String("model", "", "directory holding meta.json and model.bin (default: $TINYDECIDE_MODEL, ~/.tinydecide, embedded model, or .)")
		state     = flag.String("state", "", "the message to reason about (overrides stdin text)")
		maxTokens = flag.Int("max-tokens", -1, "max state tokens allowed (default: model's ts_max)")
		strict    = flag.Bool("strict", false, "error out instead of truncating when state exceeds the limit")
		indent    = flag.Bool("indent", false, "pretty-print the JSON output")
		vectors   = flag.Bool("vectors", false, "include internal vectors (qvec, z0, ids) in the output")
		mcpFlag   = flag.Bool("mcp", false, "run as an MCP server over stdio instead of answering once")
		choices   stringList
		nouls     stringList
		scores    stringList
		spans     stringList
	)
	flag.Var(&choices, "choice", "choice question, \"text=opt1,opt2,...\" (repeatable)")
	flag.Var(&nouls, "noul", "noul (yes/no) question, \"text\" (repeatable)")
	flag.Var(&scores, "score", "score question, \"text=low,...,high\" (repeatable)")
	flag.Var(&spans, "span", "span extraction question, \"text\" (repeatable)")
	flag.Parse()

	if *mcpFlag {
		m, lerr := tinydecide.Load(resolveModelDir(*model))
		if lerr != nil {
			return lerr
		}
		return runMCP(m, *maxTokens, *strict)
	}

	req, err := buildRequest(*state, choices, nouls, scores, spans)
	if err != nil {
		return err
	}
	if strings.TrimSpace(req.State) == "" {
		return fmt.Errorf("no state: pipe a message on stdin or pass --state")
	}
	if len(req.Questions) == 0 {
		return fmt.Errorf("no questions: pass --choice/--noul/--score/--span or a JSON request on stdin")
	}

	m, lerr := tinydecide.Load(resolveModelDir(*model))
	if lerr != nil {
		return lerr
	}

	limit := m.Meta().Format.TSMax
	if *maxTokens >= 0 {
		limit = *maxTokens
	}
	n := m.Tokenizer().Count(req.State)
	if n > limit {
		if *strict {
			return fmt.Errorf("state is %d tokens, over the max of %d", n, limit)
		}
		fmt.Fprintf(os.Stderr, "tinydecide: warning: state is %d tokens, over %d; the model will truncate it\n", n, limit)
	}

	clean, aerr := answer(m, req, *maxTokens, *strict, *vectors)
	if aerr != nil {
		return aerr
	}
	enc := json.NewEncoder(os.Stdout)
	if *indent {
		enc.SetIndent("", "  ")
	}
	return enc.Encode(clean)
}

// answer runs the model on one request and returns the JSON-ready result: a
// generic structure with NaN/Inf fields turned to null, the internal vectors
// stripped unless vectors is true. It enforces the token limit the same way the
// CLI does: over the limit with strict set is an error, otherwise the model
// truncates. Shared by the one-shot CLI and the MCP server.
func answer(m *tinydecide.TinyDecide, req request, maxTokens int, strict, vectors bool) (any, error) {
	limit := m.Meta().Format.TSMax
	if maxTokens >= 0 {
		limit = maxTokens
	}
	if n := m.Tokenizer().Count(req.State); n > limit && strict {
		return nil, fmt.Errorf("state is %d tokens, over the max of %d", n, limit)
	}
	r, aerr := m.Answer(req.State, req.Questions)
	if aerr != nil {
		return nil, aerr
	}
	// The Answer struct leaves unset numeric fields as NaN, which is not valid
	// JSON; round-trip through a sanitiser that turns NaN/Inf into null.
	clean, err := sanitizeJSON(r)
	if err != nil {
		return nil, err
	}
	if !vectors {
		dropVectors(clean)
	}
	return clean, nil
}

// resolveModelDir picks the directory to load the model from. An explicit
// --model always wins. Otherwise it tries $TINYDECIDE_MODEL and ~/.tinydecide,
// using the first that holds both meta.json and model.bin. If none do it
// returns ".", which makes Load fall back to a model embedded at build time
// (go build -tags embed) when there is one — so a binary on $PATH just works.
func resolveModelDir(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	if env := os.Getenv("TINYDECIDE_MODEL"); env != "" {
		return env
	}
	if home, err := os.UserHomeDir(); err == nil {
		if dir := filepath.Join(home, ".tinydecide"); hasModel(dir) {
			return dir
		}
	}
	return "."
}

// hasModel reports whether dir contains both model files.
func hasModel(dir string) bool {
	for _, f := range []string{"meta.json", "model.bin"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return false
		}
	}
	return true
}

// buildRequest assembles the request from stdin plus flags. If stdin holds a
// JSON object it is used as the request; its state/questions are merged with
// (and overridden by) anything given on the command line.
func buildRequest(state string, choices, nouls, scores, spans stringList) (request, error) {
	stdin, err := readStdin()
	if err != nil {
		return request{}, err
	}
	return assembleRequest(stdin, state, choices, nouls, scores, spans)
}

// assembleRequest builds the request from already-read stdin content and the
// command-line flags. Separated from buildRequest so it can be tested without
// touching os.Stdin.
func assembleRequest(stdin, state string, choices, nouls, scores, spans stringList) (request, error) {
	var req request

	trimmed := strings.TrimSpace(stdin)
	if strings.HasPrefix(trimmed, "{") {
		if err := json.Unmarshal([]byte(trimmed), &req); err != nil {
			return req, fmt.Errorf("parsing JSON request on stdin: %w", err)
		}
	} else if trimmed != "" {
		req.State = trimmed
	}

	if state != "" {
		req.State = state
	}

	for _, c := range choices {
		text, opts, err := splitOptions(c, "choice")
		if err != nil {
			return req, err
		}
		req.Questions = append(req.Questions, tinydecide.NewChoice(text, opts))
	}
	for _, s := range scores {
		text, opts, err := splitOptions(s, "score")
		if err != nil {
			return req, err
		}
		req.Questions = append(req.Questions, tinydecide.NewScore(text, opts))
	}
	for _, n := range nouls {
		req.Questions = append(req.Questions, tinydecide.NewNoul(n))
	}
	for _, s := range spans {
		req.Questions = append(req.Questions, tinydecide.NewSpan(s))
	}
	return req, nil
}

// splitOptions parses "text=opt1,opt2,..." for choice/score questions.
func splitOptions(spec, kind string) (string, []string, error) {
	text, list, ok := strings.Cut(spec, "=")
	if !ok {
		return "", nil, fmt.Errorf("%s question %q needs options: \"text=opt1,opt2,...\"", kind, spec)
	}
	var opts []string
	for _, o := range strings.Split(list, ",") {
		if o = strings.TrimSpace(o); o != "" {
			opts = append(opts, o)
		}
	}
	return strings.TrimSpace(text), opts, nil
}

// sanitizeJSON renders v to a generic structure in which every non-finite
// float (NaN, +Inf, -Inf) is replaced by nil, so it can be JSON-encoded. It
// works field by field without knowing the concrete types, using reflection.
func sanitizeJSON(v any) (any, error) {
	return scrub(reflect.ValueOf(v)), nil
}

func scrub(v reflect.Value) any {
	switch v.Kind() {
	case reflect.Interface, reflect.Ptr:
		if v.IsNil() {
			return nil
		}
		return scrub(v.Elem())
	case reflect.Float32, reflect.Float64:
		f := v.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil
		}
		return f
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return nil
		}
		out := make([]any, v.Len())
		for i := 0; i < v.Len(); i++ {
			out[i] = scrub(v.Index(i))
		}
		return out
	case reflect.Map:
		out := make(map[string]any, v.Len())
		for _, k := range v.MapKeys() {
			out[fmt.Sprint(k.Interface())] = scrub(v.MapIndex(k))
		}
		return out
	case reflect.Struct:
		t := v.Type()
		out := make(map[string]any, t.NumField())
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if f.PkgPath != "" {
				continue // unexported
			}
			name, omit := jsonFieldName(f)
			if name == "-" {
				continue
			}
			val := scrub(v.Field(i))
			if omit && isEmptyValue(v.Field(i)) {
				continue
			}
			out[name] = val
		}
		return out
	default:
		return v.Interface()
	}
}

// jsonFieldName returns the JSON key for a struct field and whether it is
// omitempty.
func jsonFieldName(f reflect.StructField) (string, bool) {
	tag := f.Tag.Get("json")
	if tag == "" {
		return f.Name, false
	}
	name, opts, _ := strings.Cut(tag, ",")
	if name == "" {
		name = f.Name
	}
	return name, strings.Contains(opts, "omitempty")
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Slice, reflect.Map, reflect.Array, reflect.String:
		return v.Len() == 0
	case reflect.Ptr, reflect.Interface:
		return v.IsNil()
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	}
	return false
}

// dropVectors removes the bulky internal vectors from a sanitised response so
// the default output stays readable.
func dropVectors(v any) {
	m, ok := v.(map[string]any)
	if !ok {
		return
	}
	delete(m, "ids")
	if answers, ok := m["answers"].([]any); ok {
		for _, a := range answers {
			if am, ok := a.(map[string]any); ok {
				delete(am, "qvec")
				delete(am, "z0")
				delete(am, "proj")
			}
		}
	}
}

// readStdin returns piped stdin, or "" when stdin is a terminal.
func readStdin() (string, error) {
	info, err := os.Stdin.Stat()
	if err != nil {
		return "", nil
	}
	if info.Mode()&os.ModeCharDevice != 0 {
		return "", nil // interactive terminal, nothing piped
	}
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", fmt.Errorf("reading stdin: %w", err)
	}
	return string(b), nil
}
