# tinydecide-go

A native Go runtime for [TinyDecide](https://huggingface.co/TheREZOR/TinyDecide),
a small typed decision model. You give it one message (the *state*) and any
number of questions written in plain language; one encoder pass returns
calibrated probabilities for all of them, with no text generation.

It reads the same `meta.json` and `model.bin` as the reference JavaScript engine
(`tinydecide.js`). This package is a line-by-line port of the official
[Rust runtime](https://huggingface.co/TheREZOR/TinyDecide/tree/main/rust) and is
checked against the published conformance fixtures: token ids, picks and span
text match exactly, and probabilities agree to within ~1e-6.

```
go get github.com/loicalleyne/tinydecide-go
```

## Use it

```go
import tinydecide "github.com/loicalleyne/tinydecide-go"

model, err := tinydecide.Load("path/to/TinyDecide") // the folder with meta.json and model.bin
if err != nil {
	log.Fatal(err)
}

r, err := model.Answer(
	"Book a table for 4 at an Italian place near the station on Friday at 7:30",
	[]tinydecide.Question{
		tinydecide.NewChoice("Which app should handle this?",
			[]string{"reminders", "music", "calendar", "restaurants", "weather"}),
		tinydecide.NewNoul("The message is urgent."),
		tinydecide.NewScore("How positive is the tone?",
			[]string{"negative", "neutral", "positive"}),
		tinydecide.NewSpan("Extract the time."),
	},
)
// r.Answers[0].Pick == 3, r.Answers[1].P ≈ 0.219,
// r.Answers[2].ScoreVal ≈ 0.517, r.Answers[3].Text == "7:30"
```

Run it with `go run ./examples/quickstart path/to/TinyDecide`. The default path
is `.`, the Hugging Face repo layout.

`tinydecide.FromBytes(metaJSON, bin)` builds a model from bytes already in memory.

## Answers

| type     | fields that are set |
|----------|---------------------|
| `Choice` | `Probs` (one per option), `Pick`, `Confidence` (1 minus normalised entropy), `Qvec`, `Z0` |
| `Score`  | the same, plus `ScoreVal` from 0 (first level) to 1 (last level) |
| `Noul`   | `P`, the probability the statement is true, plus `Qvec` and `Z0` |
| `Span`   | `Text`, `PPresent`, `PSpan`, `Tok` (first and last state token), `Char` |

- `Char` holds **UTF-8 byte offsets** into the state string, so `state[a:b]` is
  the span before trimming. The JavaScript engine reports UTF-16 units instead.
- `Response` also carries `IDs`, `Tokens` (state, questions, total), `Truncated`
  and `MS`.
- The model reads the first 127 tokens of a message and sets `Truncated` when it
  cuts the rest.
- A question longer than 192 tokens returns an `*Error` of kind `ErrRequest`.
- A choice or score question needs 2 to 32 options; any other count is an
  `ErrRequest`, with the same message as the JavaScript engine.
- Unset numeric fields are `NaN`; unset `Pick` is `-1`; unset `Tok`/`Char` are `nil`.

## CLI

`cmd/tinydecide` is a command-line front end: it takes one message and any
number of questions, answers them in a single pass, and prints the result as
JSON. Unset `NaN` fields are rendered as `null` so the output is valid JSON.

```
# default: self-contained, full model baked in (13.8M, 30,534-token vocab),
# runs from $PATH with no arguments
make install

# smaller embedded model (10.4M, 16k-token vocab)
make install-embed

# no embedded model: needs a model dir at run time (see "Finding the model").
# Note the /cmd/tinydecide suffix — the module root is the library, not a binary.
make install-nomodel
go install github.com/loicalleyne/tinydecide-go/cmd/tinydecide@latest

# or run from a checkout without installing
go run ./cmd/tinydecide --model path/to/TinyDecide [flags]
```

> **Which command embeds a model?** Only the `make` targets do, because
> embedding is a build tag (`-tags embedfull` / `-tags embed`) and `go install`
> accepts no build tags. So `make install` gives you a self-contained binary
> with the full model, while `go install …/cmd/tinydecide@latest` embeds
> **nothing** and must find a model at run time.

### Using the full model with `go install`

A binary from `go install github.com/loicalleyne/tinydecide-go/cmd/tinydecide@latest`
has no model baked in, so give it the full weights at run time with any of:

```bash
# A) drop the full weights where the binary looks by default
mkdir -p ~/.tinydecide
cp assets/full/meta.json assets/full/model.bin ~/.tinydecide/

# B) point at them with an environment variable
export TINYDECIDE_MODEL=/path/to/assets/full

# C) pass them per-invocation
tinydecide --model /path/to/assets/full --noul "Is this urgent?"
```

`assets/full/` in this repo holds the full 30,534-token model; `assets/` holds
the smaller 16k-token one. The Hugging Face repo ships the same `meta.json` and
`model.bin` layout. To instead embed the full model in the binary, use
`make install` (or `go install -tags embedfull …/cmd/tinydecide` from a
checkout).

### Finding the model

So a binary on `$PATH` runs without pointing it at a model every time, the
model directory is resolved in this order:

1. the `--model` flag;
2. the `$TINYDECIDE_MODEL` environment variable;
3. `~/.tinydecide` (when it holds both `meta.json` and `model.bin`);
4. a model **embedded at build time** — `make install` bakes in the full model
   (13.8M, 30,534-token vocab) and `make install-embed` the smaller one (10.4M,
   16k-token vocab); either way the binary needs no model files at all;
5. the current directory.

For an installed CLI the easiest setup is `make install` (full model embedded);
alternatively drop the Hugging Face `meta.json` and `model.bin` into
`~/.tinydecide`.

### Questions and input

The message (the *state*) is piped on stdin or passed with `--state`; questions
are added with the repeatable `--choice`, `--noul`, `--score` and `--span`
flags. `--choice` and `--score` take options as `"text=opt1,opt2,..."`.

**Piped input** — pipe the message, add questions as flags:

```
echo "Book a table for 4 at an Italian place near the station on Friday at 7:30" \
  | tinydecide --model path/to/TinyDecide --indent \
      --choice "Which app should handle this?=reminders,music,calendar,restaurants,weather" \
      --noul  "The message is urgent." \
      --score "How positive is the tone?=negative,neutral,positive" \
      --span  "Extract the time."
```

**CLI argument input** — give the message with `--state` instead of a pipe:

```
tinydecide --model path/to/TinyDecide \
  --state "The server is on fire, help now!" \
  --noul "Is this urgent?" \
  --choice "Severity?=low,high"
```

A full request can also be piped as a single JSON object; stdin starting with
`{` is read as `{"state": "...", "questions": [{"type": "noul", "text": "..."}]}`.
Flags still merge in, and `--state` overrides the JSON's state.

| flag | meaning |
|------|---------|
| `--model` | model directory; defaults to `$TINYDECIDE_MODEL`, `~/.tinydecide`, an embedded model, then `.` |
| `--state` | the message (overrides piped text) |
| `--max-tokens` | max state tokens allowed (default: the model's `ts_max`) |
| `--strict` | exit non-zero when the state exceeds the limit instead of truncating |
| `--indent` | pretty-print the JSON |
| `--vectors` | include the internal `qvec`/`z0`/`ids` vectors (hidden by default) |
| `--mcp` | run as an MCP server over stdio (see below) |

The state is tokenised before answering. By default it is measured against the
model's `ts_max`; `--max-tokens` sets a different cap. When the state is over
the limit, `--strict` makes the command exit non-zero, otherwise it warns on
stderr and lets the model truncate the message (`"truncated": true` in the
output).

### Output

The command prints one JSON object. `answers` has one entry per question, in
the order the questions were given, plus token counts and timing:

```json
{
  "answers": [
    { "kind": "choice", "pick": 3, "confidence": 0.94,
      "probs": [0.003, 0.001, 0.012, 0.983, 0.002] },
    { "kind": "noul", "p": 0.219 },
    { "kind": "score", "pick": 1, "score": 0.517,
      "probs": [0.013, 0.928, 0.059] },
    { "kind": "span", "text": "7:30", "char": [69, 73], "tok": [15, 17],
      "p_present": 0.9999, "p_span": 0.9977 }
  ],
  "tokens": { "state": 19, "questions": 42, "total": 61 },
  "truncated": false,
  "ms": 12.3
}
```

Every answer carries all the fields; those that do not apply to its `kind` are
`null` (`pick` is `-1`). The fields that matter per type:

| `kind` | read these |
|--------|------------|
| `choice` | `pick` (index of the chosen option), `probs` (one per option), `confidence` |
| `score` | the same, plus `score` (0 = first level, 1 = last level) |
| `noul` | `p` — the probability the statement is true |
| `span` | `text` (the extracted substring), `char` (UTF-8 byte offsets into the state), `tok`, `p_present`, `p_span` |

`--vectors` adds the internal `qvec`/`z0`/`ids` arrays, off by default because
they are large and only useful for the corrections workflow.

### Choosing a model

Two models ship in the repo; they differ only in vocabulary size and binary
size, not in the question types or the token limits (both read the first 127
tokens of the message and cap a question at 192 tokens):

| build | directory | vocab | binary |
|-------|-----------|-------|--------|
| `make install`, `-tags embedfull` | `assets/full/` | 30,534 tokens | 13.8M |
| `make install-embed`, `-tags embed` | `assets/` | 16,000 tokens | 10.4M |

The full model handles rarer words with fewer `[UNK]` tokens; the smaller one
makes a lighter binary. When a word is out of vocabulary both still work — it
just becomes an unknown token.

### MCP server

With `--mcp` the binary runs as a [Model Context
Protocol](https://modelcontextprotocol.io) server over stdio instead of
answering once, so an MCP client (an editor or agent) can call TinyDecide as a
tool. `--model`, `--max-tokens` and `--strict` still apply; the others are
ignored.

It exposes a single tool, **`decide`**, taking a message and a list of
questions and returning the same JSON the CLI prints:

```json
{
  "state": "Book a table on Friday at 7:30",
  "questions": [
    {"type": "noul", "text": "Is this urgent?"},
    {"type": "choice", "text": "Which app?", "options": ["calendar", "restaurants"]},
    {"type": "span", "text": "Extract the time."}
  ]
}
```

A typical client config points at the installed binary:

```json
{
  "mcpServers": {
    "tinydecide": { "command": "tinydecide", "args": ["--mcp"] }
  }
}
```

The server loads its model the same way as the one-shot CLI (the resolution
order under "Finding the model"), so `make install` gives a self-contained
server that needs no `--model`. To serve the full model from a `go install`ed
binary, put the weights in `~/.tinydecide` or set `$TINYDECIDE_MODEL`.

The tool returns its JSON both as text content and as `structuredContent`. An
invalid request — an unknown question `type`, no questions, an empty `state`, or
a `--strict` token overflow — comes back as a tool error (`isError: true`)
rather than crashing the server, so the client can show the message and retry.

## Corrections

Store `(Qvec, Z0)` under the option a person picked, then pass prototypes on
later calls. Nothing retrains the model.

```go
q := tinydecide.NewChoice("What kind of note is this?",
	[]string{"shopping", "task", "event"})
lists := make([][]tinydecide.Example, 3)
for _, n := range notes { // n.k is the option a person picked
	r, _ := model.Answer(n.text, []tinydecide.Question{q})
	lists[n.k] = append(lists[n.k], tinydecide.ExampleFromAnswer(&r.Answers[0]))
}
protos := tinydecide.MakeProtos(q.Kind, lists, model.Beta(), nil) // nil: centre on the examples
r, _ := model.AnswerWith("pick up eggs", []tinydecide.Question{q},
	[]*tinydecide.Protos{protos})
```

For a noul question, list 0 is "false" and list 1 is "true". Without a center,
the centre is the mean of the examples, so a single example has no effect. If you
track the mean `Qvec` of every message asked with a question, pass it as the
`center`. `examples/corrections` runs the snippet above.

## Embed the model

Build tags bake a model into the binary so it ships without separate files.
The model files are committed to this repo under `assets/`, so embedding needs
no download. Two builds are available:

| tag         | build      | vocab        | size   |
|-------------|------------|--------------|--------|
| `embed`     | the 10.4M  | 16,000-token | 6.2 MB |
| `embedfull` | the 13.8M  | 30,534-token | 10 MB  |

Both are the same 4-bit format.

```sh
go build -tags embed ./...       # 10.4M build
go build -tags embedfull ./...   # 13.8M build
```

Because the `assets/` files ship in the module, a build tag works the same way
whether you build this repo from a checkout or pass it while building a program
that imports the module — `go build` applies tags to dependency packages too:

```sh
go build -tags embed ./...   # in your own program; embeds the 10.4M model
```

With an embed tag set, `Load(dir)` still prefers a directory that contains both
`meta.json` and `model.bin`, and falls back to the embedded model otherwise —
so pass an empty or non-existent path (e.g. `Load("")`) to use the baked-in one.
If both `embed` and `embedfull` are set, `embedfull` wins.

If you would rather control embedding entirely in your own code — for example to
embed your own model files or avoid the build tag — embed the bytes yourself and
pass them to `FromBytes`:

```go
import (
	_ "embed"

	tinydecide "github.com/loicalleyne/tinydecide-go"
)

//go:embed assets/meta.json
var metaJSON []byte

//go:embed assets/model.bin
var modelBin []byte

// model, err := tinydecide.FromBytes(metaJSON, modelBin)
```

## Test

The conformance test reads a model and fixtures from `testdata/`. They are not
committed because of their size; fetch them from the Hugging Face repo:

```sh
base=https://huggingface.co/TheREZOR/TinyDecide/resolve/main
mkdir -p testdata/model testdata/conformance
curl -sL $base/meta.json  -o testdata/model/meta.json
curl -sL $base/model.bin  -o testdata/model/model.bin
curl -sL $base/conformance/cases.json         -o testdata/conformance/cases.json
curl -sL $base/conformance/expected.S768.json -o testdata/conformance/expected.S768.json
go test ./...
```

## License

Apache-2.0, like the upstream model.
