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
