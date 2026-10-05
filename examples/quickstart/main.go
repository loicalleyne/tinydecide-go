// Command quickstart runs TinyDecide on one message with four question types.
//
//	go run ./examples/quickstart path/to/TinyDecide
//
// The directory holds meta.json and model.bin (default: ".", the Hugging Face repo layout).
package main

import (
	"fmt"
	"os"

	tinydecide "github.com/loicalleyne/tinydecide-go"
)

func main() {
	dir := "."
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	model, err := tinydecide.Load(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
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
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	app, urgent, tone, time := r.Answers[0], r.Answers[1], r.Answers[2], r.Answers[3]
	fmt.Printf("app    %v pick %d\n", app.Probs, app.Pick) // one probability per option
	fmt.Printf("urgent %.3f\n", urgent.P)                  // P(true)
	fmt.Printf("tone   %.3f\n", tone.ScoreVal)             // 0 = negative, 1 = positive
	fmt.Printf("time   %q\n", time.Text)                   // "7:30"
	fmt.Printf("%d tokens, %.0f ms\n", r.Tokens.Total, r.MS)
}
