// Command corrections stores one correction per option and asks again with the
// resulting prototypes.
//
//	go run ./examples/corrections path/to/TinyDecide
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
	q := tinydecide.NewChoice("What kind of note is this?",
		[]string{"shopping", "task", "event"})

	lists := make([][]tinydecide.Example, 3)
	notes := []struct {
		text string
		k    int
	}{
		{"buy oat milk", 0},
		{"call the plumber", 1},
		{"dentist thursday 4pm", 2},
	}
	for _, n := range notes {
		r, err := model.Answer(n.text, []tinydecide.Question{q})
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		lists[n.k] = append(lists[n.k], tinydecide.ExampleFromAnswer(&r.Answers[0]))
	}
	protos := tinydecide.MakeProtos(q.Kind, lists, model.Beta(), nil)

	before, _ := model.Answer("pick up eggs", []tinydecide.Question{q})
	after, _ := model.AnswerWith("pick up eggs", []tinydecide.Question{q}, []*tinydecide.Protos{protos})
	fmt.Printf("zero-shot        %v\n", before.Answers[0].Probs)
	fmt.Printf("with corrections %v\n", after.Answers[0].Probs)
}
