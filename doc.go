// Package tinydecide is a native Go runtime for TinyDecide, a small typed
// decision model. You give it one message (the state) and any number of
// questions written in plain language; one encoder pass returns calibrated
// probabilities for all of them, with no text generation.
//
// It reads the same meta.json and model.bin as the reference JavaScript engine
// (tinydecide.js) and is a port of the official Rust runtime, checked against
// the published conformance fixtures.
//
//	model, err := tinydecide.Load("path/to/TinyDecide")
//	r, err := model.Answer("Book a table for Friday at 7:30", []tinydecide.Question{
//		tinydecide.NewSpan("Extract the time."),
//	})
//	// r.Answers[0].Text == "7:30"
package tinydecide
