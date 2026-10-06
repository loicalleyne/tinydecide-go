package main

import (
	"reflect"
	"testing"

	tinydecide "github.com/loicalleyne/tinydecide-go"
)

func TestDecideInput_ToRequest(t *testing.T) {
	in := decideInput{
		State: "Book a table on Friday at 7:30",
		Questions: []mcpQuestion{
			{Type: "choice", Text: "Which app?", Options: []string{"calendar", "restaurants"}},
			{Type: "score", Text: "How positive?", Options: []string{"low", "high"}},
			{Type: "noul", Text: "Is this urgent?"},
			{Type: "span", Text: "Extract the time."},
		},
	}
	req, err := in.toRequest()
	if err != nil {
		t.Fatalf("toRequest: %v", err)
	}
	if req.State != in.State {
		t.Errorf("state = %q", req.State)
	}
	want := []struct {
		kind tinydecide.QType
		text string
		opts []string
	}{
		{tinydecide.Choice, "Which app?", []string{"calendar", "restaurants"}},
		{tinydecide.Score, "How positive?", []string{"low", "high"}},
		{tinydecide.Noul, "Is this urgent?", nil},
		{tinydecide.Span, "Extract the time.", nil},
	}
	if len(req.Questions) != len(want) {
		t.Fatalf("got %d questions, want %d", len(req.Questions), len(want))
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

func TestDecideInput_ToRequest_UnknownType(t *testing.T) {
	in := decideInput{
		State:     "hi",
		Questions: []mcpQuestion{{Type: "bogus", Text: "?"}},
	}
	if _, err := in.toRequest(); err == nil {
		t.Fatal("expected an error for an unknown question type")
	}
}

func TestToolError(t *testing.T) {
	r := toolError(errString("boom"))
	if !r.IsError {
		t.Error("IsError should be true")
	}
	if len(r.Content) != 1 {
		t.Fatalf("got %d content items, want 1", len(r.Content))
	}
}

type errString string

func (e errString) Error() string { return string(e) }
