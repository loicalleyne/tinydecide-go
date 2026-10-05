package tinydecide

import "fmt"

// ErrKind classifies the errors this package returns.
type ErrKind int

const (
	// ErrIO means reading meta.json or model.bin failed.
	ErrIO ErrKind = iota
	// ErrJSON means meta.json is not valid JSON or misses a field.
	ErrJSON
	// ErrModel means the model files are inconsistent (missing tensor, wrong
	// shape, offset past the end, ...).
	ErrModel
	// ErrRequest means the request cannot be answered, for example a question
	// longer than the model accepts.
	ErrRequest
)

// Error is everything that can go wrong when loading a model or answering a
// request. It mirrors the Rust port's error enum.
type Error struct {
	Kind ErrKind
	Msg  string
	Err  error
}

func (e *Error) Error() string {
	switch e.Kind {
	case ErrJSON:
		return "meta.json: " + e.Msg
	default:
		return e.Msg
	}
}

func (e *Error) Unwrap() error { return e.Err }

func modelErr(format string, a ...any) *Error {
	return &Error{Kind: ErrModel, Msg: fmt.Sprintf(format, a...)}
}

func requestErr(format string, a ...any) *Error {
	return &Error{Kind: ErrRequest, Msg: fmt.Sprintf(format, a...)}
}

func ioErr(err error) *Error {
	return &Error{Kind: ErrIO, Msg: err.Error(), Err: err}
}

func jsonErr(err error) *Error {
	return &Error{Kind: ErrJSON, Msg: err.Error(), Err: err}
}
