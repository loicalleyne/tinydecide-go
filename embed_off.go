//go:build !embed && !embedfull

package tinydecide

// embeddedModel reports whether a model was baked into the binary at build
// time. With neither the "embed" nor "embedfull" build tag, no model is
// embedded.
func embeddedModel() (meta, bin []byte, ok bool) {
	return nil, nil, false
}
