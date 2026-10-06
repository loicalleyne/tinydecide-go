//go:build embed && !embedfull

package tinydecide

import _ "embed"

// The "embed" tag bakes the 10.4M build into the binary: 4-bit matrices and
// embeddings with a 16,000-token vocabulary (assets/model.bin + assets/meta.json,
// committed to the repo). To build:
//
//	go build -tags embed ./...
//
// The embedded files must live inside this package directory (go:embed cannot
// reference parent paths). When built this way, Load uses the embedded model
// unless a directory with both meta.json and model.bin is passed to it.

//go:embed assets/meta.json
var embeddedMeta []byte

//go:embed assets/model.bin
var embeddedBin []byte

// embeddedModel returns the model baked into the binary at build time.
func embeddedModel() (meta, bin []byte, ok bool) {
	return embeddedMeta, embeddedBin, true
}
