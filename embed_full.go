//go:build embedfull

package tinydecide

import _ "embed"

// The "embedfull" tag bakes the 13.8M build into the binary: the same 4-bit
// format but with the full 30,534-token vocabulary (assets/full/model.bin +
// assets/full/meta.json, committed to the repo). To build:
//
//	go build -tags embedfull ./...
//
// embedfull takes precedence over embed if both tags are set. The embedded
// files must live inside this package directory (go:embed cannot reference
// parent paths). When built this way, Load uses the embedded model unless a
// directory with both meta.json and model.bin is passed to it.

//go:embed assets/full/meta.json
var embeddedMeta []byte

//go:embed assets/full/model.bin
var embeddedBin []byte

// embeddedModel returns the model baked into the binary at build time.
func embeddedModel() (meta, bin []byte, ok bool) {
	return embeddedMeta, embeddedBin, true
}
