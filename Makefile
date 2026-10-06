.PHONY: build build-embed build-embedfull test install install-nomodel install-embed install-embedfull

CMD := ./cmd/tinydecide

# Default build: no model embedded; Load reads from a directory.
build:
	go build ./...

# Embed the 10.4M build (4-bit, 16,000-token vocab). Model files ship in assets/.
build-embed:
	go build -tags embed ./...

# Embed the 13.8M build (4-bit, full 30,534-token vocab). Model files ship in assets/full/.
build-embedfull:
	go build -tags embedfull ./...

# Install the CLI with the full model baked in (13.8M, 30,534-token vocab).
# Self-contained: runs from $PATH with no model files or arguments. This is the
# default install.
install: install-embedfull

# Install the CLI with no embedded model. Needs a model dir at run time
# (--model, $TINYDECIDE_MODEL, or ~/.tinydecide).
install-nomodel:
	go install $(CMD)

# Install a self-contained CLI with the 10.4M model baked in (16,000-token vocab).
# Runs from $PATH with no model files or arguments.
install-embed:
	go install -tags embed $(CMD)

# Install a self-contained CLI with the full 13.8M model baked in (30,534-token vocab).
install-embedfull:
	go install -tags embedfull $(CMD)

test:
	go test ./...
