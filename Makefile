.PHONY: build build-embed build-embedfull test

# Default build: no model embedded; Load reads from a directory.
build:
	go build ./...

# Embed the 10.4M build (4-bit, 16,000-token vocab). Model files ship in assets/.
build-embed:
	go build -tags embed ./...

# Embed the 13.8M build (4-bit, full 30,534-token vocab). Model files ship in assets/full/.
build-embedfull:
	go build -tags embedfull ./...

test:
	go test ./...
