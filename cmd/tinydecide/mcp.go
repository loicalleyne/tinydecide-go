package main

import (
	"context"
	"fmt"

	tinydecide "github.com/loicalleyne/tinydecide-go"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// version is the server version reported to MCP clients.
const version = "v1.0.0"

// decideInput is the MCP tool's input schema: one message plus the questions to
// answer about it, mirroring the JSON request the CLI accepts on stdin.
type decideInput struct {
	State     string        `json:"state" jsonschema:"the message to reason about"`
	Questions []mcpQuestion `json:"questions" jsonschema:"the questions to answer about the message"`
}

// mcpQuestion is a question in the tool's input schema. It mirrors
// tinydecide.Question but spells out the allowed kinds for the client.
type mcpQuestion struct {
	Type    string   `json:"type" jsonschema:"one of choice, noul, score, span"`
	Text    string   `json:"text" jsonschema:"the question, in plain language"`
	Options []string `json:"options,omitempty" jsonschema:"options for choice (2-32) or ordered levels for score; omit for noul and span"`
}

// runMCP serves the model as an MCP server over stdio until the client
// disconnects. maxTokens and strict carry the same meaning as the CLI flags.
func runMCP(m *tinydecide.TinyDecide, maxTokens int, strict bool) error {
	server := mcp.NewServer(
		&mcp.Implementation{Name: "tinydecide", Version: version},
		nil,
	)

	handler := func(ctx context.Context, _ *mcp.CallToolRequest, in decideInput) (*mcp.CallToolResult, any, error) {
		req, err := in.toRequest()
		if err != nil {
			return toolError(err), nil, nil
		}
		if req.State == "" {
			return toolError(fmt.Errorf("state is required")), nil, nil
		}
		if len(req.Questions) == 0 {
			return toolError(fmt.Errorf("at least one question is required")), nil, nil
		}
		result, err := answer(m, req, maxTokens, strict, false)
		if err != nil {
			return toolError(err), nil, nil
		}
		return nil, result, nil
	}

	mcp.AddTool(server, &mcp.Tool{
		Name: "decide",
		Description: "Answer questions about one message with the TinyDecide model. " +
			"Supports four question types: choice (pick one of 2-32 options), " +
			"noul (yes/no, returns the probability the statement is true), " +
			"score (place the message on ordered levels), and span (extract text). " +
			"Returns calibrated probabilities as JSON.",
	}, handler)

	return server.Run(context.Background(), &mcp.StdioTransport{})
}

// toRequest converts the MCP tool input into the internal request type,
// building typed questions and validating the kind.
func (in decideInput) toRequest() (request, error) {
	req := request{State: in.State}
	for i, q := range in.Questions {
		switch q.Type {
		case "choice":
			req.Questions = append(req.Questions, tinydecide.NewChoice(q.Text, q.Options))
		case "score":
			req.Questions = append(req.Questions, tinydecide.NewScore(q.Text, q.Options))
		case "noul":
			req.Questions = append(req.Questions, tinydecide.NewNoul(q.Text))
		case "span":
			req.Questions = append(req.Questions, tinydecide.NewSpan(q.Text))
		default:
			return request{}, fmt.Errorf("question %d: unknown type %q (want choice, noul, score or span)", i+1, q.Type)
		}
	}
	return req, nil
}

// toolError wraps an error as a failed tool result the client can read.
func toolError(err error) *mcp.CallToolResult {
	return &mcp.CallToolResult{
		IsError: true,
		Content: []mcp.Content{&mcp.TextContent{Text: err.Error()}},
	}
}
