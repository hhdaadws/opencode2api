// Package protocol translates Chat, Responses, Anthropic, and Gemini payloads
// and streams, and carries the opaque System One decision payloads.
package protocol

import "net/url"

type Protocol string

const (
	Chat      Protocol = "chat"
	Responses Protocol = "responses"
	Anthropic Protocol = "anthropic"
	// SystemOne is OpenCode Zen's structured decision endpoint. A request pairs
	// a free-form state with typed questions and the reply carries typed
	// answers, so the payload shares no shape with the chat/responses bridge and
	// is forwarded as-is.
	SystemOne Protocol = "systemone"
	// Google is the Gemini generateContent wire format. It is an upstream-only
	// protocol: clients keep using the three message endpoints and the bridge
	// converts their requests, responses and streams.
	Google Protocol = "google"
)

func Valid(p Protocol) bool {
	return p == Chat || p == Responses || p == Anthropic || p == SystemOne || p == Google
}

// StreamOnly reports whether the upstream is always called in streaming mode.
// Gemini pins SSE framing in the URL, which is also how OpenCode's own client
// talks to it, so a non-streaming downstream request is served by collapsing
// the upstream stream.
func StreamOnly(p Protocol) bool {
	return p == Google
}

func Path(protocol Protocol) string {
	switch protocol {
	case Responses:
		return "/v1/responses"
	case Anthropic:
		return "/v1/messages"
	case SystemOne:
		return "/v1/systemone"
	default:
		return "/v1/chat/completions"
	}
}

// UpstreamPath is the request path for a model on an upstream protocol. Gemini
// embeds the model ID in the path and carries no model field in the body.
func UpstreamPath(protocol Protocol, model string) string {
	if protocol == Google {
		return "/v1/models/" + url.PathEscape(model) + ":streamGenerateContent?alt=sse"
	}
	return Path(protocol)
}
