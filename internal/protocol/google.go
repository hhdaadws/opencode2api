package protocol

import (
	"encoding/base64"
	"errors"
	"fmt"
	"mime"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"opencode2api/internal/identity"
	"opencode2api/internal/jsonutil"
)

const (
	// googleSignatureSeparator joins a Gemini function call ID with the thought
	// signature the model attached to that call. Gemini 3 rejects a replayed
	// function call without its signature, and none of the client protocols has
	// a field for one on a tool call, but every client returns the tool call ID
	// verbatim. The signature is stored base64url-encoded so the joined ID stays
	// within [A-Za-z0-9_-], which Anthropic tool_use IDs require.
	googleSignatureSeparator = "__thought__"
	// googleSkipSignature is the sentinel Google documents for replaying a
	// Gemini 3 function call whose signature was lost, for example a call that
	// another model made earlier in the conversation.
	googleSkipSignature = "skip_thought_signature_validator"
)

var (
	googleModelPattern       = regexp.MustCompile(`(?i)(^|/)gemini-`)
	googleLegacyModelPattern = regexp.MustCompile(`(?i)(^|/)gemini-(?:(?:1|2)(?:[.-]|$)|pro(?:-vision)?$)`)
	googleMajorPattern       = regexp.MustCompile(`(?i)^gemini(?:-live)?-(\d+)`)
	// googleThinkingLevels is Gemini's thinkingLevel ladder, lowest first.
	googleThinkingLevels = []string{"minimal", "low", "medium", "high"}
)

// googleRequiresSignature reports whether a model validates thought signatures
// on replayed function calls. Gemini 1.x and 2.x do not; model IDs are
// open-ended, so any other Gemini ID gets the current behavior.
func googleRequiresSignature(model string) bool {
	return googleModelPattern.MatchString(model) && !googleLegacyModelPattern.MatchString(model)
}

// googleOmitsCallIDs reports whether a model predates explicit function call
// IDs; Gemini 1.x and 2.x ignore or reject them.
func googleOmitsCallIDs(model string) bool {
	match := googleMajorPattern.FindStringSubmatch(model)
	if match == nil {
		return false
	}
	major, err := strconv.Atoi(match[1])
	return err == nil && major < 3
}

// googleToolID returns the client-facing ID of a Gemini function call.
func googleToolID(id, signature string) string {
	if signature == "" {
		return id
	}
	raw, err := base64.StdEncoding.DecodeString(signature)
	if err != nil {
		// A signature that is not base64 cannot be embedded safely. Dropping it
		// falls back to the skip sentinel when the call is replayed.
		return id
	}
	return id + googleSignatureSeparator + base64.RawURLEncoding.EncodeToString(raw)
}

// splitGoogleToolID reverses googleToolID. An ID that does not carry a valid
// signature, including one minted by another protocol, is returned unchanged.
func splitGoogleToolID(id string) (string, string) {
	base, encoded, ok := strings.Cut(id, googleSignatureSeparator)
	if !ok || base == "" {
		return id, ""
	}
	raw, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil || len(raw) == 0 {
		return id, ""
	}
	return base, base64.StdEncoding.EncodeToString(raw)
}

type googleContent struct {
	role      string
	parts     []any
	responses bool
}

func encodeGoogleRequest(request bridgeRequest) (map[string]any, error) {
	output := map[string]any{}
	instructions := append(append([]bridgeBlock(nil), request.System...), request.Developer...)
	var system []any
	for _, block := range instructions {
		switch block.Kind {
		case "text":
			if block.Text != "" {
				system = append(system, map[string]any{"text": block.Text})
			}
		default:
			return nil, fmt.Errorf("%s content is not valid inside Gemini system instructions; send it as a user message", block.Kind)
		}
	}
	if len(system) > 0 {
		output["systemInstruction"] = map[string]any{"parts": system}
	}

	// Gemini names the function a response belongs to; the bridge only keeps
	// the call ID on a tool result, so the names come from the history.
	names := make(map[string]string)
	for _, message := range request.Messages {
		for _, block := range message.Blocks {
			if block.Kind == "tool_call" && block.ID != "" {
				names[block.ID] = block.Name
			}
		}
	}
	omitIDs := googleOmitsCallIDs(request.Model)
	requireSignature := googleRequiresSignature(request.Model)
	var contents []googleContent
	// Consecutive turns of one role are merged, except that Gemini wants every
	// response to a batch of calls in one user turn of its own: a function
	// response turn never absorbs text, and text never joins it.
	appendContent := func(role string, parts []any, responses bool) {
		if len(parts) == 0 {
			return
		}
		if n := len(contents); n > 0 {
			last := &contents[n-1]
			if last.role == role && (role == "model" || last.responses == responses) {
				last.parts = append(last.parts, parts...)
				return
			}
		}
		contents = append(contents, googleContent{role: role, parts: parts, responses: responses})
	}

	for i, message := range request.Messages {
		if message.Role == "assistant" {
			var parts []any
			signed := false
			for _, block := range message.Blocks {
				switch block.Kind {
				case "reasoning":
					// Thought summaries are not replayed: Gemini does not need them
					// for context, and the signature it does need travels with the
					// function call ID.
				case "tool_call":
					if block.Name == "" {
						return nil, fmt.Errorf("messages[%d]: tool call must contain a name", i)
					}
					id, signature := splitGoogleToolID(block.ID)
					call := map[string]any{"name": block.Name, "args": googleArguments(block)}
					if !omitIDs && id != "" {
						call["id"] = id
					}
					part := map[string]any{"functionCall": call}
					// Parallel calls carry one signature, on the first call. An
					// unsigned call before any signed one lost its signature and
					// is replayed with the sentinel instead.
					if signature != "" {
						part["thoughtSignature"] = signature
						signed = true
					} else if requireSignature && !signed {
						part["thoughtSignature"] = googleSkipSignature
					}
					parts = append(parts, part)
				case "tool_result":
					return nil, fmt.Errorf("messages[%d]: assistant message contains tool results", i)
				default:
					part, err := googlePart(block)
					if err != nil {
						return nil, fmt.Errorf("messages[%d]: %w", i, err)
					}
					if part != nil {
						parts = append(parts, part)
					}
				}
			}
			appendContent("model", parts, false)
			continue
		}
		var responses, parts []any
		for _, block := range message.Blocks {
			switch block.Kind {
			case "reasoning":
			case "tool_call":
				return nil, fmt.Errorf("messages[%d]: user message contains tool calls", i)
			case "tool_result":
				response, err := googleFunctionResponse(block, names, omitIDs)
				if err != nil {
					return nil, fmt.Errorf("messages[%d]: %w", i, err)
				}
				responses = append(responses, response)
			default:
				part, err := googlePart(block)
				if err != nil {
					return nil, fmt.Errorf("messages[%d]: %w", i, err)
				}
				if part != nil {
					parts = append(parts, part)
				}
			}
		}
		appendContent("user", responses, true)
		appendContent("user", parts, false)
	}
	encoded := make([]any, 0, len(contents))
	for _, content := range contents {
		encoded = append(encoded, map[string]any{"role": content.role, "parts": content.parts})
	}
	output["contents"] = encoded

	if len(request.Tools) > 0 {
		declarations := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			declaration := map[string]any{"name": tool.Name, "parametersJsonSchema": googleSchema(schemaOrDefault(tool.Schema))}
			if tool.Description != "" {
				declaration["description"] = tool.Description
			}
			declarations = append(declarations, declaration)
		}
		output["tools"] = []any{map[string]any{"functionDeclarations": declarations}}
		if config := googleToolConfig(request.ToolChoice); config != nil {
			output["toolConfig"] = config
		}
	}

	generation := map[string]any{}
	jsonutil.Put(generation, "maxOutputTokens", request.MaxTokens)
	jsonutil.Put(generation, "temperature", request.Temperature)
	jsonutil.Put(generation, "topP", request.TopP)
	jsonutil.Put(generation, "stopSequences", googleStopSequences(request.Stop))
	jsonutil.Put(generation, "frequencyPenalty", request.FrequencyPenalty)
	jsonutil.Put(generation, "presencePenalty", request.PresencePenalty)
	jsonutil.Put(generation, "seed", request.Seed)
	googleResponseFormat(generation, request.ResponseFormat)
	if thinking := googleThinkingConfig(request.Reasoning); thinking != nil {
		generation["thinkingConfig"] = thinking
	}
	if len(generation) > 0 {
		output["generationConfig"] = generation
	}
	return output, nil
}

func googlePart(block bridgeBlock) (map[string]any, error) {
	switch block.Kind {
	case "text":
		if block.Text == "" {
			return nil, nil
		}
		return map[string]any{"text": block.Text}, nil
	case "image", "file":
		mediaType, data, err := googleInlineData(block)
		if err != nil {
			return nil, err
		}
		return map[string]any{"inlineData": map[string]any{"mimeType": mediaType, "data": data}}, nil
	default:
		return nil, nil
	}
}

// googleInlineData resolves a media block into the inline payload Gemini
// accepts. Gemini does not fetch remote URLs and cannot see OpenAI file IDs,
// so both are rejected with an explanation instead of being dropped.
func googleInlineData(block bridgeBlock) (string, string, error) {
	data, mediaType := block.Data, block.MediaType
	if data == "" && strings.HasPrefix(block.URL, "data:") {
		data = block.URL
	}
	if strings.HasPrefix(data, "data:") {
		header, payload, ok := strings.Cut(strings.TrimPrefix(data, "data:"), ",")
		media, isBase64 := strings.CutSuffix(header, ";base64")
		if !ok || !isBase64 || payload == "" {
			return "", "", fmt.Errorf("%s has an invalid data URI", block.Kind)
		}
		data = payload
		if media != "" {
			mediaType = media
		}
	}
	if data == "" {
		switch {
		case block.URL != "":
			return "", "", fmt.Errorf("Gemini does not fetch remote %s URLs; send the %s as base64 data", block.Kind, block.Kind)
		case block.FileID != "":
			return "", "", fmt.Errorf("file %q cannot be sent to Gemini without its data", block.FileID)
		default:
			return "", "", fmt.Errorf("%s has no data", block.Kind)
		}
	}
	if mediaType == "" && block.Filename != "" {
		mediaType, _, _ = strings.Cut(mime.TypeByExtension(filepath.Ext(block.Filename)), ";")
	}
	if mediaType == "" {
		return "", "", fmt.Errorf("%s requires a media type for Gemini", block.Kind)
	}
	return mediaType, data, nil
}

// googleArguments returns a call's arguments as the object Gemini requires.
func googleArguments(block bridgeBlock) map[string]any {
	value := block.Arguments
	if value == nil {
		value = jsonutil.ParseJSONOrString(block.ArgumentsJSON)
	}
	if object, ok := value.(map[string]any); ok {
		return object
	}
	return map[string]any{}
}

func googleFunctionResponse(block bridgeBlock, names map[string]string, omitIDs bool) (map[string]any, error) {
	name := names[block.CallID]
	if name == "" {
		return nil, fmt.Errorf("tool result %q has no matching tool call", block.CallID)
	}
	var text string
	var media []any
	if parts, ok := block.Result.([]bridgeBlock); ok {
		var builder strings.Builder
		for _, part := range parts {
			if part.Kind == "text" {
				if builder.Len() > 0 {
					builder.WriteString("\n")
				}
				builder.WriteString(part.Text)
				continue
			}
			mediaType, data, err := googleInlineData(part)
			if err != nil {
				return nil, err
			}
			media = append(media, map[string]any{"inlineData": map[string]any{"mimeType": mediaType, "data": data}})
		}
		text = builder.String()
	} else {
		text = bridgeToolResultContent(block.Result)
	}
	// Gemini documents "output" for a function's result and "error" for its
	// failure details.
	key := "output"
	if block.IsError {
		key = "error"
	}
	response := map[string]any{"name": name, "response": map[string]any{key: text}}
	if id, _ := splitGoogleToolID(block.CallID); !omitIDs && id != "" {
		response["id"] = id
	}
	if len(media) > 0 {
		response["parts"] = media
	}
	return map[string]any{"functionResponse": response}, nil
}

func googleToolConfig(choice bridgeToolChoice) any {
	switch choice.Mode {
	case "auto":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "AUTO"}}
	case "none":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "NONE"}}
	case "required":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY"}}
	case "named":
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "ANY", "allowedFunctionNames": []any{choice.Name}}}
	default:
		return nil
	}
}

func googleStopSequences(value any) any {
	switch typed := value.(type) {
	case string:
		if typed == "" {
			return nil
		}
		return []any{typed}
	case []any:
		if len(typed) == 0 {
			return nil
		}
		return typed
	default:
		return nil
	}
}

// googleResponseFormat maps a Chat or Responses response format onto Gemini's
// JSON mode. Plain text needs no configuration.
func googleResponseFormat(generation map[string]any, format any) {
	object, ok := format.(map[string]any)
	if !ok {
		return
	}
	switch jsonutil.StringAt(object, "type") {
	case "json_object":
		generation["responseMimeType"] = "application/json"
	case "json_schema":
		generation["responseMimeType"] = "application/json"
		if schema := jsonutil.FirstAny(jsonutil.AnyAt(object, "json_schema", "schema"), object["schema"]); schema != nil {
			generation["responseJsonSchema"] = googleSchema(schema)
		}
	}
}

// googleThinkingConfig maps a bridge-level reasoning value onto Gemini's
// thinkingLevel. Asking for reasoning also asks for the thought summaries, so
// clients see the reasoning they requested.
func googleThinkingConfig(reasoning any) map[string]any {
	if reasoning == nil {
		return nil
	}
	effort, ok := reasoningEffort(reasoning).(string)
	if !ok {
		return nil
	}
	level := googleThinkingLevel(effort)
	if level == "" {
		return nil
	}
	return map[string]any{"thinkingLevel": level, "includeThoughts": true}
}

// googleThinkingLevel folds the OpenAI/Anthropic effort ladder onto Gemini's,
// whose top rung is "high".
func googleThinkingLevel(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal":
		return "minimal"
	case "low":
		return "low"
	case "medium":
		return "medium"
	case "high", "xhigh", "max":
		return "high"
	default:
		return ""
	}
}

func applyGoogleForcedEffort(body map[string]any, effort string, disable bool) {
	generation, _ := body["generationConfig"].(map[string]any)
	if disable {
		if generation != nil {
			delete(generation, "thinkingConfig")
			if len(generation) == 0 {
				delete(body, "generationConfig")
			}
		}
		return
	}
	level := googleThinkingLevel(effort)
	if level == "" {
		return
	}
	if generation == nil {
		generation = map[string]any{}
		body["generationConfig"] = generation
	}
	thinking, _ := generation["thinkingConfig"].(map[string]any)
	if thinking == nil {
		thinking = map[string]any{}
		generation["thinkingConfig"] = thinking
	}
	// Gemini rejects a request that sets both a level and a budget.
	delete(thinking, "thinkingBudget")
	thinking["thinkingLevel"] = level
	thinking["includeThoughts"] = true
}

// FitGoogleRequest adapts a prepared Gemini body to what the OpenCode catalog
// declares for the model. A thinking level the model does not offer moves to
// the nearest one it does (downward first), and the temperature is dropped for
// a model that does not accept one. Empty inputs leave the body unchanged.
func FitGoogleRequest(body map[string]any, efforts []string, dropTemperature bool) {
	generation, _ := body["generationConfig"].(map[string]any)
	if generation == nil {
		return
	}
	if dropTemperature {
		delete(generation, "temperature")
	}
	if thinking, _ := generation["thinkingConfig"].(map[string]any); thinking != nil {
		if level, ok := thinking["thinkingLevel"].(string); ok {
			thinking["thinkingLevel"] = nearestGoogleThinkingLevel(level, efforts)
		}
	}
	if len(generation) == 0 {
		delete(body, "generationConfig")
	}
}

func nearestGoogleThinkingLevel(level string, supported []string) string {
	want := slices.Index(googleThinkingLevels, level)
	if want < 0 {
		return level
	}
	allowed := make([]bool, len(googleThinkingLevels))
	declared := false
	for _, value := range supported {
		if index := slices.Index(googleThinkingLevels, strings.ToLower(strings.TrimSpace(value))); index >= 0 {
			allowed[index] = true
			declared = true
		}
	}
	if !declared || allowed[want] {
		return level
	}
	for index := want - 1; index >= 0; index-- {
		if allowed[index] {
			return googleThinkingLevels[index]
		}
	}
	for index := want + 1; index < len(googleThinkingLevels); index++ {
		if allowed[index] {
			return googleThinkingLevels[index]
		}
	}
	return level
}

var (
	// googleSchemaMaps are keywords whose value maps names to subschemas.
	googleSchemaMaps = map[string]bool{"properties": true, "patternProperties": true, "$defs": true, "definitions": true, "dependentSchemas": true, "dependencies": true}
	// googleSchemaValues are keywords whose value is data, not a schema.
	googleSchemaValues = map[string]bool{"const": true, "default": true, "enum": true, "examples": true, "dependentRequired": true}
)

// googleSchema prepares a tool or response schema for parametersJsonSchema.
// Gemini accepts standard JSON Schema but rejects a few shapes that published
// schemas commonly contain; this rewrites only those, the same way OpenCode's
// own client does, and leaves everything else unchanged.
func googleSchema(schema any) any {
	normalized, ok := googleSchemaNode(schema).(map[string]any)
	if !ok {
		return map[string]any{}
	}
	for _, key := range []string{"$defs", "definitions"} {
		definitions, ok := normalized[key].(map[string]any)
		if !ok {
			continue
		}
		cut := make(map[string]any, len(definitions))
		for name, definition := range definitions {
			cut[name] = googleCutLoops(definition, "#/"+key+"/"+name, false)
		}
		normalized[key] = cut
	}
	result, ok := googleCutLoops(normalized, "#", false).(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return result
}

func googleSchemaNode(schema any) any {
	switch typed := schema.(type) {
	case []any:
		items := make([]any, len(typed))
		for i, item := range typed {
			items[i] = googleSchemaNode(item)
		}
		return items
	case map[string]any:
		properties, _ := typed["properties"].(map[string]any)
		output := make(map[string]any, len(typed))
		for key, value := range typed {
			if googleSchemaValues[key] {
				output[key] = value
				continue
			}
			if nested, ok := value.(map[string]any); ok && googleSchemaMaps[key] {
				mapped := make(map[string]any, len(nested))
				for name, child := range nested {
					mapped[name] = googleSchemaNode(child)
				}
				output[key] = mapped
				continue
			}
			switch key {
			case "required":
				// required may only name declared properties.
				if names, ok := value.([]any); ok && properties != nil {
					kept := make([]any, 0, len(names))
					for _, name := range names {
						if text, ok := name.(string); ok {
							if _, declared := properties[text]; declared {
								kept = append(kept, text)
							}
						}
					}
					output[key] = kept
					continue
				}
			case "exclusiveMinimum", "exclusiveMaximum":
				// Draft-04 boolean bounds become the numeric form.
				if exclusive, ok := value.(bool); ok {
					bound := "minimum"
					if key == "exclusiveMaximum" {
						bound = "maximum"
					}
					if number, isNumber := typed[bound].(float64); exclusive && isNumber {
						output[key] = number
					}
					continue
				}
			case "minimum", "maximum":
				if exclusive, _ := typed["exclusive"+strings.ToUpper(key[:1])+key[1:]].(bool); exclusive {
					continue
				}
			case "items":
				// Draft-07 tuples (an items array plus additionalItems) are
				// prefixItems plus items in 2020-12.
				if tuple, ok := value.([]any); ok {
					output["prefixItems"] = googleSchemaNode(tuple)
					continue
				}
			case "additionalItems":
				if _, tuple := typed["items"].([]any); tuple {
					output["items"] = googleSchemaNode(value)
					continue
				}
			}
			output[key] = googleSchemaNode(value)
		}
		return output
	default:
		return schema
	}
}

// googleCutLoops replaces a recursive $ref with an unconstrained schema unless
// the loop passes through an optional property or a possibly empty array, the
// only recursion Gemini accepts.
func googleCutLoops(schema any, target string, safe bool) any {
	switch typed := schema.(type) {
	case []any:
		items := make([]any, len(typed))
		for i, item := range typed {
			items[i] = googleCutLoops(item, target, safe)
		}
		return items
	case map[string]any:
		if ref, _ := typed["$ref"].(string); ref == target && !safe {
			return map[string]any{}
		}
		required, _ := typed["required"].([]any)
		output := make(map[string]any, len(typed))
		for key, value := range typed {
			switch {
			case googleSchemaValues[key]:
				output[key] = value
			case key == "items":
				output[key] = googleCutLoops(value, target, safe || jsonutil.IntAt(typed, "minItems") <= 0)
			case key == "properties":
				properties, ok := value.(map[string]any)
				if !ok {
					output[key] = googleCutLoops(value, target, safe)
					continue
				}
				mapped := make(map[string]any, len(properties))
				for name, child := range properties {
					mapped[name] = googleCutLoops(child, target, safe || !slices.Contains(required, any(name)))
				}
				output[key] = mapped
			default:
				if nested, ok := value.(map[string]any); ok && googleSchemaMaps[key] {
					mapped := make(map[string]any, len(nested))
					for name, child := range nested {
						mapped[name] = googleCutLoops(child, target, safe)
					}
					output[key] = mapped
					continue
				}
				output[key] = googleCutLoops(value, target, safe)
			}
		}
		return output
	default:
		return schema
	}
}

func decodeGoogleResponse(input map[string]any, response *bridgeResponse) error {
	if response.ID == "" {
		response.ID = jsonutil.StringAt(input, "responseId")
	}
	if response.Model == "" {
		response.Model = jsonutil.StringAt(input, "modelVersion")
	}
	if input["error"] != nil {
		response.Error = googleErrorMessage(input, "upstream Gemini request failed")
		return nil
	}
	response.Usage = decodeGoogleUsage(jsonutil.MapAt(input, "usageMetadata"))
	candidates := jsonutil.SliceAt(input, "candidates")
	if len(candidates) == 0 {
		if jsonutil.StringAt(input, "promptFeedback", "blockReason") != "" {
			response.Stop = "content_filter"
			return nil
		}
		return errors.New("Gemini response contains no candidates")
	}
	candidate, _ := candidates[0].(map[string]any)
	seen := make(map[string]bool)
	for _, raw := range jsonutil.SliceAt(candidate, "content", "parts") {
		part, _ := raw.(map[string]any)
		switch {
		case part == nil:
		case part["functionCall"] != nil:
			response.Tools = append(response.Tools, googleToolBlock(part, seen))
		case jsonutil.BoolAt(part, "thought"):
			if text := jsonutil.StringAt(part, "text"); text != "" {
				response.Reasoning = append(response.Reasoning, bridgeBlock{Kind: "reasoning", Text: text})
			}
		default:
			response.Text += jsonutil.StringAt(part, "text")
		}
	}
	stop, failure := googleStop(jsonutil.StringAt(candidate, "finishReason"), len(response.Tools) > 0)
	response.Stop = stop
	response.Error = failure
	return nil
}

// googleToolBlock decodes a functionCall part. Gemini 1.x omits call IDs and a
// repeated ID would replay as two identical calls, so both get a fresh one.
func googleToolBlock(part map[string]any, seen map[string]bool) bridgeBlock {
	call := jsonutil.MapAt(part, "functionCall")
	id := jsonutil.StringAt(call, "id")
	if id == "" || seen[id] {
		id = identity.RandomID("call", 12)
	}
	seen[id] = true
	arguments := call["args"]
	if arguments == nil {
		arguments = map[string]any{}
	}
	return bridgeBlock{
		Kind:      "tool_call",
		ID:        googleToolID(id, jsonutil.StringAt(part, "thoughtSignature")),
		Name:      jsonutil.StringAt(call, "name"),
		Arguments: arguments,
	}
}

func encodeGoogleResponse(response bridgeResponse) map[string]any {
	parts := make([]any, 0, len(response.Reasoning)+len(response.Tools)+1)
	for _, block := range response.Reasoning {
		if block.Text != "" {
			parts = append(parts, map[string]any{"text": block.Text, "thought": true})
		}
	}
	if response.Text != "" {
		parts = append(parts, map[string]any{"text": response.Text})
	}
	for _, tool := range response.Tools {
		id, signature := splitGoogleToolID(tool.ID)
		call := map[string]any{"name": tool.Name, "args": googleArguments(tool)}
		if id != "" {
			call["id"] = id
		}
		part := map[string]any{"functionCall": call}
		if signature != "" {
			part["thoughtSignature"] = signature
		}
		parts = append(parts, part)
	}
	return map[string]any{
		"responseId":   response.ID,
		"modelVersion": response.Model,
		"candidates": []any{map[string]any{
			"index":        0,
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": googleFinishReason(response.Stop),
		}},
		"usageMetadata": googleUsage(response.Usage),
	}
}

// googleStop maps a Gemini finish reason onto a canonical bridge stop. Gemini
// reports STOP for a turn that ended in function calls, so the calls decide.
// The second value describes a finish that means the response failed.
func googleStop(reason string, tools bool) (string, string) {
	switch reason {
	case "MAX_TOKENS":
		return "length", ""
	case "CONTINUATION":
		// Gemini can split a long response across requests and finishes each
		// part with CONTINUATION. The gateway does not resume it, so the turn
		// is reported as cut short, the same as an output cap.
		return "length", ""
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MODEL_ARMOR", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "LANGUAGE":
		return "content_filter", ""
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE", "MALFORMED_RESPONSE", "NO_IMAGE":
		return "error", "Gemini stopped with " + reason
	}
	if tools {
		return "tool_calls", ""
	}
	return "stop", ""
}

func googleFinishReason(stop string) string {
	switch stop {
	case "length", "max_tokens":
		return "MAX_TOKENS"
	case "content_filter":
		return "SAFETY"
	default:
		return "STOP"
	}
}

// decodeGoogleUsage reads usageMetadata. promptTokenCount already includes the
// cached tokens, while candidatesTokenCount excludes the thoughts, so the two
// are summed into the inclusive output the bridge uses.
func decodeGoogleUsage(usage map[string]any) Usage {
	input := jsonutil.IntAt(usage, "promptTokenCount") + jsonutil.IntAt(usage, "toolUsePromptTokenCount")
	thoughts := jsonutil.IntAt(usage, "thoughtsTokenCount")
	output := jsonutil.IntAt(usage, "candidatesTokenCount") + thoughts
	total := jsonutil.IntAt(usage, "totalTokenCount")
	if total == 0 {
		total = input + output
	}
	return Usage{Input: input, Output: output, Total: total, Cached: jsonutil.IntAt(usage, "cachedContentTokenCount"), Reasoning: thoughts}
}

func googleUsage(usage Usage) map[string]any {
	return map[string]any{
		"promptTokenCount":        usage.Input,
		"candidatesTokenCount":    max(usage.Output-usage.Reasoning, 0),
		"thoughtsTokenCount":      usage.Reasoning,
		"cachedContentTokenCount": usage.Cached,
		"totalTokenCount":         usage.Total,
	}
}

// googleUsageReported tells a real count from the first chunk's metadata,
// which carries only the traffic type.
func googleUsageReported(usage map[string]any) bool {
	for _, key := range []string{"promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount"} {
		if jsonutil.IntAt(usage, key) > 0 {
			return true
		}
	}
	return false
}

func googleErrorMessage(value map[string]any, fallback string) string {
	if message := jsonutil.StringAt(value, "error", "message"); message != "" {
		return message
	}
	if status := jsonutil.StringAt(value, "error", "status"); status != "" {
		return status
	}
	return fallback
}

// parseGoogle reads one streamGenerateContent chunk. Gemini sends no terminal
// event: the chunk carrying finishReason is the last one, and it also carries
// the final usage, so it ends the stream.
func (parser *bridgeStreamParser) parseGoogle(value map[string]any) []bridgeStreamEvent {
	if value["error"] != nil {
		return []bridgeStreamEvent{{
			Kind:      "error",
			Error:     googleErrorMessage(value, "upstream Gemini stream error"),
			ErrorType: jsonutil.FirstString(strings.ToLower(jsonutil.StringAt(value, "error", "status")), "upstream_error"),
		}}
	}
	events := make([]bridgeStreamEvent, 0, 4)
	if !parser.started {
		parser.started = true
		events = append(events, bridgeStreamEvent{Kind: "start", ResponseID: jsonutil.StringAt(value, "responseId")})
	}
	if usageMap := jsonutil.MapAt(value, "usageMetadata"); googleUsageReported(usageMap) {
		usage := decodeGoogleUsage(usageMap)
		events = append(events, bridgeStreamEvent{Kind: "usage", Usage: &usage})
	}
	candidates := jsonutil.SliceAt(value, "candidates")
	if len(candidates) == 0 {
		if jsonutil.StringAt(value, "promptFeedback", "blockReason") != "" {
			events = append(events, bridgeStreamEvent{Kind: "finish", Stop: "content_filter"}, bridgeStreamEvent{Kind: "done"})
		}
		return events
	}
	candidate, _ := candidates[0].(map[string]any)
	if parser.googleCallIDs == nil {
		parser.googleCallIDs = make(map[string]bool)
	}
	for _, raw := range jsonutil.SliceAt(candidate, "content", "parts") {
		part, _ := raw.(map[string]any)
		if part == nil {
			continue
		}
		if part["functionCall"] != nil {
			block := googleToolBlock(part, parser.googleCallIDs)
			key := fmt.Sprintf("google-%d", len(parser.toolOrder))
			parser.rememberTool(key, block.ID, block.Name)
			parser.tools[key] = true
			events = append(events,
				bridgeStreamEvent{Kind: "tool_start", ToolKey: key, ToolID: block.ID, ToolName: block.Name},
				bridgeStreamEvent{Kind: "tool_delta", ToolKey: key, ToolID: block.ID, ToolName: block.Name, Text: bridgeArgumentsJSON(block)},
			)
			continue
		}
		text := jsonutil.StringAt(part, "text")
		if text == "" {
			continue
		}
		if jsonutil.BoolAt(part, "thought") {
			events = append(events, bridgeStreamEvent{Kind: "reasoning", Text: text})
		} else {
			events = append(events, bridgeStreamEvent{Kind: "text", Text: text})
		}
	}
	if reason := jsonutil.StringAt(candidate, "finishReason"); reason != "" {
		stop, failure := googleStop(reason, len(parser.toolOrder) > 0)
		if failure != "" {
			return append(events, bridgeStreamEvent{Kind: "error", Error: failure, ErrorType: "upstream_error"})
		}
		events = append(events, bridgeStreamEvent{Kind: "finish", Stop: stop}, bridgeStreamEvent{Kind: "done"})
	}
	return events
}
