package antigravity

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNeutralizeAgentIdentities(t *testing.T) {
	tests := []struct {
		name string
		text string
		want string
	}{
		{
			name: "DeepSeek Harness identity",
			text: "You are an AI agent powered by DeepSeek Harness.",
			want: "You are an AI agent.",
		},
		{
			name: "DeepSeek Harness identity followed by instructions",
			text: "You are an AI agent powered by DeepSeek Harness.\n\nYou must be concise. Preserve user code style.",
			want: "You are an AI agent.\n\nYou must be concise. Preserve user code style.",
		},
		{
			name: "ZCode identity",
			text: "You are ZCode, an interactive agent that helps developers write code.",
			want: "You are an AI agent that helps developers write code.",
		},
		{
			name: "ZCode identity with trailing period",
			text: "You are ZCode, an interactive agent.\nFollow repository conventions.",
			want: "You are an AI agent.\nFollow repository conventions.",
		},
		{
			name: "generic branded agent keeps its description",
			text: "You are Cline, an autonomous coding agent built for VS Code. Always confirm destructive commands.",
			want: "You are an AI agent built for VS Code. Always confirm destructive commands.",
		},
		{
			name: "ordinary role prompt is untouched",
			text: "You are a helpful assistant.",
			want: "You are a helpful assistant.",
		},
		{
			name: "expert prompt is untouched",
			text: "You are an expert Go developer. Answer concisely.",
			want: "You are an expert Go developer. Answer concisely.",
		},
		{
			name: "vendor mention mid-text is untouched",
			text: "Explain how ZCode and DeepSeek Harness work.",
			want: "Explain how ZCode and DeepSeek Harness work.",
		},
		{
			name: "Antigravity identity is preserved",
			text: "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team.",
			want: "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team.",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, NeutralizeSystemIdentity(tt.text))
		})
	}
}

// DSH injects bootstrap notes about its own web runtime. Those lines describe the
// client's implementation, not the user's task, and are a non-official-client
// signal for upstream; the surrounding real instructions must survive.
func TestNeutralizeSystemIdentity_StripsDSHWebRuntimeContext(t *testing.T) {
	text := "You are an AI agent powered by DeepSeek Harness.\n\n" +
		"# DSH Web\n\n" +
		"The UI is served by the local runtime. Boot data is injected as window.__DSH_BOOT__ before the app renders.\n\n" +
		"Start the web UI with:\n\n" +
		"```bash\npnpm run dev:web\n```\n\n" +
		"Or launch the shell:\n\n" +
		"dsh web\n\n" +
		"Always read AGENTS.md before editing. Preserve the user's existing style."

	want := "You are an AI agent.\n\n" +
		"# DSH Web\n\n" +
		"Start the web UI with:\n\n" +
		"Or launch the shell:\n\n" +
		"Always read AGENTS.md before editing. Preserve the user's existing style."

	require.Equal(t, want, NeutralizeSystemIdentity(text))
}

// End-to-end guard for the Claude→Gemini transform: an agent's branded system
// blocks (top-level system + role=="system" messages) must be neutralized while
// every tool declaration and the user turn survive untouched. This is the shape
// ZCode / DSH / Claude Code send through the Anthropic-protocol endpoints.
func TestTransformClaudeToGeminiNeutralizesAgentIdentity(t *testing.T) {
	req := &ClaudeRequest{
		Model: "gemini-3.8-flash-high",
		System: json.RawMessage(`[
			{"type":"text","text":"You are ZCode, an interactive agent that helps developers write code."},
			{"type":"text","text":"You are an AI agent powered by DeepSeek Harness."},
			{"type":"text","text":"Review the tool result before answering."}
		]`),
		Messages: []ClaudeMessage{
			{Role: "system", Content: json.RawMessage(`"You are an AI agent powered by DeepSeek Harness."`)},
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
		Tools: []ClaudeTool{
			{Name: "Read", Description: "read a file", InputSchema: map[string]any{
				"$schema":              "https://json-schema.org/draft/2020-12/schema",
				"type":                 "object",
				"additionalProperties": false,
				"properties":           map[string]any{"path": map[string]any{"type": "string"}},
				"required":             []any{"path"},
			}},
			{Name: "Bash", Description: "run a command", InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"command": map[string]any{"type": "string"}},
			}},
		},
	}

	body, err := TransformClaudeToGeminiWithOptions(req, "test-project", "gemini-3.8-flash-high", DefaultTransformOptions())
	require.NoError(t, err)

	out := string(body)
	require.NotContains(t, out, "ZCode", "client identity must not reach upstream")
	require.NotContains(t, out, "DeepSeek Harness")
	require.Contains(t, out, "You are Antigravity", "official identity preamble is injected")
	require.Contains(t, out, `"name":"Read"`, "tool declarations must survive")
	require.Contains(t, out, `"name":"Bash"`)
	require.Contains(t, out, "Review the tool result before answering.", "real user instructions must survive")

	var parsed V1InternalRequest
	require.NoError(t, json.Unmarshal(body, &parsed))
	require.NotNil(t, parsed.Request.SystemInstruction)
	for _, part := range parsed.Request.SystemInstruction.Parts {
		require.NotEmpty(t, strings.TrimSpace(part.Text), "empty system parts are rejected upstream")
	}
}

func TestNeutralizeSystemParts(t *testing.T) {
	parts := []GeminiPart{
		{Text: "You are an AI agent powered by DeepSeek Harness."},
		{Text: "```\npnpm run dev:web\n```"},
		{Text: "Keep this instruction."},
		{Text: "", Thought: true},
	}
	got := NeutralizeSystemParts(parts)
	require.Len(t, got, 2)
	require.Equal(t, "You are an AI agent.", got[0].Text)
	require.Equal(t, "Keep this instruction.", got[1].Text)
	require.True(t, parts[0].Text != got[0].Text, "input slice must not be mutated")
}
