//go:build unit

package service

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

// The compat / native Gemini paths build systemInstruction themselves, so the
// identity neutralization happens inside injectIdentityPatchToGeminiRequest.
// A part that consists only of client runtime context becomes empty and must be
// dropped: upstream rejects empty text parts.
func TestInjectIdentityPatchToGeminiRequest_DropsEmptyNeutralizedParts(t *testing.T) {
	body := []byte(`{
		"contents": [{"role": "user", "parts": [{"text": "hi"}]}],
		"systemInstruction": {"parts": [
			{"text": "You are an AI agent powered by DeepSeek Harness."},
			{"text": "Boot data is injected as window.__DSH_BOOT__ before the app renders."},
			{"text": "Keep this rule."}
		]}
	}`)

	got, err := injectIdentityPatchToGeminiRequest(body)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(got, &payload))
	sysInst := payload["systemInstruction"].(map[string]any)
	parts := sysInst["parts"].([]any)
	require.Len(t, parts, 3, "runtime-only part must be dropped")

	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		texts = append(texts, part.(map[string]any)["text"].(string))
	}
	require.Contains(t, texts[0], "You are Antigravity", "official identity preamble is prepended")
	require.Equal(t, "You are an AI agent.", texts[1])
	require.Equal(t, "Keep this rule.", texts[2])
}
