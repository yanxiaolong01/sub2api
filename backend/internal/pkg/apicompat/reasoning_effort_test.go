package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeReasoningEffort(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"low", "low"},
		{"Medium", "medium"},
		{"High", "high"},
		{"HIGH ", "high"},
		{"XHIGH", "xhigh"},
		{"extra-high", "xhigh"},
		{"extra_high", "xhigh"},
		{"maximal", "max"},
		{"Maximum", "max"},
		{"none", "none"},
		{"", ""},
		// Unknown tokens keep their lowercased form so behavior is unchanged.
		{"turbo", "turbo"},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			require.Equal(t, tt.want, normalizeReasoningEffort(tt.in))
		})
	}
}

func TestMapResponsesEffortToAnthropic(t *testing.T) {
	require.Equal(t, "high", mapResponsesEffortToAnthropic("High"))
	// "EXTRA-HIGH" 归一化为 xhigh，与 "XHIGH" 一样映射到 Anthropic 的 max。
	require.Equal(t, "max", mapResponsesEffortToAnthropic("EXTRA-HIGH"))
	require.Equal(t, "max", mapResponsesEffortToAnthropic("XHIGH"))
	require.Equal(t, "max", mapResponsesEffortToAnthropic("maximal"))
}

// ZCode sends reasoning.effort with a capital H; the conversion must not leak the
// raw token into Anthropic's output_config where only lowercase values are valid.
func TestResponsesToAnthropicRequest_CapitalizedEffort(t *testing.T) {
	req := &ResponsesRequest{
		Model:     "claude-sonnet-4-5",
		Input:     json.RawMessage(`"hi"`),
		Reasoning: &ResponsesReasoning{Effort: "High"},
	}
	out, err := ResponsesToAnthropicRequest(req)
	require.NoError(t, err)
	require.NotNil(t, out.OutputConfig)
	require.Equal(t, "high", out.OutputConfig.Effort)
	require.NotNil(t, out.Thinking)
	require.Equal(t, "enabled", out.Thinking.Type)
	require.Equal(t, 10240, out.Thinking.BudgetTokens)
}
