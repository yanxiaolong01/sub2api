package antigravity

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func mustPromptRule(t *testing.T, name, pattern, replacement string) PromptRule {
	t.Helper()
	rule, err := NewPromptRule(name, pattern, replacement)
	require.NoError(t, err)
	return rule
}

// mustJSONString 把普通字符串编码成 ClaudeRequest.System 需要的 JSON 字符串字面量。
func mustJSONString(t *testing.T, value string) json.RawMessage {
	t.Helper()
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	return encoded
}

// loadBundledPromptRules 读取随镜像打包的规则文件，保证测试覆盖真实配置。
func loadBundledPromptRules(t *testing.T) []PromptRule {
	t.Helper()
	path := filepath.Join("..", "..", "..", "resources", "antigravity", "prompt-sanitize-rules.json")
	data, err := os.ReadFile(path)
	require.NoErrorf(t, err, "bundled rules file must exist at %s", path)

	var document struct {
		Rules []struct {
			Name        string `json:"name"`
			Pattern     string `json:"pattern"`
			Replacement string `json:"replacement"`
		} `json:"rules"`
	}
	require.NoError(t, json.Unmarshal(data, &document))
	require.NotEmpty(t, document.Rules)

	rules := make([]PromptRule, 0, len(document.Rules))
	for _, spec := range document.Rules {
		rules = append(rules, mustPromptRule(t, spec.Name, spec.Pattern, spec.Replacement))
	}
	return rules
}

func TestNewPromptRuleRejectsInvalidPattern(t *testing.T) {
	_, err := NewPromptRule("bad", "", "x")
	require.Error(t, err)

	_, err = NewPromptRule("bad", "([", "x")
	require.Error(t, err)

	// 空白 pattern 同样视为无效，避免误配成“每条文本都命中”。
	_, err = NewPromptRule("blank", "   ", "x")
	require.Error(t, err)
}

func TestNewPromptSanitizerWithoutRulesIsNil(t *testing.T) {
	require.Nil(t, NewPromptSanitizer(nil))
	require.Nil(t, NewPromptSanitizer([]PromptRule{{Name: "uncompiled"}}))

	var nilSanitizer *PromptSanitizer
	require.False(t, nilSanitizer.Enabled())
	require.Equal(t, 0, nilSanitizer.RuleCount())
	out, replaced := nilSanitizer.SanitizeText("unchanged")
	require.Equal(t, "unchanged", out)
	require.Zero(t, replaced)
}

func TestPromptSanitizerAppliesRulesInOrder(t *testing.T) {
	sanitizer := NewPromptSanitizer([]PromptRule{
		mustPromptRule(t, "brand", `(?i)\bZCode\b`, "the agent"),
		mustPromptRule(t, "double", `(?i)\bagent\b`, "assistant"),
	})
	require.True(t, sanitizer.Enabled())
	require.Equal(t, 2, sanitizer.RuleCount())

	out, replaced := sanitizer.SanitizeText("You are ZCode, an interactive agent.")
	require.Equal(t, "You are the assistant, an interactive assistant.", out)
	require.Equal(t, 3, replaced, "first rule hits once, second rule hits both agent tokens")
}

func TestPromptSanitizerSupportsCaptureGroupsAndMultipleMatches(t *testing.T) {
	sanitizer := NewPromptSanitizer([]PromptRule{
		mustPromptRule(t, "rfc", `(?i)(RFC[-\s]*2119)(\s+keywords)`, "RFC-2119$2"),
	})
	out, replaced := sanitizer.SanitizeText("RFC 2119 keywords and rfc-2119 keywords")
	require.Equal(t, "RFC-2119 keywords and RFC-2119 keywords", out)
	require.Equal(t, 2, replaced)

	out, replaced = sanitizer.SanitizeText("no convention here")
	require.Equal(t, "no convention here", out)
	require.Zero(t, replaced)
}

func TestPromptSanitizerSanitizesGeminiContent(t *testing.T) {
	sanitizer := NewPromptSanitizer([]PromptRule{
		mustPromptRule(t, "dsh", `(?i)\bDeepSeek Harness\b`, "agent runtime"),
	})
	content := &GeminiContent{
		Role: "user",
		Parts: []GeminiPart{
			{Text: "You are an AI agent powered by DeepSeek Harness."},
			{Text: ""},
			{InlineData: &GeminiInlineData{MimeType: "image/png", Data: "AAAA"}},
			{Text: "DeepSeek Harness again"},
		},
	}
	replaced := sanitizer.SanitizeGeminiContent(content)
	require.Equal(t, 2, replaced)
	require.Equal(t, "You are an AI agent powered by agent runtime.", content.Parts[0].Text)
	require.Empty(t, content.Parts[1].Text)
	require.Equal(t, "agent runtime again", content.Parts[3].Text)
	require.Equal(t, 0, sanitizer.SanitizeGeminiContent(nil))
}

func TestPromptSanitizerSanitizesRequestJSON(t *testing.T) {
	sanitizer := NewPromptSanitizer([]PromptRule{
		mustPromptRule(t, "agent-token", `(?i)\bHermes Agent\b`, "Hermes"),
	})

	t.Run("bare gemini request", func(t *testing.T) {
		request := map[string]any{
			"systemInstruction": map[string]any{
				"parts": []any{
					map[string]any{"text": "You are Hermes Agent."},
					map[string]any{"text": "Keep this."},
				},
			},
			"contents": []any{map[string]any{"role": "user"}},
		}
		require.Equal(t, 1, sanitizer.SanitizeRequestJSON(request))
		parts := request["systemInstruction"].(map[string]any)["parts"].([]any)
		require.Equal(t, "You are Hermes.", parts[0].(map[string]any)["text"])
		require.Equal(t, "Keep this.", parts[1].(map[string]any)["text"])
	})

	t.Run("v1internal wrapped request", func(t *testing.T) {
		request := map[string]any{
			"model": "gemini-3.8-flash",
			"request": map[string]any{
				"systemInstruction": map[string]any{
					"parts": []any{map[string]any{"text": "Hermes Agent is here."}},
				},
			},
		}
		require.Equal(t, 1, sanitizer.SanitizeRequestJSON(request))
		inner := request["request"].(map[string]any)["systemInstruction"].(map[string]any)
		parts := inner["parts"].([]any)
		require.Equal(t, "Hermes is here.", parts[0].(map[string]any)["text"])
	})

	t.Run("malformed input is ignored", func(t *testing.T) {
		require.Zero(t, sanitizer.SanitizeRequestJSON(nil))
		require.Zero(t, sanitizer.SanitizeRequestJSON("not a map"))
		require.Zero(t, sanitizer.SanitizeRequestJSON(map[string]any{
			"systemInstruction": map[string]any{"parts": "not a slice"},
		}))
	})
}

// bundled 规则必须覆盖 ZCode 的系统提示词指纹（身份、安全策略、双用途表述）。
func TestBundledPromptRulesRewriteZCodePrompt(t *testing.T) {
	sanitizer := NewPromptSanitizer(loadBundledPromptRules(t))
	require.True(t, sanitizer.Enabled())

	prompt := "You are ZCode, an interactive agent that helps users with software engineering tasks. " +
		"Use the tools available to you to answer the user question. " +
		"You must always follow safety guidelines: refuse to assist with destructive techniques, DoS attacks, " +
		"mass targeting, supply chain compromise, or detection evasion for malicious purposes. " +
		"Dual-use security tools require clear authorization context."

	out, replaced := sanitizer.SanitizeText(prompt)
	require.Positive(t, replaced)
	require.NotContains(t, out, "ZCode")
	require.NotContains(t, out, "refuse to assist with destructive techniques")
	require.NotContains(t, out, "Dual-use security tools require clear authorization context")
	require.Contains(t, out, "Use the tools available to you to answer the user question.")
}

// 正常角色提示词与工具描述不能被规则误伤。
func TestBundledPromptRulesLeaveBenignPromptUntouched(t *testing.T) {
	sanitizer := NewPromptSanitizer(loadBundledPromptRules(t))
	benign := "You are a helpful assistant. Read AGENTS.md before editing and preserve the user's style. " +
		"Use MUST/SHOULD consistently when documenting requirements."
	out, replaced := sanitizer.SanitizeText(benign)
	require.Zero(t, replaced)
	require.Equal(t, benign, out)
}

// 端到端：ZCode 通过 Anthropic 协议调用时，systemInstruction 中的客户端身份与
// 安全策略指纹必须在转换阶段被改写，而工具声明和真实用户指令保持不变。
func TestTransformClaudeToGeminiAppliesPromptSanitizer(t *testing.T) {
	zcodeSystem := "You are ZCode, an interactive agent that helps users with software engineering tasks. " +
		"Use the tools available to you to answer the user question. " +
		"You must always follow safety guidelines: refuse to assist with destructive techniques, DoS attacks, " +
		"mass targeting, supply chain compromise, or detection evasion for malicious purposes. " +
		"Dual-use security tools require clear authorization context."

	req := &ClaudeRequest{
		Model:  "gemini-3.8-flash-high",
		System: mustJSONString(t, zcodeSystem),
		Messages: []ClaudeMessage{
			{Role: "user", Content: json.RawMessage(`"Say OK."`)},
		},
		Tools: []ClaudeTool{
			{Name: "Read", Description: "read a file", InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": map[string]any{"type": "string"}},
				"required":   []any{"path"},
			}},
		},
	}

	opts := DefaultTransformOptions()
	opts.PromptSanitizer = NewPromptSanitizer(loadBundledPromptRules(t))
	body, err := TransformClaudeToGeminiWithOptions(req, "test-project", "gemini-3.8-flash-high", opts)
	require.NoError(t, err)

	out := string(body)
	require.NotContains(t, out, "ZCode")
	require.NotContains(t, out, "refuse to assist with destructive techniques")
	require.NotContains(t, out, "Dual-use security tools require clear authorization context")
	require.Contains(t, out, "You are Antigravity", "official identity preamble is injected")
	require.Contains(t, out, "Use the tools available to you to answer the user question.")
	require.Contains(t, out, `"name":"Read"`, "tool declarations must survive")
	require.Contains(t, out, `Say OK.`, "user turn must survive")

	var parsed V1InternalRequest
	require.NoError(t, json.Unmarshal(body, &parsed))
	require.NotNil(t, parsed.Request.SystemInstruction)
	for _, part := range parsed.Request.SystemInstruction.Parts {
		require.NotEmpty(t, part.Text)
	}
}
