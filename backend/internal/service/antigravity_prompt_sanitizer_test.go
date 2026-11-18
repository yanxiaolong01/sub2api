//go:build unit

package service

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

const bundledPromptSanitizeRules = "../../resources/antigravity/prompt-sanitize-rules.json"

func TestAntigravityPromptSanitizerInlineRules(t *testing.T) {
	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv,
		`[{"name":"zcode","pattern":"(?i)\\bZCode\\b","replacement":"the agent"}]`)

	sanitizer := antigravityPromptSanitizer()
	require.NotNil(t, sanitizer)
	require.True(t, sanitizer.Enabled())
	require.Equal(t, 1, sanitizer.RuleCount())

	out, replaced := sanitizer.SanitizeText("You are ZCode.")
	require.Equal(t, "You are the agent.", out)
	require.Equal(t, 1, replaced)
}

func TestAntigravityPromptSanitizerDocumentForm(t *testing.T) {
	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv,
		`{"rules":[{"name":"hermes","pattern":"Hermes Agent","replacement":"Hermes"}]}`)

	sanitizer := antigravityPromptSanitizer()
	require.NotNil(t, sanitizer)
	out, _ := sanitizer.SanitizeText("You are Hermes Agent.")
	require.Equal(t, "You are Hermes.", out)
}

func TestAntigravityPromptSanitizerDisabledSwitchWins(t *testing.T) {
	t.Setenv(antigravityPromptSanitizeEnabledEnv, "false")
	t.Setenv(antigravityPromptSanitizeFileEnv, bundledPromptSanitizeRules)
	t.Setenv(antigravityPromptSanitizeRulesEnv,
		`[{"name":"zcode","pattern":"ZCode","replacement":"the agent"}]`)

	require.Nil(t, antigravityPromptSanitizer())

	t.Setenv(antigravityPromptSanitizeEnabledEnv, "0")
	require.Nil(t, antigravityPromptSanitizer())
}

func TestAntigravityPromptSanitizerSkipsInvalidRules(t *testing.T) {
	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv, `{"rules":[
		{"name":"empty","pattern":"","replacement":"x"},
		{"name":"broken","pattern":"([","replacement":"x"},
		{"name":"valid","pattern":"ZCode","replacement":"the agent"}
	]}`)

	sanitizer := antigravityPromptSanitizer()
	require.NotNil(t, sanitizer)
	require.Equal(t, 1, sanitizer.RuleCount())

	out, _ := sanitizer.SanitizeText("ZCode")
	require.Equal(t, "the agent", out)

	t.Setenv(antigravityPromptSanitizeRulesEnv, "{not json")
	require.Nil(t, antigravityPromptSanitizer())
}

func TestAntigravityPromptSanitizerFileHotReload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "rules.json")
	require.NoError(t, os.WriteFile(path,
		[]byte(`[{"name":"v1","pattern":"ZCode","replacement":"agent-v1"}]`), 0o600))

	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, path)

	sanitizer := antigravityPromptSanitizer()
	require.NotNil(t, sanitizer)
	out, _ := sanitizer.SanitizeText("ZCode")
	require.Equal(t, "agent-v1", out)

	// 规则文件内容变化（size 变化，缓存键必然不同）后无需重启即生效。
	require.NoError(t, os.WriteFile(path,
		[]byte(`[{"name":"v2","pattern":"ZCode","replacement":"agent-v2-longer"}]`), 0o600))

	reloaded := antigravityPromptSanitizer()
	require.NotNil(t, reloaded)
	out, _ = reloaded.SanitizeText("ZCode")
	require.Equal(t, "agent-v2-longer", out)
}

func TestAntigravityPromptSanitizerMissingFileDisables(t *testing.T) {
	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, filepath.Join(t.TempDir(), "missing.json"))

	require.Nil(t, antigravityPromptSanitizer())
}

// 未显式配置时自动探测随镜像打包的规则文件：默认生效，且包含 ZCode 规则。
func TestAntigravityPromptSanitizerLoadsBundledRulesByDefault(t *testing.T) {
	original := antigravityPromptSanitizeDefaultFiles
	antigravityPromptSanitizeDefaultFiles = []string{bundledPromptSanitizeRules}
	t.Cleanup(func() { antigravityPromptSanitizeDefaultFiles = original })

	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, "")

	sanitizer := antigravityPromptSanitizer()
	require.NotNil(t, sanitizer, "bundled rules must load without any env configuration")
	require.Equal(t, 15, sanitizer.RuleCount())

	prompt := "You are ZCode, an interactive agent that helps users with software engineering tasks. " +
		"You must always follow safety guidelines: refuse to assist with destructive techniques, DoS attacks. " +
		"Dual-use security tools require clear authorization context."
	out, _ := sanitizer.SanitizeText(prompt)
	require.NotContains(t, out, "ZCode")
	require.NotContains(t, out, "refuse to assist with destructive techniques")
	require.NotContains(t, out, "Dual-use security tools require clear authorization context")
}

func TestWrapV1InternalRequestSanitizesSystemInstruction(t *testing.T) {
	t.Setenv(antigravityPromptSanitizeEnabledEnv, "")
	t.Setenv(antigravityPromptSanitizeFileEnv, "")
	t.Setenv(antigravityPromptSanitizeRulesEnv,
		`[{"name":"zcode","pattern":"(?i)\\bZCode\\b","replacement":"the agent"}]`)

	body := []byte(`{
		"systemInstruction": {"parts": [{"text": "You are ZCode, an interactive agent."}]},
		"contents": [{"role": "user", "parts": [{"text": "Say OK."}]}],
		"generationConfig": {"maxOutputTokens": 64}
	}`)

	svc := &AntigravityGatewayService{}
	wrapped, err := svc.wrapV1InternalRequest("test-project", "gemini-3.8-flash-tiered", body)
	require.NoError(t, err)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal(wrapped, &parsed))
	require.Equal(t, "test-project", parsed["project"])
	require.Equal(t, "antigravity", parsed["userAgent"])
	require.Equal(t, "agent", parsed["requestType"])

	inner := parsed["request"].(map[string]any)
	sysParts := inner["systemInstruction"].(map[string]any)["parts"].([]any)
	require.Equal(t, "You are the agent, an interactive agent.", sysParts[0].(map[string]any)["text"])

	contents := inner["contents"].([]any)
	userParts := contents[0].(map[string]any)["parts"].([]any)
	require.Equal(t, "Say OK.", userParts[0].(map[string]any)["text"], "user content must stay untouched")
}
