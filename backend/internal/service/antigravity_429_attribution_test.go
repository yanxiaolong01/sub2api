//go:build unit

package service

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

// Antigravity returns a bare RESOURCE_EXHAUSTED 429 when the request content itself is
// rejected (client identity / runtime context / safety policy). That body carries no
// details[] and no quota keyword, so it cannot be attributed to the account's budget.
func TestClassifyAntigravity429Attribution(t *testing.T) {
	tests := []struct {
		name         string
		body         string
		attributable bool
		familyWide   bool
		reason       string
	}{
		{
			name:         "bare resource exhausted is not attributable",
			body:         `{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`,
			attributable: false,
			reason:       "unattributable",
		},
		{
			name:         "non json body is not attributable",
			body:         `<html>429</html>`,
			attributable: false,
			reason:       "unattributable",
		},
		{
			name:         "error info with model cools only that model",
			body:         `{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"RATE_LIMIT_EXCEEDED","metadata":{"model":"gemini-3.8-flash"}},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}]}}`,
			attributable: true,
			familyWide:   false,
			reason:       "model_rate_limit",
		},
		{
			name:         "structured details without model are account wide",
			body:         `{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}]}}`,
			attributable: true,
			familyWide:   true,
			reason:       "account_rate_limit",
		},
		{
			name:         "quota keyword is account wide",
			body:         `{"error":{"code":429,"message":"QUOTA_EXHAUSTED","status":"RESOURCE_EXHAUSTED"}}`,
			attributable: true,
			familyWide:   true,
			reason:       "quota_exhausted",
		},
		{
			name:         "unknown detail type is not attributable",
			body:         `{"error":{"code":429,"message":"denied","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.LocalizedMessage","message":"no"}]}}`,
			attributable: false,
			reason:       "unattributable",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyAntigravity429Attribution([]byte(tt.body))
			require.Equal(t, tt.attributable, got.Attributable)
			require.Equal(t, tt.familyWide, got.FamilyWide)
			require.Equal(t, tt.reason, got.Reason)
		})
	}
}

func TestMarkAntigravityRequestScoped429(t *testing.T) {
	body := []byte(`{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`)

	marked := markAntigravityRequestScoped429(&UpstreamFailoverError{StatusCode: http.StatusTooManyRequests}, body)
	require.NotNil(t, marked)
	require.True(t, marked.RequestScopedTransient, "handler must not temp-ban the account")
	require.True(t, marked.RetryableOnSameAccount)
	require.Equal(t, 2, marked.SameAccountRetryMax)

	attributable := markAntigravityRequestScoped429(&UpstreamFailoverError{StatusCode: http.StatusTooManyRequests}, []byte(`{"error":{"code":429,"message":"QUOTA_EXHAUSTED"}}`))
	require.False(t, attributable.RequestScopedTransient)
	require.False(t, attributable.RetryableOnSameAccount)

	not429 := markAntigravityRequestScoped429(&UpstreamFailoverError{StatusCode: http.StatusBadRequest}, body)
	require.False(t, not429.RequestScopedTransient)
	require.Nil(t, markAntigravityRequestScoped429(nil, body))
}

// A bare 429 must not write any cooldown: one bad request cannot be allowed to take the
// whole account (and therefore the whole Gemini family) out of rotation.
func TestHandleUpstreamError_Bare429SkipsAccountCooldown(t *testing.T) {
	repo := &stubAntigravityAccountRepo{}
	svc := &AntigravityGatewayService{accountRepo: repo}
	account := &Account{ID: 7, Name: "ag", Platform: PlatformAntigravity}

	body := []byte(`{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`)
	result := svc.handleUpstreamError(context.Background(), "[test]", account, http.StatusTooManyRequests, http.Header{}, body, "gemini-3.8-flash-high", 0, "", false)

	require.Nil(t, result)
	require.Empty(t, repo.modelRateLimitCalls, "bare 429 must not cool any model")
	require.Empty(t, repo.rateCalls, "bare 429 must not rate limit the account")
}

// When upstream names the quota as exhausted (no model), the whole Gemini family is cooled
// intentionally: that is real account-level evidence, unlike a bare 429.
func TestHandleUpstreamError_QuotaExhaustedCoolsFamily(t *testing.T) {
	repo := &stubAntigravityAccountRepo{}
	svc := &AntigravityGatewayService{accountRepo: repo}
	account := &Account{ID: 7, Name: "ag", Platform: PlatformAntigravity}

	body := []byte(`{"error":{"code":429,"message":"QUOTA_EXHAUSTED","status":"RESOURCE_EXHAUSTED"}}`)
	svc.handleUpstreamError(context.Background(), "[test]", account, http.StatusTooManyRequests, http.Header{}, body, "gemini-3.8-flash-high", 0, "", false)

	keys := make([]string, 0, len(repo.modelRateLimitCalls))
	for _, call := range repo.modelRateLimitCalls {
		keys = append(keys, call.modelKey)
	}
	require.ElementsMatch(t, []string{"gemini-3.8-flash-high", "antigravity:gemini"}, keys)
	require.Empty(t, repo.rateCalls)
}

func TestAntigravityModelRateLimitKeysScoped(t *testing.T) {
	require.Equal(t, []string{"gemini-3.8-flash-high"}, antigravityModelRateLimitKeysScoped("gemini-3.8-flash-high", false))
	require.Equal(t, []string{"gemini-3.8-flash-high", "antigravity:gemini"}, antigravityModelRateLimitKeysScoped("gemini-3.8-flash-high", true))
	require.Equal(t, []string{"gemini-3.1-flash-image"}, antigravityModelRateLimitKeysScoped("gemini-3.1-flash-image", true),
		"the image endpoint is separate from the text family")
	require.Equal(t, []string{"antigravity:gemini"}, antigravityModelRateLimitKeysScoped("antigravity:gemini", true))
	require.Nil(t, antigravityModelRateLimitKeysScoped("  ", true))
}

func TestAntigravityRequestContentErrorMessage(t *testing.T) {
	msg, ok := antigravityRequestContentErrorMessage(http.StatusBadRequest, " Request contains an invalid argument. ")
	require.True(t, ok)
	require.Equal(t, "Request contains an invalid argument.", msg)

	msg, ok = antigravityRequestContentErrorMessage(http.StatusRequestEntityTooLarge, "")
	require.True(t, ok)
	require.NotEmpty(t, msg)

	_, ok = antigravityRequestContentErrorMessage(http.StatusTooManyRequests, "slow down")
	require.False(t, ok, "429 keeps the rate limit mapping")
	_, ok = antigravityRequestContentErrorMessage(http.StatusInternalServerError, "boom")
	require.False(t, ok)
}
