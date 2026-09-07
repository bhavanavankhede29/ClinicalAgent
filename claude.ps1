param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Prompt,

    [string]$Model = $(if ($env:CLAUDE_MODEL) { $env:CLAUDE_MODEL } else { 'claude-sonnet-4-20250514' }),

    [int]$MaxTokens = 1024
)

if ([string]::IsNullOrWhiteSpace($env:ANTHROPIC_API_KEY)) {
    throw 'ANTHROPIC_API_KEY is not set. Set it in the current terminal session and try again.'
}

$headers = @{
    'x-api-key' = $env:ANTHROPIC_API_KEY
    'anthropic-version' = '2023-06-01'
    'content-type' = 'application/json'
}

$body = @{
    model = $Model
    max_tokens = $MaxTokens
    messages = @(
        @{
            role = 'user'
            content = $Prompt
        }
    )
} | ConvertTo-Json -Depth 5

try {
    $response = Invoke-RestMethod `
        -Method Post `
        -Uri 'https://api.anthropic.com/v1/messages' `
        -Headers $headers `
        -Body $body

    $response.content | ForEach-Object { $_.text }
} catch {
    $details = $_.ErrorDetails.Message
    if ($details) {
        throw "Claude API request failed: $details"
    }
    throw
}
