# Switch between Kaggle accounts seamlessly
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("main", "123", "second")]
    [string]$Account
)

$kaggleDir = "$HOME\.kaggle"
$metaFile = "kernel\kernel-metadata.json"

switch ($Account) {
    "main" {
        if (Test-Path "$kaggleDir\access_token") {
            Remove-Item "$kaggleDir\access_token" -Force
        }
        $username = "parthchallawar"
    }
    "123" {
        if (-not (Test-Path "$kaggleDir\access_token_123")) {
            Write-Error "Token for 123 not found at $kaggleDir\access_token_123"
            exit 1
        }
        Copy-Item "$kaggleDir\access_token_123" "$kaggleDir\access_token" -Force
        $username = "parthchallawar123"
    }
    "second" {
        if (-not (Test-Path "$kaggleDir\access_token_second")) {
            Write-Error "Token for second friend not found at $kaggleDir\access_token_second. Please add it first!"
            exit 1
        }
        Copy-Item "$kaggleDir\access_token_second" "$kaggleDir\access_token" -Force
        $username = "parthrchallawar"
    }
}

if (Test-Path $metaFile) {
    $content = Get-Content $metaFile -Raw
    $updated = $content -replace '"id":\s*"[^/]+/adl-encrypted-traffic-train"', "`"id`": `"$username/adl-encrypted-traffic-train`""
    Set-Content $metaFile $updated
}

Write-Host "Successfully switched to Kaggle account: $username" -ForegroundColor Green
Write-Host "Verifying authentication..."
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('Active Kaggle User:', api.config_values.get('username'))"

