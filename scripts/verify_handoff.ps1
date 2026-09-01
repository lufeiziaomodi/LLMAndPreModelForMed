param(
    [string]$ExpectedCommit = "",
    [string]$ExpectedBranch = "refactor/data-flow"
)

$ErrorActionPreference = "Stop"

function Invoke-Git([string[]]$Arguments) {
    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed: $output"
    }
    return ($output | Out-String).Trim()
}

$repoRoot = Invoke-Git @("rev-parse", "--show-toplevel")
Set-Location -LiteralPath $repoRoot

$branch = Invoke-Git @("branch", "--show-current")
$head = Invoke-Git @("rev-parse", "--short", "HEAD")
$trackedChanges = Invoke-Git @("status", "--short", "--untracked-files=no")
$userApiKey = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")
$processApiKey = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "Process")

$kgCsv = @(Get-ChildItem -Path "data\kg" -Filter "*.csv" -File -ErrorAction SilentlyContinue)
$kgCache = @(
    Get-ChildItem -Path "data\kg" -Filter "*.pkl" -File -ErrorAction SilentlyContinue
    Get-ChildItem -Path "data\kg" -Filter "*.pt" -File -ErrorAction SilentlyContinue
)

$checks = [ordered]@{
    RepoRoot = $repoRoot
    Branch = $branch
    Head = $head
    TrackedWorktreeClean = [string]::IsNullOrWhiteSpace($trackedChanges)
    ExpectedCommitIsAncestor = $null
    PrimeKgCsvCount = $kgCsv.Count
    PrimeKgCacheCount = $kgCache.Count
    ModelsDirectoryExists = Test-Path -LiteralPath "models"
    ResultsDirectoryExists = Test-Path -LiteralPath "results"
    DashScopeApiKeyAvailable = -not [string]::IsNullOrWhiteSpace($userApiKey) -or -not [string]::IsNullOrWhiteSpace($processApiKey)
}

if ($ExpectedCommit) {
    & git merge-base --is-ancestor $ExpectedCommit HEAD 2>$null
    $checks.ExpectedCommitIsAncestor = $LASTEXITCODE -eq 0
}

$checks.GetEnumerator() | ForEach-Object {
    "{0,-28}: {1}" -f $_.Key, $_.Value
}

$errors = @()
if ($branch -ne $ExpectedBranch) {
    $errors += "branch is '$branch', expected '$ExpectedBranch'"
}
if (-not [string]::IsNullOrWhiteSpace($trackedChanges)) {
    $errors += "tracked worktree changes exist"
}
if ($ExpectedCommit -and -not $checks.ExpectedCommitIsAncestor) {
    $errors += "expected commit '$ExpectedCommit' is not contained in HEAD"
}

if ($errors.Count -gt 0) {
    Write-Error ("Handoff verification failed: " + ($errors -join "; "))
    exit 1
}

Write-Output "HANDOFF_STATUS              : PASS"
