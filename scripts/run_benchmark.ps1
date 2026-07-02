# PowerShell entry point for run_benchmark.sh on Windows.
#
# Usage:
#   .\scripts\run_benchmark.ps1 --dry-run --tasks objective_fact_judgment --methods RawDialogue --limit 1
#   .\scripts\run_benchmark.ps1 --tasks objective_fact_judgment --methods RawDialogue,MemZero --limit 5
#
# PowerShell treats commas as argument separators and may also mangle commas when
# passing arguments to native executables. This wrapper normalizes list options and
# invokes Git Bash via a single -lc command with shell-safe quoting.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$ShScript = Join-Path $ScriptDir "run_benchmark.sh"

function Convert-ToBashArgs {
    param([string[]]$InputArgs)

    $listOptionNames = @{
        "--tasks" = $true
        "--methods" = $true
        "--model" = $true
        "--models" = $true
    }

    $result = @()
    $index = 0
    while ($index -lt $InputArgs.Length) {
        $arg = $InputArgs[$index]
        if ($listOptionNames.ContainsKey($arg)) {
            $valueParts = @()
            $index++
            while ($index -lt $InputArgs.Length -and -not $InputArgs[$index].StartsWith("--")) {
                $valueParts += $InputArgs[$index]
                $index++
            }
            if ($valueParts.Count -eq 0) {
                throw "Missing value for $arg"
            }
            $result += $arg
            $result += ($valueParts -join ",")
            continue
        }
        $result += $arg
        $index++
    }
    return , $result
}

function Quote-BashSingle {
    param([string]$Text)
    return "'" + ($Text -replace "'", "'\\''") + "'"
}

function Find-GitBash {
    $candidates = @(
        (Join-Path ${env:ProgramFiles} "Git\bin\bash.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

$bash = Find-GitBash
if (-not $bash) {
    Write-Error @"
Git Bash was not found. Install Git for Windows, then rerun:

  .\scripts\run_benchmark.ps1 ...

Or invoke bash explicitly:

  & '<path-to-git>\bin\bash.exe' scripts/run_benchmark.sh ...
"@
    exit 1
}

$bashArgs = Convert-ToBashArgs -InputArgs $args
$repoPath = ($RepoRoot -replace '\\', '/')
$scriptPath = ($ShScript -replace '\\', '/')

$commandParts = @(
    "cd $(Quote-BashSingle $repoPath)",
    "&&",
    "bash",
    (Quote-BashSingle $scriptPath)
)
foreach ($arg in $bashArgs) {
    $commandParts += (Quote-BashSingle $arg)
}

& $bash -lc ($commandParts -join ' ')
exit $LASTEXITCODE
