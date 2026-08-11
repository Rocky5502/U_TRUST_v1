$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Ext = Join-Path $Root "external"
New-Item -ItemType Directory -Force -Path $Ext | Out-Null

$AgentDojoSha = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
$InjecAgentSha = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"

function Clone-And-Pin($Url, $Dir, $Sha) {
    if (-not (Test-Path (Join-Path $Dir ".git"))) {
        git clone $Url $Dir
    }
    git -C $Dir fetch --all --tags --prune
    git -C $Dir checkout --detach $Sha
    Write-Host "Pinned $Dir -> $(git -C $Dir rev-parse HEAD)"
}

Clone-And-Pin "https://github.com/ethz-spylab/agentdojo.git" (Join-Path $Ext "agentdojo") $AgentDojoSha
Clone-And-Pin "https://github.com/uiuc-kang-lab/InjecAgent.git" (Join-Path $Ext "InjecAgent") $InjecAgentSha
