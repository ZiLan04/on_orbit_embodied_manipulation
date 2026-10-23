# deploy.ps1 鈥斺€?鎶婃湰鍦版潈濞佷唬鐮佸崟鍚戝悓姝ュ埌鏈嶅姟鍣?#
# 鐢ㄦ硶:
#   .\deploy\deploy.ps1                 # 鍚屾 genesis_tools锛堥粯璁わ級
#   .\deploy\deploy.ps1 -DryRun         # 鍙湅浼氫紶浠€涔?#   .\deploy\deploy.ps1 -Force          # 鍏佽瑕嗙洊杩滅鏇存柊鐨勬枃浠讹紙鎱庣敤锛?#
# 鈿狅笍 2026-09-17 浜嬫晠璁板綍涓庢暀璁?# ---------------------------------------------------------------
# 鏈剼鏈師鍏堝悓姝?`starvla_bridge`锛岃€岄偅涓洰褰曞湪鏈嶅姟鍣ㄤ笂琚彟涓€浣?agent 鎸佺画
# 寮€鍙戯紙鍔犱簡 --alignment-target / --manual-action / state adapter 绛夛級銆?# 涓€娆?scp 鎶婁粬浠殑鏀瑰姩鏁翠釜瑕嗙洊鎺変簡锛屽彧鍓?__pycache__ 閲岀殑 .pyc 鍙仮澶?# 锛堝凡澶囦唤鍒?/data/Yanhan/genesis/recovered/锛夈€?#
# 鍥犳鐜板湪鏈変袱鏉＄‖瑙勫垯锛?#   1. `starvla_bridge` 涓嶅啀绾冲叆鍚屾銆傚畠鏄粠 Genesis 鏈烘媺鍥炴潵鐨?*鍙傝€冨揩鐓?*锛?#      涓嶆槸閮ㄧ讲婧愶紱鏈嶅姟鍣ㄤ笂閭ｄ唤鐢卞湪閭ｈ竟宸ヤ綔鐨勪汉缁存姢銆?#   2. 瑕嗙洊鍓嶉€愭枃浠舵瘮杈?mtime锛氳繙绔洿鏂扮殑鏂囦欢涓€寰?*鎷掔粷瑕嗙洊**骞跺垪鍑烘潵銆?#
# 涓€鍙ヨ瘽鏁欒锛?*鍒妸鍒汉姝ｅ湪缂栬緫鐨勭洰褰曟斁杩涜嚜鍔ㄨ鐩栫殑鍚屾鑼冨洿銆?*

param(
    [string]$TargetHost = "10.110.10.12",
    [string]$TargetUser = "student",
    [string]$RemoteRoot = "/data/Yanhan/genesis",
    [string[]]$Dirs = @("code"),
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$AskPass  = Join-Path $RepoRoot ".staging\ssh-askpass.cmd"   # local-only helper, not in this repo

if (-not (Test-Path $AskPass)) { throw "missing askpass helper: $AskPass" }

$env:SSH_ASKPASS         = $AskPass
$env:SSH_ASKPASS_REQUIRE = "force"
if (-not $env:DSH_SSH_PW) { throw "set DSH_SSH_PW in the environment; no default is committed" }
$sshTarget = "${TargetUser}@${TargetHost}"
$RepoRoot  = (Resolve-Path $RepoRoot).Path

if ($Dirs -contains "starvla_bridge") {
    throw "starvla_bridge is excluded on purpose (another agent maintains it on the server). See this script's header."
}

$CodeDirs = $Dirs | Where-Object { Test-Path (Join-Path $RepoRoot $_) }
if ($CodeDirs.Count -eq 0) { throw "no code directories found to deploy" }

$Commit = (git -C $RepoRoot rev-parse --short HEAD)
$Dirty  = if (git -C $RepoRoot status --porcelain) { "true" } else { "false" }

Write-Host "==> repo    : $RepoRoot"
Write-Host "==> commit  : $Commit (dirty=$Dirty)"
Write-Host "==> target  : ${sshTarget}:${RemoteRoot}/code"
Write-Host "==> dirs    : $($CodeDirs -join ', ')"

# --- guard: never clobber a remote file that is newer than the local one ---
#
# 鈿狅笍 2026-09-18 鈥?do NOT put an `ssh` call inside a per-file loop.
# That is exactly what this guard used to do (one connection per file), and it
# tripped the server's connection rate limiter: `kex_exchange_identification:
# read: Connection reset`, then `Connection refused` on every port for ~12
# minutes -- twice in a row. Self-inflicted downtime that looks just like the
# machine rebooting. So: collect every remote path first, then ask about all of
# them in ONE connection.
Write-Host "==> checking for newer remote files ..."
$clobber = @()
$exts = @(".py", ".sh", ".md", ".json", ".yaml", ".yml")

$remotePaths = New-Object System.Collections.Generic.List[string]
$localByPath = @{}
foreach ($d in $CodeDirs) {
    $localDir = Join-Path $RepoRoot $d
    $localFiles = Get-ChildItem -Path $localDir -Recurse -File | Where-Object { $exts -contains $_.Extension }
    foreach ($lf in $localFiles) {
        $rel = $lf.FullName.Substring($localDir.Length + 1) -replace '\\', '/'
        $remotePath = "$RemoteRoot/$d/$rel"
        $remotePaths.Add($remotePath)
        $localByPath[$remotePath] = $lf
    }
}

if ($remotePaths.Count -gt 0) {
    $remoteCmd = 'while IFS= read -r p; do printf "%s\t" "$p"; stat -c %Y "$p" 2>/dev/null || echo 0; done'
    $statOut = ($remotePaths.ToArray() -join "`n") | & ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR $sshTarget $remoteCmd 2>$null
    foreach ($line in $statOut) {
        if ("$line" -notmatch "`t") { continue }
        $parts = "$line".Split("`t")
        if ($parts.Count -lt 2) { continue }
        $remotePath = $parts[0].Trim()
        $remoteEpoch = $parts[1].Trim()
        if ($remoteEpoch -notmatch '^\d+$') { continue }
        if ([int64]$remoteEpoch -le 0) { continue }
        if (-not $localByPath.ContainsKey($remotePath)) { continue }
        $lf = $localByPath[$remotePath]
        $remoteUtc = [DateTimeOffset]::FromUnixTimeSeconds([int64]$remoteEpoch).UtcDateTime
        if ($remoteUtc -gt $lf.LastWriteTimeUtc.AddSeconds(2)) {
            $rel = $remotePath.Substring("$RemoteRoot/".Length)
            $clobber += ("{0}  (remote {1:u} > local {2:u})" -f $rel, $remoteUtc, $lf.LastWriteTimeUtc)
        }
    }
    Write-Host "==> queried $($remotePaths.Count) remote file(s) over 1 connection"
}
if ($clobber.Count -gt 0) {
    Write-Host "==> REMOTE FILES ARE NEWER THAN LOCAL:" -ForegroundColor Yellow
    $clobber | ForEach-Object { Write-Host "      $_" -ForegroundColor Yellow }
    if (-not $Force) {
        throw "aborting: $($clobber.Count) remote file(s) are newer. Nothing was overwritten. Re-run with -Force only if certain."
    }
    Write-Host "==> -Force given; overwriting anyway" -ForegroundColor Yellow
} else {
    Write-Host "==> no newer remote files; safe to overwrite"
}

if ($DryRun) { Write-Host "==> DRY RUN, nothing sent"; exit 0 }

& ssh -o StrictHostKeyChecking=no $sshTarget "mkdir -p ${RemoteRoot}/code ${RemoteRoot}/runs"
if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed" }

foreach ($d in $CodeDirs) {
    Write-Host "==> syncing $d ..."
    # -p preserves mtimes. Without it scp stamps the remote files with the transfer
    # time, which would make the newer-file guard above trip on every deploy.
    #
    # NOTE: this copies the whole `code/` tree. The server currently still holds the
    # PRE-restructure layout (code/starlink_cube_rl/, code/genesis_tools/, ...), so a
    # run today adds code/tasks/ and code/bridges/ *beside* it rather than replacing
    # it. The cut-over is a manual, reversible step documented in
    # documents/README.en.md -- do not treat a deploy as the cut-over.
    & scp -p -o StrictHostKeyChecking=no -r (Join-Path $RepoRoot $d) "${sshTarget}:${RemoteRoot}/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $d" }
}

$payload = @{
    commit     = $Commit
    dirty      = $Dirty
    deployedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    dirs       = $CodeDirs
    source     = "local git repo"
    note       = "starvla_bridge is NOT deployed; it is maintained on the server"
} | ConvertTo-Json

$tmp = [System.IO.Path]::GetTempFileName()
[System.IO.File]::WriteAllText($tmp, $payload)
& scp -o StrictHostKeyChecking=no $tmp "${sshTarget}:${RemoteRoot}/DEPLOYED.json" | Out-Null
Remove-Item $tmp -Force

Write-Host "==> done. DEPLOYED.json written."
