param(
    [string]$ProjectDir = "C:\Users\balt5\fed_grid_diplo\fed-grid-diplo",
    [string]$YmlDir     = ".\experiments_yml\pc",
    [int]   $StartFrom  = 1
)

function Write-Header { param($msg)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $msg"                                    -ForegroundColor Cyan
    Write-Host "========================================"  -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green  }
function Write-Warn { param($msg) Write-Host "[!!] $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "[XX] $msg" -ForegroundColor Red    }
function Write-Info { param($msg) Write-Host "     $msg" -ForegroundColor Gray   }

$AllExperiments = @(
    @{ Id="01"; Name="fedavg_no_attack";      Algo="fedavg";      Attack="none";            Pct=0  }
    @{ Id="02"; Name="fedavg_la_10pct";       Algo="fedavg";      Attack="label_flipping";  Pct=10 }
    @{ Id="03"; Name="fedavg_si_10pct";       Algo="fedavg";      Attack="sign_flipping";   Pct=10 }
    @{ Id="04"; Name="fedavg_ga_10pct";       Algo="fedavg";      Attack="gaussian_noise";  Pct=10 }
    @{ Id="05"; Name="fedavg_la_20pct";       Algo="fedavg";      Attack="label_flipping";  Pct=20 }
    @{ Id="06"; Name="fedavg_si_20pct";       Algo="fedavg";      Attack="sign_flipping";   Pct=20 }
    @{ Id="07"; Name="fedavg_ga_20pct";       Algo="fedavg";      Attack="gaussian_noise";  Pct=20 }
    @{ Id="08"; Name="fedavg_la_30pct";       Algo="fedavg";      Attack="label_flipping";  Pct=30 }
    @{ Id="09"; Name="fedavg_si_30pct";       Algo="fedavg";      Attack="sign_flipping";   Pct=30 }
    @{ Id="10"; Name="fedavg_ga_30pct";       Algo="fedavg";      Attack="gaussian_noise";  Pct=30 }
    @{ Id="11"; Name="krum_no_attack";        Algo="krum";        Attack="none";            Pct=0  }
    @{ Id="12"; Name="krum_la_10pct";         Algo="krum";        Attack="label_flipping";  Pct=10 }
    @{ Id="13"; Name="krum_si_10pct";         Algo="krum";        Attack="sign_flipping";   Pct=10 }
    @{ Id="14"; Name="krum_ga_10pct";         Algo="krum";        Attack="gaussian_noise";  Pct=10 }
    @{ Id="15"; Name="krum_la_20pct";         Algo="krum";        Attack="label_flipping";  Pct=20 }
    @{ Id="16"; Name="krum_si_20pct";         Algo="krum";        Attack="sign_flipping";   Pct=20 }
    @{ Id="17"; Name="krum_ga_20pct";         Algo="krum";        Attack="gaussian_noise";  Pct=20 }
    @{ Id="18"; Name="krum_la_30pct";         Algo="krum";        Attack="label_flipping";  Pct=30 }
    @{ Id="19"; Name="krum_si_30pct";         Algo="krum";        Attack="sign_flipping";   Pct=30 }
    @{ Id="20"; Name="krum_ga_30pct";         Algo="krum";        Attack="gaussian_noise";  Pct=30 }
    @{ Id="21"; Name="multi_krum_no_attack";  Algo="multi_krum";  Attack="none";            Pct=0  }
    @{ Id="22"; Name="multi_krum_la_10pct";   Algo="multi_krum";  Attack="label_flipping";  Pct=10 }
    @{ Id="23"; Name="multi_krum_si_10pct";   Algo="multi_krum";  Attack="sign_flipping";   Pct=10 }
    @{ Id="24"; Name="multi_krum_ga_10pct";   Algo="multi_krum";  Attack="gaussian_noise";  Pct=10 }
    @{ Id="25"; Name="multi_krum_la_20pct";   Algo="multi_krum";  Attack="label_flipping";  Pct=20 }
    @{ Id="26"; Name="multi_krum_si_20pct";   Algo="multi_krum";  Attack="sign_flipping";   Pct=20 }
    @{ Id="27"; Name="multi_krum_ga_20pct";   Algo="multi_krum";  Attack="gaussian_noise";  Pct=20 }
    @{ Id="28"; Name="multi_krum_la_30pct";   Algo="multi_krum";  Attack="label_flipping";  Pct=30 }
    @{ Id="29"; Name="multi_krum_si_30pct";   Algo="multi_krum";  Attack="sign_flipping";   Pct=30 }
    @{ Id="30"; Name="multi_krum_ga_30pct";   Algo="multi_krum";  Attack="gaussian_noise";  Pct=30 }
    @{ Id="31"; Name="tmean_no_attack";       Algo="tmean";       Attack="none";            Pct=0  }
    @{ Id="32"; Name="tmean_la_10pct";        Algo="tmean";       Attack="label_flipping";  Pct=10 }
    @{ Id="33"; Name="tmean_si_10pct";        Algo="tmean";       Attack="sign_flipping";   Pct=10 }
    @{ Id="34"; Name="tmean_ga_10pct";        Algo="tmean";       Attack="gaussian_noise";  Pct=10 }
    @{ Id="35"; Name="tmean_la_20pct";        Algo="tmean";       Attack="label_flipping";  Pct=20 }
    @{ Id="36"; Name="tmean_si_20pct";        Algo="tmean";       Attack="sign_flipping";   Pct=20 }
    @{ Id="37"; Name="tmean_ga_20pct";        Algo="tmean";       Attack="gaussian_noise";  Pct=20 }
    @{ Id="38"; Name="tmean_la_30pct";        Algo="tmean";       Attack="label_flipping";  Pct=30 }
    @{ Id="39"; Name="tmean_si_30pct";        Algo="tmean";       Attack="sign_flipping";   Pct=30 }
    @{ Id="40"; Name="tmean_ga_30pct";        Algo="tmean";       Attack="gaussian_noise";  Pct=30 }
    @{ Id="41"; Name="fltrust_no_attack";     Algo="fltrust";     Attack="none";            Pct=0  }
    @{ Id="42"; Name="fltrust_la_10pct";      Algo="fltrust";     Attack="label_flipping";  Pct=10 }
    @{ Id="43"; Name="fltrust_si_10pct";      Algo="fltrust";     Attack="sign_flipping";   Pct=10 }
    @{ Id="44"; Name="fltrust_ga_10pct";      Algo="fltrust";     Attack="gaussian_noise";  Pct=10 }
    @{ Id="45"; Name="fltrust_la_20pct";      Algo="fltrust";     Attack="label_flipping";  Pct=20 }
    @{ Id="46"; Name="fltrust_si_20pct";      Algo="fltrust";     Attack="sign_flipping";   Pct=20 }
    @{ Id="47"; Name="fltrust_ga_20pct";      Algo="fltrust";     Attack="gaussian_noise";  Pct=20 }
    @{ Id="48"; Name="fltrust_la_30pct";      Algo="fltrust";     Attack="label_flipping";  Pct=30 }
    @{ Id="49"; Name="fltrust_si_30pct";      Algo="fltrust";     Attack="sign_flipping";   Pct=30 }
    @{ Id="50"; Name="fltrust_ga_30pct";      Algo="fltrust";     Attack="gaussian_noise";  Pct=30 }
)

$Experiments = $AllExperiments | Where-Object { [int]$_.Id -ge $StartFrom }
$TotalExps   = ($Experiments | Measure-Object).Count

$ResultsDir = "$ProjectDir\results\replication_run"
if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir | Out-Null }

$LogFile = "$ResultsDir\run_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
function Log { param($msg) "$(Get-Date -Format 'HH:mm:ss')  $msg" | Out-File $LogFile -Append }

$StartAll   = Get-Date
$CurrentExp = 0

Write-Host "`nStarting from exp $StartFrom | $TotalExps experiments total" -ForegroundColor Yellow
Write-Host "Results -> $ResultsDir" -ForegroundColor Yellow
Log "START -- $TotalExps experiments from exp $StartFrom"

foreach ($Exp in $Experiments) {
    $CurrentExp++
    $YmlFile      = "$YmlDir\exp$($Exp.Id)_$($Exp.Name)_pc.yml"
    $RunLabel     = "exp$($Exp.Id)_$($Exp.Name)"
    $RunResultDir = "$ResultsDir\$RunLabel"
    $Prefix       = "exp$($Exp.Id)"

    Write-Header "[$CurrentExp/$TotalExps] exp$($Exp.Id) | $($Exp.Algo.ToUpper()) | attack=$($Exp.Attack) | attackers=$($Exp.Pct)%"

    if (-not (Test-Path $YmlFile)) {
        Write-Fail "YML not found: $YmlFile -- SKIPPING"
        Log "SKIP exp$($Exp.Id) -- YML missing"
        continue
    }

    if (-not (Test-Path $RunResultDir)) {
        New-Item -ItemType Directory -Path $RunResultDir | Out-Null
    }

    Log "START $RunLabel"
    $StartRun = Get-Date

    Push-Location $ProjectDir

    # -- Patch all relative paths to absolute in a temp YML ----------------
    $ProjectDirUnix = $ProjectDir.Replace('\', '/')
    $TempYml = "$env:TEMP\fedgrid_exp$($Exp.Id)_temp.yml"
    (Get-Content $YmlFile -Raw -Encoding UTF8) `
        -replace '\./fog/mosquitto\.conf',   "$ProjectDirUnix/fog/mosquitto.conf" `
        -replace '\./cloud/mosquitto\.conf', "$ProjectDirUnix/cloud/mosquitto.conf" `
        -replace '\./data/house_',           "$ProjectDirUnix/lcl_data/house_" `
        -replace '\./data:',                 "$ProjectDirUnix/lcl_data:" `
        -replace '\./fog/models',            "$ProjectDirUnix/fog/models" `
        -replace '\./cloud/models',          "$ProjectDirUnix/cloud/models" `
        -replace '\./cloud/status',          "$ProjectDirUnix/cloud/status" `
        -replace '\./edge/models/',          "$ProjectDirUnix/edge/models/" |
        Set-Content $TempYml -Encoding UTF8
    Write-Info "Patched YML -> $TempYml"

    # -- Teardown ----------------------------------------------------------
    Write-Info "Teardown previous containers..."
    try { docker compose -f $TempYml down --remove-orphans --timeout 30 2>$null } catch {}
    docker ps -a --format "{{.Names}}" 2>$null |
        Where-Object { $_ -match "^$Prefix" } |
        ForEach-Object { docker rm -f $_ 2>$null }

    # -- Full state cleanup ------------------------------------------------
    Write-Info "Cleaning stale state..."

    # Edge + fog models
    Get-ChildItem "$ProjectDir\edge\models"  -Recurse -Filter "*.keras" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem "$ProjectDir\fog\models"   -Recurse -Filter "*.keras" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    # Cloud models + status (round_id.json causes TEST_ROUND bug, metrics.json causes parse errors)
    Get-ChildItem "$ProjectDir\cloud\models" -Recurse -Filter "*.keras" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Remove-Item "$ProjectDir\cloud\status\round_id.json" -Force -ErrorAction SilentlyContinue
    Remove-Item "$ProjectDir\cloud\status\metrics.json"  -Force -ErrorAction SilentlyContinue

    # FLTrust flags
    Get-ChildItem "$ProjectDir\fog" -Recurse -Filter "fltrust_initialized.flag" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem "$ProjectDir\fog" -Recurse -Filter "fltrust_prev_weights.npz"  -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    # -- Run ---------------------------------------------------------------
    Write-Info "Starting experiment..."
    Write-Host "------------------------------------------" -ForegroundColor DarkGray

    docker compose -f $TempYml up --abort-on-container-exit --exit-code-from cloud_app

    Write-Host "------------------------------------------" -ForegroundColor DarkGray

    $Duration = [int]((Get-Date) - $StartRun).TotalMinutes
    Write-OK "Finished in ~$Duration min"
    Log "DONE $RunLabel (${Duration}min)"

    # -- Save logs ---------------------------------------------------------
    Write-Info "Saving logs..."

    docker logs "$($Prefix)_cloud_app" 2>$null |
        Out-File "$RunResultDir\cloud_app.log" -Encoding UTF8 -ErrorAction SilentlyContinue

    docker logs "$($Prefix)_fog_app" 2>$null |
        Out-File "$RunResultDir\fog_app.log" -Encoding UTF8 -ErrorAction SilentlyContinue

    Select-String -Path "$RunResultDir\fog_app.log" -Pattern "AGG METRICS" -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Line } |
        Out-File "$RunResultDir\agg_metrics.log" -Encoding UTF8 -ErrorAction SilentlyContinue

    for ($n = 1; $n -le 10; $n++) {
        docker logs "$($Prefix)_edge_node_$n" 2>$null |
            Out-File "$RunResultDir\edge_${n}.log" -Encoding UTF8 -ErrorAction SilentlyContinue
    }

    if ($Exp.Pct -ge 10) { docker logs "$($Prefix)_edge_node_10" 2>$null | Out-File "$RunResultDir\attacker_10.log" -Encoding UTF8 -ErrorAction SilentlyContinue }
    if ($Exp.Pct -ge 20) { docker logs "$($Prefix)_edge_node_9"  2>$null | Out-File "$RunResultDir\attacker_9.log"  -Encoding UTF8 -ErrorAction SilentlyContinue }
    if ($Exp.Pct -ge 30) { docker logs "$($Prefix)_edge_node_8"  2>$null | Out-File "$RunResultDir\attacker_8.log"  -Encoding UTF8 -ErrorAction SilentlyContinue }

    Write-OK "Logs saved -> $RunResultDir"

    # -- Teardown ----------------------------------------------------------
    Write-Info "Teardown..."
    try { docker compose -f $TempYml down --remove-orphans --timeout 30 2>$null } catch {}
    Remove-Item $TempYml -Force -ErrorAction SilentlyContinue

    Pop-Location

    Write-Info "Cooling down 20s..."
    Start-Sleep -Seconds 20
}

$TotalMin = [int]((Get-Date) - $StartAll).TotalMinutes
Write-Header "ALL DONE"
Write-OK "Completed: $CurrentExp/$TotalExps experiments | ~$TotalMin min total"
Write-OK "Results -> $ResultsDir"
Write-OK "Log     -> $LogFile"
Log "ALL DONE -- ${CurrentExp}/${TotalExps} experiments | ${TotalMin}min"