# 🎮 ML TRAINING COMMAND SHORTCUTS
# ================================
# 
# This PowerShell script provides convenient aliases and functions for 
# training different games and generating videos across multiple systems.
#
# Usage:
#   . .\scripts\ml_commands.ps1  # Source this file to load functions
#   
# Then use the functions:
#   Train-Game breakout 5h
#   Train-Game tetris 2h -Test
#   Make-Video space_invaders 30min
#   Train-System atari -Epic 1 -Hours 8

# Activate virtual environment if not already active
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
        & .venv\Scripts\Activate.ps1
    }
}

# Helper function to parse duration
function Parse-Duration {
    param([string]$Duration)
    
    $Duration = $Duration.ToLower().Trim()
    
    if ($Duration -match '^(\d+(?:\.\d+)?)min?$') {
        return [float]$Matches[1] / 60.0
    }
    elseif ($Duration -match '^(\d+(?:\.\d+)?)h(?:ours?)?$') {
        return [float]$Matches[1]
    }
    else {
        # Assume hours if no unit
        return [float]$Duration
    }
}

# 🎮 TRAINING FUNCTIONS
# ====================

function Train-Game {
    <#
    .SYNOPSIS
    Train a specific game for a custom duration
    
    .EXAMPLE
    Train-Game breakout 5h
    Train-Game tetris 2h -Epic 2 -Test
    Train-Game space_invaders 30min
    #>
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet("breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger", "tetris")]
        [string]$Game,
        
        [Parameter(Mandatory=$true)]
        [string]$Duration,
        
        [ValidateRange(1,3)]
        [int]$Epic = 1,
        
        [switch]$Test
    )
    
    $Hours = Parse-Duration $Duration
    
    Write-Host "🎮 Training $Game for $Duration (Epic $Epic)" -ForegroundColor Green
    
    $Args = @(
        "scripts\custom_training_commands.py", "train",
        "--game", $Game,
        "--hours", $Hours,
        "--epic", $Epic
    )
    
    if ($Test) { $Args += "--test" }
    
    python @Args
}

function Train-System {
    <#
    .SYNOPSIS
    Train all games in a gaming system
    
    .EXAMPLE
    Train-System atari -Hours 5 -Epic 1
    Train-System gameboy -Hours 2 -Test
    #>
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet("atari", "gameboy")]
        [string]$System,
        
        [float]$Hours = 10,
        
        [ValidateRange(1,3)]
        [int]$Epic = 1,
        
        [switch]$Test
    )
    
    Write-Host "🎮 Training all $System games (Epic $Epic, ${Hours}h each)" -ForegroundColor Green
    
    $Args = @(
        "scripts\custom_training_commands.py", "system-train",
        "--system", $System,
        "--hours", $Hours,
        "--epic", $Epic
    )
    
    if ($Test) { $Args += "--test" }
    
    python @Args
}

function Train-Batch {
    <#
    .SYNOPSIS
    Train multiple games in batch
    
    .EXAMPLE
    Train-Batch "breakout,tetris,pong" 2h -Epic 1
    Train-Batch "space_invaders,asteroids" 1h -Test
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Games,
        
        [Parameter(Mandatory=$true)]
        [string]$Duration,
        
        [ValidateRange(1,3)]
        [int]$Epic = 1,
        
        [switch]$Test
    )
    
    $Hours = Parse-Duration $Duration
    
    Write-Host "🎮 Batch training: $Games ($Duration each, Epic $Epic)" -ForegroundColor Green
    
    $Args = @(
        "scripts\custom_training_commands.py", "batch",
        "--games", $Games,
        "--hours", $Hours,
        "--epic", $Epic
    )
    
    if ($Test) { $Args += "--test" }
    
    python @Args
}

# 📹 VIDEO FUNCTIONS
# ==================

function Make-Video {
    <#
    .SYNOPSIS
    Generate a video of specific length for a game
    
    .EXAMPLE
    Make-Video breakout 30min
    Make-Video tetris 2h -Epic 2 -Quality high
    #>
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet("breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger", "tetris")]
        [string]$Game,
        
        [Parameter(Mandatory=$true)]
        [string]$Length,
        
        [ValidateRange(1,3)]
        [int]$Epic = 1,
        
        [ValidateSet("low", "medium", "high")]
        [string]$Quality = "medium"
    )
    
    Write-Host "📹 Generating $Length video for $Game (Epic $Epic)" -ForegroundColor Cyan
    
    python scripts\custom_training_commands.py video --game $Game --length $Length --epic $Epic --quality $Quality
}

# 🔧 UTILITY FUNCTIONS
# ====================

function Check-Epic-Status {
    <#
    .SYNOPSIS
    Check the status of epic training sessions
    #>
    Write-Host "📊 Checking epic training status..." -ForegroundColor Yellow
    python epic_training\scripts\check_epic_status.py
}

function Quick-Train {
    <#
    .SYNOPSIS
    Quick test training (1 minute) for any game
    
    .EXAMPLE
    Quick-Train tetris
    Quick-Train breakout
    #>
    param(
        [Parameter(Mandatory=$true)]
        [ValidateSet("breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger", "tetris")]
        [string]$Game
    )
    
    Write-Host "⚡ Quick test training: $Game (1 minute)" -ForegroundColor Yellow
    python epic_training\scripts\train_epic_continuous.py --game $Game --epic 1 --test
}

function Eval-Model {
    <#
    .SYNOPSIS
    Evaluate a trained model
    
    .EXAMPLE
    Eval-Model "models\checkpoints\latest.zip" -Episodes 5
    Eval-Model "games\breakout\epic_001_from_scratch\models\final\epic_001_final_model.zip" -Deterministic
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Checkpoint,
        
        [int]$Episodes = 2,
        [int]$Seconds = 120,
        [switch]$Deterministic,
        [switch]$NoVideo
    )
    
    Write-Host "🎯 Evaluating model: $Checkpoint" -ForegroundColor Magenta
    
    $Args = @(
        "training\eval.py",
        "--checkpoint", $Checkpoint,
        "--episodes", $Episodes,
        "--seconds", $Seconds
    )
    
    if ($Deterministic) { $Args += "--deterministic" }
    if ($NoVideo) { $Args += "--no-video" }
    
    python @Args
}

# 📋 PRESET COMMANDS
# ==================

function Start-BreakoutJourney {
    <#
    .SYNOPSIS
    Start a complete Breakout learning journey (3 epics)
    #>
    Write-Host "🚀 Starting complete Breakout learning journey..." -ForegroundColor Green
    
    Train-Game breakout 10h -Epic 1
    if ($LASTEXITCODE -eq 0) {
        Train-Game breakout 10h -Epic 2
        if ($LASTEXITCODE -eq 0) {
            Train-Game breakout 10h -Epic 3
        }
    }
}

function Start-TetrisJourney {
    <#
    .SYNOPSIS
    Start a complete Tetris learning journey (3 epics)
    #>
    Write-Host "🧩 Starting complete Tetris learning journey..." -ForegroundColor Green
    
    Train-Game tetris 10h -Epic 1
    if ($LASTEXITCODE -eq 0) {
        Train-Game tetris 10h -Epic 2
        if ($LASTEXITCODE -eq 0) {
            Train-Game tetris 10h -Epic 3
        }
    }
}

function Demo-AllGames {
    <#
    .SYNOPSIS
    Run quick demos of all supported games
    #>
    Write-Host "🎮 Running quick demos of all games..." -ForegroundColor Yellow
    
    $Games = @("breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger", "tetris")
    
    foreach ($Game in $Games) {
        Write-Host "`n--- Testing $Game ---" -ForegroundColor Cyan
        Quick-Train $Game
    }
}

# 📖 HELP FUNCTIONS
# =================

function Show-MLCommands {
    <#
    .SYNOPSIS
    Show all available ML training commands
    #>
    
    Write-Host @"

🎮 ML TRAINING COMMANDS
======================

TRAINING:
  Train-Game <game> <duration> [-Epic <1-3>] [-Test]
  Train-System <atari|gameboy> [-Hours <num>] [-Epic <1-3>] [-Test]
  Train-Batch "<game1,game2>" <duration> [-Epic <1-3>] [-Test]

VIDEO GENERATION:
  Make-Video <game> <length> [-Epic <1-3>] [-Quality <low|medium|high>]

EVALUATION:
  Eval-Model <checkpoint> [-Episodes <num>] [-Deterministic] [-NoVideo]

UTILITIES:
  Quick-Train <game>           # 1-minute test
  Check-Epic-Status           # Check training status
  Demo-AllGames              # Test all games

PRESETS:
  Start-BreakoutJourney      # Complete Breakout 3-epic journey
  Start-TetrisJourney        # Complete Tetris 3-epic journey

EXAMPLES:
  Train-Game breakout 5h
  Train-Game tetris 30min -Test
  Make-Video space_invaders 2h -Epic 2
  Train-System atari -Hours 8 -Epic 1
  Train-Batch "breakout,tetris" 1h -Test

SUPPORTED GAMES:
  Atari: breakout, pong, space_invaders, asteroids, pacman, frogger
  Gameboy: tetris

"@ -ForegroundColor White
}

# Show help on load
Write-Host "🎮 ML Training Commands Loaded!" -ForegroundColor Green
Write-Host "Type 'Show-MLCommands' for help" -ForegroundColor Yellow
