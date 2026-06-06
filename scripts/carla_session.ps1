# Start the CARLA agent bridge in this terminal (required for Cursor agent CARLA access).
# Run with CarlaUE4.exe already open.
Set-Location $PSScriptRoot\..
Write-Host "Starting CARLA agent bridge — leave this terminal open during agent CARLA work."
& .\.venv\Scripts\python.exe scripts\carla_agent_bridge.py
