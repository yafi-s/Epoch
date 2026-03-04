@echo off
cd /d "%~dp0"

echo Starting Epoch...

if not exist results mkdir results

start "Epoch Scheduler" cmd /k ""build\scheduler\epoch_scheduler.exe" --listen-address 0.0.0.0:50051"

echo Waiting for scheduler to start...
timeout /t 5 /nobreak >nul

start "Epoch Worker 0" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w0 --log-level INFO"
start "Epoch Worker 1" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w1 --log-level INFO"
start "Epoch Worker 2" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w2 --log-level INFO"
start "Epoch Worker 3" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w3 --log-level INFO"
start "Epoch Worker 4" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w4 --log-level INFO"
start "Epoch Worker 5" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w5 --log-level INFO"
start "Epoch Worker 6" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w6 --log-level INFO"
start "Epoch Worker 7" cmd /k "poetry run python -m worker.main --scheduler-address localhost:50051 --worker-id w7 --log-level INFO"

echo Waiting for worker to connect...
timeout /t 5 /nobreak >nul

echo Running GA controller...
poetry run python run_ga.py

echo Done. Results saved to results\my_run.json
pause
