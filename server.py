cd C:\src\floormetric-api
powershell -NoProfile -Command "[System.IO.File]::WriteAllText('C:\src\floormetric-api\server.py', (Get-Clipboard -Raw))"
findstr /C:"gemini-3.8-flash" server.py
git add server.py
git commit -m "Deploy complete server.py with gemini-3.8-flash"
git push -f origin main