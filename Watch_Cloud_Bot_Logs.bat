@echo off
title NQ Quant Bot - Live Cloud Logs (Drop01-Trade)
powershell -NoExit -Command "ssh root@178.128.197.187 'cd ~/nq_ibkr_bot && docker compose logs -f --tail=50 nq-bot'"
