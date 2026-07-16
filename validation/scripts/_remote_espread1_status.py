#!/usr/bin/env python3
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.31.5", username="v", password="v", timeout=15)
_, o, _ = c.exec_command(
    "echo PROCESSES; "
    "ps -eo pid,etime,cmd | grep water_espread1 | grep -v grep || echo none; "
    "echo CSVS; "
    "ls -1 /home/v/gpu/validation/topas/output/e*_espread1_development_energy_deposit.csv 2>/dev/null; "
    "echo LOG_TAIL; "
    "tail -n 8 /home/v/gpu/validation/topas/output/espread1_all_development.nohup.log 2>/dev/null"
)
print(o.read().decode(errors="replace"), flush=True)
c.close()
