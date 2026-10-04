"""Read-only progress for the single-process RT07575 rerun."""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
text = (ROOT / 'group000.log').read_text()
rows = re.findall(r'\[batch-wall\] index=(\d+) seconds=([\d.eE+-]+) histories=(\d+)', text)
recent = rows[-10:]
histories = sum(int(row[2]) for row in rows)
rate = sum(int(row[2]) for row in recent) / sum(float(row[1]) for row in recent) if recent else None
launch = json.loads((ROOT / 'launch.json').read_text())
requested = json.loads((ROOT / 'manifest.json').read_text())['histories']
print(json.dumps(dict(completed_batches=len(rows), completed_histories=histories,
                     elapsed_seconds=time.time()-launch['started_unix'],
                     recent_primary_per_second=rate,
                     estimated_remaining_mc_seconds=(requested-histories)/rate if rate else None,
                     bank_cache_hits=text.count('[electron-bank-cache] hit'),
                     last_batch_seconds=float(rows[-1][1]) if rows else None), indent=2))
