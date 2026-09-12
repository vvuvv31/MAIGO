from pathlib import Path
import hashlib,json,csv
r=Path(__file__).resolve().parent
pins=json.loads((r/'input_sha256.json').read_text())
for name,h in pins.items():
 assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
assert json.loads((r/'process_audit.json').read_text())==json.loads((r/'expected_process_audit.json').read_text()),'process/model/dataset mismatch'
for case,m in json.loads((r/'migration_manifest.json').read_text()).items():
 counts=[sum(int(x['weight']) for x in csv.DictReader((r/case/f'shard_{i:02d}'/'spots.csv').open())) for i in range(1,6)]
 assert counts==m['shard_histories'] and sum(counts)==m['total_histories']
print('PASS: 105 input SHA pins, 18 projectile process/model/dataset matches, exact per-case history totals')
