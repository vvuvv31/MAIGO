from pathlib import Path
import yaml,csv,json
root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');out=root/'evidence/review_fix_20260923'
original=Path('/mnt/sdb/wuwei/MAIGO_pristine')
# Full chain uses the existing matched Water_75eV, 0.05 mm-cut TOPAS reference.
N=250000
spots=out/'single_center_250k.csv'
spots.write_text('spot_id,x,y,energy,weight\n0,0,0,3000.0,250000\n')
base=yaml.safe_load((root/'config/beam_minibeam_single_center_em_only_e250_urban_v2_cuwater_10m.yaml').read_text())
for i in (1,2,3):
 cfg=dict(base);cfg.update(number_of_histories=N,tps_spots_file=str(spots),random_seed=2026092600+i,
    minibeam_water_urban_loss_range_file=str(out/'active_water005.csv'),
    minibeam_copper_loss_range_file=str(out/'active_copper005.csv'))
 for k,v in list(cfg.items()):
  if isinstance(v,str) and v.startswith('data/'):cfg[k]=str(root/v)
 (root/f'config/review_fullchain_s{i}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
# Small paired step-ceiling check. Same 2,000 source records on both engines.
source=out/'step_source_2k'
source.with_suffix('.csv').write_text('\n'.join((out/'water_beamlet_source.csv').read_text().splitlines()[:2001])+'\n')
source.with_suffix('.phsp').write_text('\n'.join((out/'water_beamlet_source.phsp').read_text().splitlines()[:2000])+'\n')
source.with_suffix('.header').write_text((out/'water_beamlet_source.header').read_text().replace('20000','2000'))
base=yaml.safe_load((root/'config/review_water_fixed_s1.yaml').read_text())
for label,step in [('050',.05),('025',.025)]:
 cfg=dict(base);cfg.update(number_of_histories=2000,tps_spots_file=str(source.with_suffix('.csv')),
    maximum_step_mm=step,minibeam_water_primary_urban_max_step_mm=step,random_seed=2026092711)
 (root/f'config/review_step{label}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 run=out/f'topas_step{label}';run.mkdir(exist_ok=True)
 text=(out/'topas_water_s1/input.txt').read_text().replace(str(out/'water_beamlet_source'),str(source))
 text=text.replace(str(out/'topas_water_s1'),str(run)).replace('= 0.05 mm\ni:Ge/Water/XBins',f'= {step} mm\ni:Ge/Water/XBins')
 (run/'input.txt').write_text(text)
# Actual low-energy/range-stop and phantom-exit paths, not only math helpers.
low=out/'terminal_source.csv'
with low.open('w') as f:
 w=csv.writer(f);w.writerow((out/'water_beamlet_source.csv').read_text().splitlines()[0].split(','))
 for i,e in enumerate([.1001,.12,.6,2.4,12,60,300,1200,3000,4800]*20):w.writerow([i+1,e,0,0,1,0,0,.0001,0,0,1,0,0,0,0,0,0,0])
cfg=dict(base);cfg.update(number_of_histories=200,tps_spots_file=str(low),random_seed=2026092712)
(root/'config/review_terminal.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
print('full-chain, paired step-ceiling and terminal configs prepared')
