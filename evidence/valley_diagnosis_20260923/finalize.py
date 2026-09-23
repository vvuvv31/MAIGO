from pathlib import Path
import json,hashlib,subprocess,tarfile,shutil,yaml
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/valley_diagnosis_20260923'
j=json.loads((o/'comparison.json').read_text());q=json.loads((o/'run_quality.json').read_text());assert len(q['runs'])==9
a=j['cases']['highstat_1m'];b=j['cases']['matched_cu005_1m']
ph=json.loads((o/'phase_attribution.json').read_text());ph2=json.loads((o/'phase_attribution_cu005.json').read_text())
def row(c,dep,width=.25):return next(x for x in c['rows'] if x['depth_mm']==dep and x['window_width_mm']==width)
def pct(v):return f'{100*v:+.2f}%'
def rt(x):return f"{pct(x['relative_difference'])} ± {100*x['gpu_only_ratio_se']:.2f} pp"
lines=['| 深度 mm | Cu 0.25 mm 谷区差 | Cu 0.05 mm 谷区差 | 匹配后峰区差 | 匹配后 FWHM 差 |','|---:|---:|---:|---:|---:|']
for dep in [20,40,60,80,100,120]:
 ar,br=row(a,dep),row(b,dep)
 lines.append(f"| {dep} | {rt(ar['valley'])} | {rt(br['valley'])} | {pct(br['peak']['relative_difference'])} | {pct(br['central_fwhm_mm']['relative_difference'])} |")
old=row(a,100);new=row(b,100);oldw=row(a,100,10);neww=row(b,100,10)
phase=ph['depths']['100']['central_valleys'];phase2=ph2['depths']['100']['central_valleys']
binary=r/'build/oneapi-nvidia-minibeam/carbon_mc'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for s in iter(lambda:f.read(8*1024*1024),b''):h.update(s)
 return h.hexdigest()
assert sha(binary)==q['binary_sha256']
assert all(v['accepted'] and not any(v['urban'][k] for k in ['fatal','cap','guard','subulp']) for v in q['runs'].values())
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
assert 'd:Ge/Aperture/MaxStepSize = 0.05 mm' in (ref/'aperture.txt').read_text()
preset=r/'config/review_fullchain_topas_matched_1m.yaml'
source=r/'config/valley_cu0051m_s2.yaml'
if not preset.exists():preset.write_text('# Matched TOPAS Aperture and Water max step: 0.05 mm.\n# Research C12 EM-only comparison; not a default physics promotion.\n'+source.read_text())
assert yaml.safe_load(preset.read_text())==yaml.safe_load(source.read_text())
report=f'''# 谷区剂量偏高：铜步长错配的配对验证

当前最明确的修复方向是**先将铜准直器步长上限从 0.25 mm 对齐到 TOPAS 的 0.05 mm**，不改 Urban 散射幅度、尾部或 stopping scale。三种子、每种子 1,000,000 历史的配对实验中，100 mm 中央谷区差从 **{pct(old['valley']['relative_difference'])} → {pct(new['valley']['relative_difference'])}**；95–105 mm 平均谷区差从 **{pct(oldw['valley']['relative_difference'])} → {pct(neww['valley']['relative_difference'])}**。

这是配置/参考上下文修复，已有编译程序未改变。匹配配置已保存为 `config/review_fullchain_topas_matched_1m.yaml`，与已通过运行检查的百万历史第 2 个种子配置逐项一致。

## 为什么先修铜段

TOPAS 参考 `aperture.txt` 明确设置 `d:Ge/Aperture/MaxStepSize = 0.05 mm`，Snout 和各空气狭缝也为 0.05 mm。此前 GPU 全链配置 `minibeam_copper_max_step_mm=0.25`，因此此前的完整束线比较并非步长上限完全相同。TOPAS v4.2.3 的 `TsPhysicsManager.cc` 会自动注册 `G4StepLimiterPhysics`。

三组比较固定 Water_75eV、源分布、FP32 计分和能损/散射系数，所有核反应关闭。配对组使用相同源种子，仅改变铜段步长上限；该参数也控制 GPU 准直器内部空气段的分步。当前 GPU 空气分支只有能损而无 MCS，因此实验识别的是这项准直器步进配置的净影响，尚未单独分离铜角采样、铜能损/涨落和空气分段的贡献。

GPU 同一次运行内用稳定 `source_history` 连接入口与 100 mm 平面，谷区的 **{100*phase['ever_copper_fluence_fraction']:.1f}%** C12 曾穿过铜；其平均能量 **{phase['copper_energy_MeVu_mean']:.1f} MeV/u**，未穿铜部分平均 **{phase['never_copper_energy_MeVu_mean']:.1f} MeV/u**。穿铜 C12 在该处的 `S(E)/|direction_z|` 局部剂量代理占比约 **{100*phase['ever_copper_local_dose_proxy_fraction']:.1f}%**。该代理不是独立剂量计分，不能当作精确剂量分量，但说明谷区对铜段粒子的角度—能量联合分布很敏感。

匹配铜步长后，同种子的 100 mm 谷区平面记录数从 {phase['count']} 变为 {phase2['count']}，穿铜通量占比变为 {100*phase2['ever_copper_fluence_fraction']:.1f}%。详细能谱与角度统计保存在 `phase_attribution*.json`，未把 TOPAS MT EventID 当作源行号。

## 百万历史配对结果

下面使用原有固定中央谷区 `abs(abs(x)-1.8)<0.45 mm`，中央峰区 `abs(x)<0.25 mm`，单个 0.25 mm 深度 bin（例如 100 mm 对应中心 99.875 mm）。± 数字是 **GPU 三种子均值的 1 SE**，单位为百分点；不含 TOPAS 参考 SE。

{chr(10).join(lines)}

总水中沉积能量差：{pct(a['energy_MeV_per_source']['relative_difference'])} → **{pct(b['energy_MeV_per_source']['relative_difference'])}**；匹配后积分深度剂量 R80 差 **{b['R80_mm']['gpu']['mean']-b['R80_mm']['topas']:+.5f} mm**。R80 来自 0.25 mm 网格插值，不代表微米级物理精度。所有深度窗口、峰区、FWHM、左右谷区和配对变化 SE 见 `comparison.json`。

此前 250k × 3 的 100 mm 谷区 +14.81% 在新的 1M × 3 Cu 0.25 mm 运行中为 {pct(old['valley']['relative_difference'])}，说明旧数值包含明显采样波动，不能将 14.81% 当作已精确确定的系统偏差。其余深度和 10 mm 深度窗口也存在稳定的偏高，因此不能把全部问题归为统计噪声。深度平均只作为附加诊断，未替代原单-bin 指标。

![谷区对照](valley_diagnosis.png)

## 建议的修复顺序及剩余限制

1. 使用已验证的 **Cu 0.05 mm / water 0.05 mm / 物理体素边界** 配置重新建立完整束线比较基线。不要对旧 Cu 0.25 mm 的差异继续拟合散射尺度或位置相关能损。
2. 剩余剂量差先做同入口的 C12 相空间与 TOPAS 电子/非电子载流粒子剂量分解，分别检查谷区通量、条件能谱和单位路程能损。当前总剂量图不能证明剩余问题一定是 Urban 散射公式。
3. 水中步长响应不一致仍是独立问题；铜步长修复不等于它也已解决。空气 MCS 的缺失也是待核对项，不能用调大/调小铜 MCS 抵消。
4. TOPAS 完整束线仍只有一个 10M 历史参考，**本轮为有配对证据的改善，不能宣称完整统计剂量验收通过**。下一轮严格验收应补独立 TOPAS 种子并检查峰值、谷值、FWHM 和总能量的联合门槛。

## 运行与证据

共新增 9 个 GPU 运行：Cu 0.25 mm 的 1M × 3、Cu 0.05 mm 的 250k × 3 和 1M × 3，全部 quality accepted，Urban fatal/cap/guard/subulp 均为 0。最大能量账本相对残差 {max(abs(v['energy_residual_relative']) for v in q['runs'].values()):.3g}。最初一个配置因本项目 YAML 子集不接受 block list 而在解析阶段退出；改为已有逗号列表语法后运行，失败记录保留，未纳入物理结果。

代码工作树：`{r}`。原始 `MAIGO_pristine` 未修改。本轮没有修改模型源码或默认参数，只新增比较配置、诊断与报告。

```bash
cd {r}
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
build/oneapi-nvidia-minibeam/carbon_mc --config config/review_fullchain_topas_matched_1m.yaml
```

该命令的新配置名对应独立输出目录；再次运行前仍应另用新配置名以保留证据。原始相空间、所有剂量输出和完整日志保留在服务器，由 `artifact_manifest.json` 固定 SHA256。下载包包含报告、图、统计、脚本和配置，体积较大的原始输出作为 manifest 中的外部证据保留。
'''
(o/'RESULTS.md').write_text(report)
marker='## 2026-09-23 Cu step ceiling and valley diagnosis'
doc=r/'urban.md';s=doc.read_text()
if marker not in s:doc.write_text(s+f'''\n\n{marker}

The matched TOPAS full-chain aperture uses 0.05 mm max steps; the previous
GPU comparison used 0.25 mm. Paired 1M x 3 tests give the central 100 mm
valley difference {pct(old['valley']['relative_difference'])} -> {pct(new['valley']['relative_difference'])} when matching the copper step.
Use config/review_fullchain_topas_matched_1m.yaml for the new research baseline.
No coefficient retuning. Full statistical acceptance remains unestablished
because the TOPAS full-chain reference has one seed. See
evidence/valley_diagnosis_20260923/RESULTS.md for limitations and raw provenance.
''')
failed=r/'failed.md';marker='Treating Cu 0.25 mm as matched to TOPAS Cu 0.05 mm'
s=failed.read_text()
if marker not in s:failed.write_text(s+f'''\n\n### 2026-09-23: {marker}

The reference aperture and slit MaxStepSize is 0.05 mm. GPU 0.25 mm therefore
cannot be treated as the same configured step ceiling or used as a basis
for fitting water Urban/valley corrections. Paired 1M x 3 runs reduce the
100 mm valley difference {pct(old['valley']['relative_difference'])} -> {pct(new['valley']['relative_difference'])} at the matching 0.05 mm setting.
Retain 0.25 mm only as an explicitly unmatched step-study branch; revisit
its adequacy only after independent convergence/phase-space evidence.
Evidence: evidence/valley_diagnosis_20260923/RESULTS.md.
''')
subprocess.run(['git','diff','--check'],cwd=r,check=True)
files={p for p in o.iterdir() if p.is_file() and p.suffix in ['.md','.json','.png','.py','.log'] and p.name!='artifact_manifest.json'}
files.update([o/'valley_comparison.csv',o/'source_1m.csv',preset,r/'urban.md',r/'failed.md'])
files.update(r.glob('config/valley_*.yaml'))
external={binary,ref/'dose.bin',ref/'dose.binheader',ref/'aperture.txt',ref/'run1.txt',ref/'run_energy_scan.txt',ref/'run_single_center_em_only_water75ev_e250_10m.txt'}
external.update(o.glob('entrance*.csv'));external.update(o.glob('planes*.csv'))
for name in list(q['runs'])+[f'boundary_final_fullchain_s{i}' for i in [1,2,3]]:
 for base in ['dose.raw','dose.mhd','energy_ledger.json','quality_report.json']:external.add(r/f'out/{name}'/base)
for name in ['src/detail/sycl_device_math.inc','src/transport_sycl.cpp','data/stopping_power_water_geant4_11_3_2.csv','evidence/review_fix_20260923/active_water005.csv','evidence/review_fix_20260923/active_copper005.csv']:external.add(r/name)
manifest={'repo':str(r),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'binary_sha256':q['binary_sha256'],
 'files':{str(p.relative_to(r)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)},
 'external_evidence':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(external)}}
(o/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.add(o/'artifact_manifest.json')
archive=o/'valley_diagnosis_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(r)))
print('ARCHIVE',archive.stat().st_size,sha(archive))
print('SUMMARY',pct(old['valley']['relative_difference']),pct(new['valley']['relative_difference']),pct(oldw['valley']['relative_difference']),pct(neww['valley']['relative_difference']))
