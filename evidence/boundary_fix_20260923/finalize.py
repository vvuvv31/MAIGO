from pathlib import Path
import hashlib,json,subprocess,difflib,tarfile,yaml

r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/boundary_fix_20260923';old=r/'evidence/review_fix_20260923'
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
j=json.loads((o/'comparison.json').read_text());q=json.loads((o/'final_run_quality.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(r/'build/oneapi-nvidia-minibeam/carbon_mc')==q['binary_sha256']
def pc(x):return f'{100*x:+.3f}%'
def row_table(case):
    c=j['cases'][case]
    lines=['| 深度中心 mm | 修复前 FWHM 差 | 修复后 FWHM 差 | 修复后峰区差 | 修复后谷区差 |','|---:|---:|---:|---:|---:|']
    for a,b in zip(c['rows'],c['before_boundary_fix']['rows']):
        lines.append(f"| {a['depth_mm']:.3f} | {pc(b['fwhm_mm']['relative_difference'])} | {pc(a['fwhm_mm']['relative_difference'])} | {pc(a['peak']['relative_difference'])} | {pc(a['valley']['relative_difference']) if a['valley']['relative_difference'] is not None else '未定义'} |")
    return '\n'.join(lines)
w=j['cases']['water050'];f=j['cases']['fullchain'];steps=j['step050_to_025']
report=f'''# 物理体素边界修复与 TOPAS/GPU 比较

本轮确认的边界衔接问题已修复并通过运行测试，**总剂量验收仍为 FAIL**。在这组六个深度中心，全链 FWHM 差的绝对值平均由约 2.10% 降至 1.13%，这是点估计改善，不是统计验收结论；100 mm 谷区仍高约 14.81%。水模 120 mm 的步长响应仍约 -3.86%，边界错配不能解释这一残差。

下一步应继续验证 Urban 的轨迹/步进移植等效性，但没有依据调散射尺度或尾部系数。优先比较同入口 C12 的角度、位置二阶矩及两者相关，再用 TOPAS 电子/非电子剂量分解区分轨迹误差与局部电子沉积近似。全链还需要锁定相同水入口记录，以隔离 Cu/空气段的角度—能量尾；当前不能将谷区偏高全部归因于水 Urban。

修复位置：`{r}`。原始 `MAIGO_pristine` 工作树未修改；没有 commit/push。

## 改动与适用范围

这次修复 primary C12 water Urban 与物理计分体素的衔接：用当前体素计算 pre/post safety，按真实过面更新 boundary 状态；内部面按出射方向确定归属，接受过面后只将相应坐标放到共享面上，取消该路径原有的位置 nudge；横向和入口反向逃逸正常记账，内部导航失败报告 fatal reason 11；不完整的末端深度体素明确拒绝。

启用条件是 research primary water Urban、`voxel_scorer_clamps_transport: true`、体素计分、均匀水模。CT、异质插入、分层、次级粒子与 production specialization 不在此修复范围内。全局开关默认值没有改变。没有调整散射系数、tail、stopping scale 或剂量归一化，保留 Water_75eV 和 GPU FP32 dose/atomics。

这些边界来自 TOPAS v4.2.3 `TsBox::ConstructVoxelStructure()` 的物理 replica，并非只为减小剂量差而添加。该修改对齐了已确认不一致的几何上下文，但不能据此宣称整个 GPU/Geant4 输运完全等效。

## 水模比较：0.05 mm 步长上限

同一批 20,000 条 250 MeV/u C12 入口记录、0.5 mm 均匀 beamlet，两引擎各三个独立种子，无 Cu；0.1 × 100 × 0.25 mm 体素。按源粒子数和真实体素质量换算，未拟合归一化。

- 总沉积能量 GPU/TOPAS 差：**{pc(w['energy_MeV_per_primary']['relative_difference'])}**。
- 积分深度剂量 R80 差：**{w['integrated_depth_R80_mm']['difference_mm']:+.6f} mm**。这是 0.25 mm 分箱数据的插值指标，不代表微米级物理精度。
- 冻结剂量验收：**{w['gate']['state']}**，退出码 {w['gate']['exit_code']}；具体条件见 `water050_dose_gate.json`。

{row_table('water050')}

![水模比较](water050_comparison.png)

## 两引擎相同步长变化

TOPAS 0.025 mm 对照已补到 20,000 histories × 3 seeds；不能继续使用此前“TOPAS 只有 2,000 histories”的判断。以下是深度 119.875 mm 的 FWHM 在步长上限 0.05 → 0.025 mm 时的变化：

| 引擎/版本 | FWHM 变化 |
|---|---:|
| TOPAS | {pc(steps['TOPAS'][-1]['relative_change'])} |
| GPU 修复前 | {pc(steps['GPU before'][-1]['relative_change'])} |
| GPU 物理体素修复后 | {pc(steps['GPU physical voxels'][-1]['relative_change'])} |

0.025 mm 的 GPU/TOPAS 剂量门槛为 **{j['cases']['water025']['gate']['state']}**。完整各深度均值、三个种子的数值、run SE 和步长对照保存在 `comparison.json`；三种子 SE 本身仍有不确定性。

先前固定能量的 `sqrt(t/X0) * (a+b*ln(t/X0))` 核心近似不能替代此实测步长响应，也不能单独证明散射公式需要调整。

![步长响应](step_response.png)

## Cu → 空气 → 水全链

GPU 250,000 histories × 3 seeds，对照现有 TOPAS 10,000,000 histories × 1 seed。两引擎源分布配置匹配，但没有共用同一批水入口记录。TOPAS 缺少独立重复，**只能作探索性比较，不能宣告统计验收通过**。

- 总水中沉积能量差：**{pc(f['energy_MeV_per_primary']['relative_difference'])}**。
- 积分深度剂量 R80 差：**{f['integrated_depth_R80_mm']['difference_mm']:+.6f} mm**。

{row_table('fullchain')}

![全链比较](fullchain_comparison.png)

## 验证和证据

- `carbon_mc`、`urban_localize` 构建通过；localize 包括所有 999 个横向面的双向归属、过面、反向、角点、安全距离和逃逸测试。
- 最终 {len(q['runs'])} 个 GPU 运行的 quality 均 accepted，Urban fatal/cap/guard/subulp 均为零；完整摘要 `final_run_quality.json`。
- 导航运行中部分初始边缘记录先被上游 absorbing geometry 吸收，因此不能把全部 200 条记录声称为完成了水中逃逸。进入水的斜入射记录用于检查横向逃逸；低能测试覆盖终止和射程边界。
- 最大绝对能量账本相对残差：{max(abs(a['energy_residual_relative']) for a in q['runs'].values()):.3g}。
- 首次 final localize 因未创建 CSV 输出目录失败，修正测试脚本后重跑通过；保留失败日志，未将其混入最终通过结果。
- 150.1 mm 水模配 0.25 mm 深度体素的无效配置已做实际运行检查，按预期以非零退出码明确拒绝（`partial_grid_rejection.json`）。
- `boundary_incremental.patch` 是本轮相对前一阶段源码快照的增量；`tracked_changes.patch` 包含工作树相对 HEAD 的全部改动，含原有 draft，不能全部归为本轮修改。

复现示例：

```bash
cd {r}
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
build/oneapi-nvidia-minibeam/carbon_mc --config config/boundary_final_water_s1.yaml
```

上述命令会使用同名输出目录；复验时应复制配置并改成新文件名，保留已有证据。`run_final.py` 会拒绝覆盖已存在的输出。
'''
(o/'RESULTS.md').write_text(report)
marker='## 2026-09-23 physical water voxel boundary follow-up'
urban=r/'urban.md'
note=f'''\n\n{marker}

Primary water Urban can now use physical voxel safety and boundary state when
`voxel_scorer_clamps_transport=true` in a homogeneous research water run.
Direction-owned shared faces replace the legacy nudges on this path; interior
navigation faults are fatal, transverse/upstream exits are accounted as escape.
Evidence: `evidence/boundary_fix_20260923/RESULTS.md` and hash manifest.

All 11 final GPU runs passed runtime quality checks; both water dose gates FAIL.
The 120 mm FWHM step response remains -3.86% (TOPAS -0.19%); boundary alignment
alone does not remove this discrepancy. Full-chain 100 mm valley remains +14.81%;
TOPAS has one seed, so full-chain statistical acceptance is not established.
No Urban scattering coefficient, tail, stopping scale or default was retuned.
'''
s=urban.read_text();s=s.split('\n\n'+marker)[0] if marker in s else s
urban.write_text(s+note)
failed=r/'failed.md';marker2='Boundary alignment alone as the explanation for the water step response'
s=failed.read_text()
if marker2 not in s:
    failed.write_text(s+f'''\n\n### 2026-09-23: {marker2}

Rejected conclusion, not a rollback of the physical-geometry fix. With current
voxel safety, boundary state and deterministic face navigation enabled, 20k x 3
water runs still show 120 mm FWHM changing -3.86% at 0.05 -> 0.025 mm (TOPAS
20k x 3: -0.19%). The full-chain 100 mm valley remains +14.81%. Runtime guards
and energy checks pass, but dose gates fail. Do not claim this boundary change
alone fixes the residual or retune Urban coefficients to compensate. Revisit
attribution only with matched C12 phase-space/step diagnostics and separated
electron deposition. Evidence: evidence/boundary_fix_20260923/RESULTS.md.
''')
paths=['src/detail/sycl_device_math.inc','src/transport_sycl.cpp','benchmark/carbonminibeam/test_urban_localize.cpp','include/carbon/transport_config.hpp']
inc=[]
for path in paths:
    p=r/path;b=o/'baseline'/path
    if not b.exists():
        assert 'diff --git a/'+path+' b/'+path not in (o/'baseline/tracked.patch').read_text()
        b.parent.mkdir(parents=True,exist_ok=True)
        b.write_bytes(subprocess.check_output(['git','show','HEAD:'+path],cwd=r))
    inc.extend(difflib.unified_diff(b.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='a/'+path,tofile='b/'+path))
(o/'boundary_incremental.patch').write_text(''.join(inc))
(o/'tracked_changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=r))
subprocess.run(['git','diff','--check'],cwd=r,check=True)

files={r/p for p in paths}
files.update([r/'AGENTS.md',r/'CMakeLists.txt',r/'build/oneapi-nvidia-minibeam/CMakeCache.txt',r/'build/oneapi-nvidia-minibeam/carbon_mc',r/'build/oneapi-nvidia-minibeam/urban_localize',r/'evidence/urban_d51e599_20260923/dose_gate.py',r/'evidence/urban_after_0f2c0ca_20260922/acceptance.yaml'])
files.add(r/'config/boundary_reject_partial.yaml')
files.update([r/'urban.md',r/'failed.md'])
files.update(p for p in o.glob('*') if p.is_file() and p.suffix not in ['.gz'] and p.name not in ['artifact_manifest.json'])
for stem,names in [('boundary_final',['navigation','terminal']+[f'{s}_s{i}' for s in ['water','water_step025','fullchain'] for i in [1,2,3]]),('review',[f'{s}_s{i}' for s in ['water_fixed','water_step025','fullchain'] for i in [1,2,3]])]:
    for name in names:
        c=r/f'config/{stem}_{name}.yaml';files.add(c)
        for basename in ['dose.raw','dose.mhd','quality_report.json','energy_ledger.json']:
            p=r/f'out/{stem}_{name}'/basename
            if p.exists():files.add(p)
for folder,stem in [(old,'topas_water'),(o,'topas_step025')]:
    for i in [1,2,3]:
        for base in ['input.txt','run.log','dose.bin','dose.binheader']:
            files.add(folder/f'{stem}_s{i}'/base)
for base in ['water_beamlet_source.csv','water_beamlet_source.phsp','water_beamlet_source.header','terminal_source.csv','single_center_250k.csv','active_water005.csv','active_copper005.csv']:
    files.add(old/base)
files.update(p for p in (o/'localize_final').glob('*') if p.is_file())
external={ref/'dose.bin',ref/'dose.binheader',ref.with_suffix('.log'),ref/'run_single_center_em_only_water75ev_e250_10m.txt'}
for c in r.glob('config/boundary_final_*.yaml'):
    for v in yaml.safe_load(c.read_text()).values():
        if isinstance(v,str) and v.startswith('/') and Path(v).is_file() and Path(v) not in files:external.add(Path(v))
manifest={'repo':str(r),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'binary_sha256':q['binary_sha256'],
    'files':{str(p.relative_to(r)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)},
    'external_references':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(external)}}
(o/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files.add(o/'artifact_manifest.json')
archive=o/'boundary_fix_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files):tar.add(p,arcname=str(p.relative_to(r)))
print('BUNDLE',archive,archive.stat().st_size,sha(archive),flush=True)
