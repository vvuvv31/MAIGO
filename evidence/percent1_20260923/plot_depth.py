from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE = Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/evidence/percent1_20260923')
OUT = BASE / 'depth_plots_20260923'
OUT.mkdir(exist_ok=True)
data = json.loads((BASE / 'depth_profiles.json').read_text())
previous = json.loads((BASE / 'comparison.json').read_text())
z = np.asarray(data['z_mm'])
ng = np.asarray(data['gpu_histories'])
nt = np.asarray(data['topas_histories'])


def estimate(batches, counts):
    batches = np.asarray(batches, dtype=float)
    mean = np.average(batches, axis=0, weights=counts)
    variance = np.sum(counts[:, None] * (batches - mean) ** 2, axis=0)
    variance /= (len(counts) - 1) * counts.sum()
    return mean, np.sqrt(variance)


curves = {}
for region in ('peak', 'valley'):
    gm, gs = estimate(data['curves'][region]['gpu_batches'], ng)
    tm, ts = estimate(data['curves'][region]['topas_batches'], nt)
    ratio = np.divide(gm, tm, out=np.full_like(gm, np.nan), where=tm > 0)
    gvar = np.divide(gs, tm, out=np.zeros_like(gs), where=tm > 0) ** 2
    tvar = np.divide(ratio * ts, tm, out=np.zeros_like(ts), where=tm > 0) ** 2
    variance = gvar + tvar
    denom = gvar ** 2 / (len(ng) - 1) + tvar ** 2 / (len(nt) - 1)
    df = np.divide(variance ** 2, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    delta = (ratio - 1) * 100
    half = student_t.ppf(.975, df) * np.sqrt(variance) * 100
    threshold = .01 * tm.max()
    curves[region] = dict(
        gpu=gm, gpu_se=gs, topas=tm, topas_se=ts, difference_percent=delta,
        ci_low_percent=delta - half, ci_high_percent=delta + half,
        error_visible=(tm >= threshold) & np.isfinite(half), threshold=threshold,
    )
    # Confirm that the full-depth extraction reproduces the reported fixed ROIs.
    for row in previous['rows']:
        if row['window_width_mm'] != .25:
            continue
        index = np.argmin(abs(z - row['actual_depth_mm']))
        expected = row[region]
        assert np.isclose(gm[index], expected['gpu']['mean'], rtol=1e-12, atol=0)
        assert np.isclose(tm[index], expected['topas']['mean'], rtol=1e-12, atol=0)
        assert np.isclose(delta[index], expected['relative_difference'] * 100, atol=1e-10)
        assert np.allclose([delta[index] - half[index], delta[index] + half[index]],
                           np.asarray(expected['CI95_approx']) * 100, atol=1e-10)

font_names = {f.name for f in font_manager.fontManager.ttflist}
chinese_font = next((name for name in ['Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei', 'SimHei'] if name in font_names), None)
CN = chinese_font is not None
plt.rcParams.update({
    'font.family': chinese_font or 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'axes.unicode_minus': False,
    'axes.edgecolor': '#a8b0be', 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#dce1e8', 'grid.linewidth': .6, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
TOPAS, GPU = '#2563b8', '#e57722'
ERROR = {'peak': '#7552a0', 'valley': '#007d83'}


def make_figure(depth_max, stem):
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3), gridspec_kw={'height_ratios': [1, 1.06]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.15, top=.85, hspace=.34, wspace=.23)
    title = 'TOPAS / GPU：峰区、谷区剂量与相对误差' if CN else 'TOPAS / GPU: peak and valley dose versus depth'
    subtitle = ('GPU 1500万 / TOPAS 1600万源粒子 · Cu / 水步长 0.05 mm · 原始层厚 0.25 mm，未平滑'
                if CN else 'GPU 15M / TOPAS 16M source histories | Cu / water step 0.05 mm | 0.25 mm bins, no smoothing')
    fig.suptitle(title, fontsize=20, fontweight='bold', x=.075, ha='left', y=.97)
    fig.text(.075, .913, subtitle, color='#475569', fontsize=11)

    for column, region in enumerate(('peak', 'valley')):
        c = curves[region]
        region_title = ('峰区 Peak' if region == 'peak' else '谷区 Valley') if CN else region.capitalize()
        roi = '|x| < 0.25 mm' if region == 'peak' else '||x| − 1.8| < 0.45 mm'
        ax = axes[0, column]
        for engine, color in [('topas', TOPAS), ('gpu', GPU)]:
            mean, se = c[engine] * 1e9, c[engine + '_se'] * 1e9
            ax.fill_between(z, mean - se, mean + se, color=color, alpha=.1, linewidth=0)
            ax.plot(z, mean, color=color, linewidth=1.7, label=engine.upper(),
                    linestyle='--' if engine == 'gpu' else '-')
        ax.set_title(region_title + '  |  ' + roi, loc='left', pad=10)
        ax.set_ylabel('平均剂量 (nGy / 源粒子)' if CN else 'Mean dose (nGy / source history)')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', ncol=2, fontsize=11)

        ax = axes[1, column]
        ax.axhspan(-1, 1, color='#d7efda', zorder=0)
        ax.axhline(0, color='#64748b', linewidth=.8)
        for target in (-1, 1):
            ax.axhline(target, color='#82b38b', linewidth=.7, linestyle='--')
        valid = c['error_visible'] & (z <= depth_max)
        delta = np.where(valid, c['difference_percent'], np.nan)
        low = np.where(valid, c['ci_low_percent'], np.nan)
        high = np.where(valid, c['ci_high_percent'], np.nan)
        ax.fill_between(z, low, high, color=ERROR[region], alpha=.15, linewidth=0)
        ax.plot(z, delta, color=ERROR[region], linewidth=.85)
        points = [row for row in previous['rows'] if row['window_width_mm'] == .25 and row['depth_mm'] <= depth_max]
        px = [row['actual_depth_mm'] for row in points]
        py = [row[region]['relative_difference'] * 100 for row in points]
        ax.scatter(px, py, color=ERROR[region], edgecolor='white', linewidth=.8, s=33, zorder=4)
        for row, xx, yy in zip(points, px, py):
            if row['depth_mm'] in (40, 100):
                ax.annotate(f'{yy:+.2f}%', (xx, yy), xytext=(0, 13), textcoords='offset points',
                            ha='center', color=ERROR[region], fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.86, pad=1.4))
        # Include every plotted confidence bound; do not silently clip tail errors.
        bottom = min(float(np.nanmin(low)), -1.7)
        top = max(float(np.nanmax(high)), 1.7)
        padding = .07 * (top - bottom)
        ax.set_ylim(bottom - padding, top + padding)
        if depth_max > z[valid][-1] + 1:
            edge = z[valid][-1] + .125
            ax.axvspan(edge, depth_max, color='#eceff3', zorder=-1)
            ax.text((edge + depth_max) / 2, .94, '低剂量\n不作比值' if CN else 'Low dose\nratio omitted',
                    transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8, color='#687589')
        ax.set_title((region_title + '相对误差') if CN else (region.capitalize() + ' relative difference'), loc='left', pad=10)
        ax.set_ylabel('(GPU / TOPAS − 1) × 100 (%)')
        handles = [Patch(facecolor=ERROR[region], alpha=.15, label='近似95%置信区间' if CN else 'Approx. 95% CI'),
                   Patch(facecolor='#d7efda', label='±1%目标范围' if CN else '±1% target')]
        ax.legend(handles=handles, loc='lower left', fontsize=9, ncol=2)

    for ax in axes.flat:
        ax.set_xlim(0, depth_max)
        ax.set_xticks(np.arange(0, depth_max + 1, 20))
        ax.set_xlabel('水中深度 (mm)' if CN else 'Depth in water (mm)')
        ax.grid(True, alpha=.6)

    footer = (
        '固定横向 ROI 的绝对每源粒子平均剂量；两侧谷区合并。上排阴影为 ±1SE，下排为独立批次合并的近似95%区间。\n'
        '相对误差仅显示 TOPAS 剂量 ≥ 本 ROI 最大剂量1%的层；未对剂量作归一化拟合。六个圆点对应此前报告的原始层。\n'
        '当前主比较使用 local 电子沉积模式；不是实验性电子迁移模式或 Cu 0.025 mm 扫描结果。'
        if CN else
        'Absolute per-source mean dose in fixed transverse ROIs; both valleys combined. Top shading: ±1 SE; bottom: approximate independent-batch 95% CI.\n'
        'Ratios shown only where TOPAS dose ≥ 1% of that ROI maximum; no fitted normalization. Dots mark the six previously reported layers.\n'
        'Main comparison: local electron deposition, Cu / water 0.05 mm; not the experimental electron-response or Cu 0.025 mm runs.'
    )
    fig.text(.075, .033, footer, fontsize=9, color='#526175', va='bottom', linespacing=1.6)
    fig.savefig(OUT / (stem + '.png'), dpi=190)
    fig.savefig(OUT / (stem + '.pdf'))
    plt.close(fig)


make_figure(140, 'topas_gpu_peak_valley_depth')
make_figure(120, 'topas_gpu_peak_valley_depth_0_120mm')

with (OUT / 'depth_profiles.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['depth_mm', 'region', 'topas_Gy_per_source', 'gpu_Gy_per_source',
                     'topas_SE_Gy_per_source', 'gpu_SE_Gy_per_source', 'relative_difference_percent',
                     'CI95_low_percent', 'CI95_high_percent', 'ratio_shown_in_plot'])
    for region, c in curves.items():
        for i, depth in enumerate(z):
            writer.writerow([depth, region, c['topas'][i], c['gpu'][i], c['topas_se'][i], c['gpu_se'][i],
                             c['difference_percent'][i], c['ci_low_percent'][i], c['ci_high_percent'][i],
                             int(c['error_visible'][i])])

readme = '''# TOPAS/GPU 峰区、谷区深度曲线

- 主图 `topas_gpu_peak_valley_depth.png`：0–140 mm，包含 Bragg 峰及末端。
- 放大图 `topas_gpu_peak_valley_depth_0_120mm.png`：0–120 mm。
- 同名 PDF 可用于放大或导出；`depth_profiles.csv` 为全部 1000 层数据。
- 数据是既有主比较 GPU 15×1M 和 TOPAS 10M+3×2M 的全部原始剂量层，没有新运行或挑选种子。
- 层厚 0.25 mm，未作深度平滑；峰区 |x|<0.25 mm，谷区 ||x|−1.8|<0.45 mm，两侧谷区合并。
- 上排剂量单位为 nGy/源历史，阴影 ±1SE；下排相对差 `(GPU/TOPAS−1)×100%`，阴影为近似 Welch-t 95%区间。两端按源历史数加权，方差由独立批次估计，TOPAS仅4批。
- 比值仅在 TOPAS 剂量 ≥ 对应 ROI 全深度最大剂量的1%时绘制；灰色末端不显示比值。此阈值仅用于作图，不改变此前六深度的验收定义，原始数值保留在 CSV 中。
- 主比较 Cu/水步长均0.05 mm，local电子沉积模式；不是 Cu0.025 mm 或实验性电子迁移诊断。
- 作图数据在六个报告深度逐项核对剂量、相对误差及95%区间，与 `comparison.json` 一致。
- 此前报告只列六个固定深度；完整曲线还显示入口峰区的较大偏差，不能由六个点推广为全深度精度。
'''
(OUT / 'README.md').write_text(readme)
shutil.copy2(BASE / 'depth_profiles.json', OUT / 'depth_profiles.json')
shutil.copy2(Path(__file__), OUT / 'plot_depth.py')
manifest = {'files': [], 'source_comparison_sha256': hashlib.sha256((BASE / 'comparison.json').read_bytes()).hexdigest(),
            'chinese_font': chinese_font, 'verified_against_previous_depths': [20, 40, 60, 80, 100, 120]}
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.name != 'manifest.json':
        manifest['files'].append({'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'output': str(OUT), 'font': chinese_font, 'files': manifest['files']}, ensure_ascii=False), flush=True)
