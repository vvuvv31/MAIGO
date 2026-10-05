# 八个当前预设与数据包

核对日期：2026-10-05。此清单替代2026-10-04的旧目录清单；历史物理验证保留在各自报告中。当前只有八个 config YAML 入口。RT07575 是头部病例，两个默认 CT minibeam 使用肺 CT，是独立场景。

## 当前运行路径

| config/ 预设 | 离子散射 | 最大离子步长 mm | 电子包 |
|---|---|---:|---|
| carbon_ct_broadbeam.yaml | Gaussian Fermi–Eyges | 0.5 | data/electrons/carbon/birth_bank.bin + data/electrons/shared/ |
| carbon_ct_minibeam.yaml | Urban | 0.1 | data/electrons/carbon/birth_bank.bin + data/electrons/shared/ |
| carbon_water_broadbeam.yaml | Gaussian Fermi–Eyges | 0.5 | 均匀水选区为空，不加载电子worker响应 |
| carbon_water_minibeam.yaml | Urban | 0.1 | data/electrons/carbon/birth_bank.bin + data/electrons/shared/ |
| proton_ct_broadbeam.yaml | Gaussian Fermi–Eyges | 0.5 | data/electrons/proton/birth_bank.bin + data/electrons/shared/ |
| proton_ct_minibeam.yaml | Urban | 0.1 | data/electrons/proton/birth_bank.bin + data/electrons/shared/ |
| proton_water_broadbeam.yaml | Gaussian Fermi–Eyges | 0.5 | 均匀水选区为空，不加载电子worker响应 |
| proton_water_minibeam.yaml | Urban | 0.1 | data/electrons/proton/birth_bank.bin + data/electrons/shared/ |

四个 broadbeam 共用 broadbeam_fast；broadbeam_electron_tracking 默认开启，CT 选区为低密度体素及5 mm邻域，均匀水选区为空。四个 minibeam 共用已经接入的 Urban/FP32电子worker，按粒子选择出生分布。全部预设开启 primary/secondary ion 涨落。队列、能量、输运和电子完成审计保留；carbon minibeam 正式剂量验收仍待完成。

## 共用离子 EM

所有预设使用 data/em/unified_em_v1.bin (EMCNT001) 和 unified_em_delta_moments_v2.bin (EMDMOMT2)，共256,458,544字节。只有26组成（25 Schneider＋Water_75eV）、18粒子、468记录；离子运行路径没有175状态目录、主机展开或GPU密度双节点插值。

每组成参考响应按真实密度和产生阈值连续修正，射程采用积分拟合。213,824字节材料标量表含密度采样，但不是完整EM响应状态。106,313,320字节的26组成基础源仅用于离线生成。电子数据仍有独立82节点，不能因离子26组成模型验收而宣称电子同样可以无误差缩减。

当前方法、验证和吞吐限制见[EM包说明](physics/unified_em_v1.md)、[连续密度报告](em_continuous_density_20261005.md)及[材料与方法](../mm_zh.md)。

## 归档

以下目录均不是默认运行依赖；当前GPU明确拒绝旧密度状态接口，不能隐式回退：

| 目录 | 内容 |
|---|---|
| data/archive/ion_em_native_20261005/ | 原生逐密度提取 |
| data/archive/ion_em_density_residual_20261005/ | 26组成＋稀疏密度残差压缩 |
| data/archive/ion_em_analytic_expanded_20261005/ | 原解析模型175状态展开接口及旧生成元数据 |

## 当前 YAML 声明的文件与目录

以下逐项从当前YAML生成，便于核对。它是显式声明清单，不穷尽index/bundle引用的传递文件，也不把条件关闭的组件称为实际加载。核反应、铜/空气散射与反冲数据仍各有用途；不能随离子EM归档一起删除。

### carbon_ct_broadbeam.yaml

| 参数 | 声明路径 |
|---|---|
| ct_grid_file | inputs/carbon_rt07575/rt07575_packed.bin |
| tps_spots_file | inputs/carbon_rt07575/spots.csv |
| tps_beam_model_file | inputs/carbon_rt07575/beam_model.csv |
| spots_upstream_air_stopping_power_file | data/stopping_power_air_geant4_11_3_2.csv |
| primary_stopping_power_file | data/stopping_power_water_geant4_11_3_2.csv |
| let_delta_electron_fraction_file | data/let_delta_electron_fraction_water_geant4_11_3_2.csv |
| particle_stopping_power_file | data/ion_stopping_power_water_geant4_11_3_2.csv |
| ct_schneider_primary_rate_file | data/schneider/schneider_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/schneider/schneider_physics_bundle_v2_1.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| ct_schneider_primary_cinel03_file | data/schneider/cinel03_c12_targets_v2_1.bin |
| material_electron_joint_pilot_file | data/electrons/shared/joint.bin |
| material_electron_birth_file | data/electrons/carbon/birth_bank.bin |
| material_photon_response_index_file | data/electrons/shared/photon/index.txt |
| material_electron_charged_index_file | data/electrons/shared/charged/index.txt |

### carbon_ct_minibeam.yaml

| 参数 | 声明路径 |
|---|---|
| primary_stopping_power_file | data/stopping_power_water_geant4_11_3_2.csv |
| ct_schneider_primary_rate_file | data/schneider/schneider_inelastic_rates_v2_1.bin |
| ct_schneider_primary_cinel03_file | data/schneider/cinel03_c12_targets_v2_1.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/schneider/schneider_physics_bundle_v2_1.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| urban_mcs_package_file | data/ct_urban/ct_copper_52.bin |
| minibeam_copper_stopping_power_file | data/carbon_copper/c12_copper_stopping_geant4_11_3_2.csv |
| minibeam_air_stopping_power_file | data/stopping_power_air_geant4_11_3_2.csv |
| minibeam_copper_cross_section_file | data/carbon_copper/c12_copper_inelastic_rate_geant4_11_3_2.csv |
| minibeam_copper_elastic_file | data/carbon_copper_elastic_20260926/carbon_copper_elastic.csv |
| minibeam_copper_inclxx_file | data/carbon_copper/copper_fragment_cascade_gapfilled.cinpkg |
| minibeam_copper_ion_stopping_power_file | data/ion_stopping_power_copper_geant4_11_3_2.csv |
| minibeam_copper_ion_cross_section_file | data/ion_cross_sections_copper_geant4_11_3_2.csv |
| minibeam_air_urban_package_file | data/air_urban_v2_20260926/urban_mcs_v2.bin |
| ct_grid_file | inputs/lung_ct/ct.bin |
| ct_schneider_file | data/HUtoMaterialSchneider.txt |
| all_ion_elastic_file | data/schneider/all_ion_elastic_v1.bin |
| elastic_recoil_stopping_file | data/schneider/elastic_recoil_stopping_v1.bin |
| minibeam_copper_loss_range_file | data/carbon_urban/active_copper005.csv |
| material_electron_birth_file | data/electrons/carbon/birth_bank.bin |
| material_electron_joint_pilot_file | data/electrons/shared/joint.bin |
| material_electron_charged_index_file | data/electrons/shared/charged/index.txt |
| material_photon_response_index_file | data/electrons/shared/photon/index.txt |

### carbon_water_broadbeam.yaml

| 参数 | 声明路径 |
|---|---|
| unified_water_material_file | data/water_unified/g4_water_material.json |
| primary_stopping_power_file | data/stopping_power_water_geant4_11_3_2.csv |
| particle_stopping_power_file | data/ion_stopping_power_water_geant4_11_3_2.csv |
| ct_schneider_primary_rate_file | data/schneider/schneider_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/schneider/schneider_physics_bundle_v2_1.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| ct_schneider_primary_cinel03_file | data/schneider/cinel03_c12_targets_v2_1.bin |

### carbon_water_minibeam.yaml

| 参数 | 声明路径 |
|---|---|
| tps_spots_file | inputs/carbon_water_minibeam/spots.csv |
| minibeam_copper_stopping_power_file | data/carbon_copper/c12_copper_stopping_geant4_11_3_2.csv |
| minibeam_air_stopping_power_file | data/stopping_power_air_geant4_11_3_2.csv |
| minibeam_copper_cross_section_file | data/carbon_copper/c12_copper_inelastic_rate_geant4_11_3_2.csv |
| minibeam_copper_elastic_file | data/carbon_copper_elastic_20260926/carbon_copper_elastic.csv |
| minibeam_copper_inclxx_file | data/carbon_copper/copper_fragment_cascade_gapfilled.cinpkg |
| minibeam_copper_ion_stopping_power_file | data/ion_stopping_power_copper_geant4_11_3_2.csv |
| minibeam_copper_ion_cross_section_file | data/ion_cross_sections_copper_geant4_11_3_2.csv |
| em_package_file | data/em/unified_em_v1.bin |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| unified_water_material_file | data/water_unified/g4_water_material.json |
| ct_schneider_primary_rate_file | data/schneider/schneider_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/schneider/schneider_physics_bundle_v2_1.json |
| primary_stopping_power_file | data/stopping_power_water_geant4_11_3_2.csv |
| particle_stopping_power_file | data/ion_stopping_power_water_geant4_11_3_2.csv |
| all_ion_elastic_file | data/schneider/all_ion_elastic_v1.bin |
| elastic_recoil_stopping_file | data/schneider/elastic_recoil_stopping_v1.bin |
| minibeam_air_urban_package_file | data/air_urban_v2_20260926/urban_mcs_v2.bin |
| ct_schneider_primary_cinel03_file | data/schneider/cinel03_c12_targets_v2_1.bin |
| urban_mcs_package_file | data/ct_urban/ct_copper_52.bin |
| material_electron_birth_file | data/electrons/carbon/birth_bank.bin |
| material_electron_joint_pilot_file | data/electrons/shared/joint.bin |
| material_photon_response_index_file | data/electrons/shared/photon/index.txt |
| material_electron_charged_index_file | data/electrons/shared/charged/index.txt |

### proton_ct_broadbeam.yaml

| 参数 | 声明路径 |
|---|---|
| ct_schneider_primary_rate_file | data/proton_therapy_20260926/proton_rates.bin |
| ct_schneider_primary_cinel03_file | data/proton_therapy_20260926/proton_cinel03.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/proton_therapy_20260926/physics_bundle.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| all_ion_elastic_file | data/proton_therapy_20260926/all_ion_elastic_proton_v1.bin |
| elastic_recoil_stopping_file | data/proton_therapy_20260926/elastic_recoil_stopping_water75.bin |
| ct_grid_file | inputs/proton_rt07575/patient_packed.bin |
| tps_beam_model_file | inputs/proton_rt07575/beam_model.csv |
| spots_upstream_air_stopping_power_file | data/proton_copper_20260926/proton_air_stopping.csv |
| primary_stopping_power_file | data/proton_therapy_20260926/proton_water_stopping.csv |
| tps_spots_file | inputs/proton_rt07575/part_01/spots.csv |
| material_electron_birth_file | data/electrons/proton/birth_bank.bin |
| material_electron_joint_pilot_file | data/electrons/shared/joint.bin |
| material_photon_response_index_file | data/electrons/shared/photon/index.txt |
| material_electron_charged_index_file | data/electrons/shared/charged/index.txt |

### proton_ct_minibeam.yaml

| 参数 | 声明路径 |
|---|---|
| primary_stopping_power_file | data/proton_therapy_20260926/proton_water_stopping.csv |
| ct_schneider_primary_rate_file | data/proton_therapy_20260926/proton_rates.bin |
| ct_schneider_primary_cinel03_file | data/proton_therapy_20260926/proton_cinel03.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/proton_therapy_20260926/physics_bundle.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| urban_mcs_package_file | data/ct_urban/ct_copper_52.bin |
| minibeam_copper_stopping_power_file | data/proton_copper_20260926/proton_copper_stopping.csv |
| minibeam_air_stopping_power_file | data/proton_copper_20260926/proton_air_stopping.csv |
| minibeam_copper_cross_section_file | data/proton_copper_20260926/proton_copper_inelastic_rate.csv |
| minibeam_copper_elastic_file | data/proton_copper_20260926/proton_copper_elastic.csv |
| minibeam_copper_inclxx_file | data/proton_copper_20260926/copper_cascade.cinpkg |
| minibeam_copper_ion_stopping_power_file | data/proton_copper_20260926/ion_copper_stopping.csv |
| minibeam_copper_ion_cross_section_file | data/ion_cross_sections_copper_geant4_11_3_2.csv |
| minibeam_air_urban_package_file | data/air_urban_v2_20260926/urban_mcs_v2.bin |
| minibeam_air_ionization_rate_file | data/proton_air_ionization_20260927/air_ionization_rate.csv |
| ct_grid_file | inputs/lung_ct/ct.bin |
| ct_schneider_file | data/HUtoMaterialSchneider.txt |
| all_ion_elastic_file | data/proton_therapy_20260926/all_ion_elastic_proton_v1.bin |
| elastic_recoil_stopping_file | data/proton_therapy_20260926/elastic_recoil_stopping_water75.bin |
| material_electron_birth_file | data/electrons/proton/birth_bank.bin |
| material_electron_joint_pilot_file | data/electrons/shared/joint.bin |
| material_photon_response_index_file | data/electrons/shared/photon/index.txt |
| material_electron_charged_index_file | data/electrons/shared/charged/index.txt |

### proton_water_broadbeam.yaml

| 参数 | 声明路径 |
|---|---|
| unified_water_material_file | data/water_unified/g4_water_material.json |
| primary_stopping_power_file | data/proton_therapy_20260926/proton_water_stopping.csv |
| particle_stopping_power_file | data/ion_stopping_power_water_geant4_11_3_2.csv |
| ct_schneider_primary_rate_file | data/proton_therapy_20260926/proton_rates.bin |
| ct_schneider_primary_cinel03_file | data/proton_therapy_20260926/proton_cinel03.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/proton_therapy_20260926/physics_bundle.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| all_ion_elastic_file | data/proton_therapy_20260926/all_ion_elastic_proton_v1.bin |
| elastic_recoil_stopping_file | data/proton_therapy_20260926/elastic_recoil_stopping_water75.bin |

### proton_water_minibeam.yaml

| 参数 | 声明路径 |
|---|---|
| unified_water_material_file | data/water_unified/g4_water_material.json |
| primary_stopping_power_file | data/proton_therapy_20260926/proton_water_stopping.csv |
| ct_schneider_primary_rate_file | data/proton_therapy_20260926/proton_rates.bin |
| ct_schneider_primary_cinel03_file | data/proton_therapy_20260926/proton_cinel03.bin |
| ct_schneider_secondary_rate_file | data/schneider/secondary_inelastic_rates_v2_1.bin |
| ct_schneider_secondary_cinel03_file | data/schneider/cinel03_secondary_targets_v2_1_14p.bin |
| ct_schneider_physics_bundle_file | data/proton_therapy_20260926/physics_bundle.json |
| em_delta_moments_file | data/em/unified_em_delta_moments_v2.bin |
| em_package_file | data/em/unified_em_v1.bin |
| all_ion_elastic_file | data/proton_therapy_20260926/all_ion_elastic_proton_v1.bin |
| elastic_recoil_stopping_file | data/proton_therapy_20260926/elastic_recoil_stopping_water75.bin |
| minibeam_copper_stopping_power_file | data/proton_copper_20260926/proton_copper_stopping.csv |
| minibeam_air_stopping_power_file | data/proton_copper_20260926/proton_air_stopping.csv |
| minibeam_copper_cross_section_file | data/proton_copper_20260926/proton_copper_inelastic_rate.csv |
| minibeam_copper_elastic_file | data/proton_copper_20260926/proton_copper_elastic.csv |
| minibeam_copper_inclxx_file | data/proton_copper_20260926/copper_cascade.cinpkg |
| minibeam_copper_ion_stopping_power_file | data/proton_copper_20260926/ion_copper_stopping.csv |
| minibeam_copper_ion_cross_section_file | data/ion_cross_sections_copper_geant4_11_3_2.csv |
| minibeam_air_urban_package_file | data/air_urban_v2_20260926/urban_mcs_v2.bin |
| minibeam_air_ionization_rate_file | data/proton_air_ionization_20260927/air_ionization_rate.csv |
| neutron_package_file | data/neutron/native/package.json |
| neutron_recoil_stopping_manifest | data/neutron/recoil/package.json |
| urban_mcs_package_file | data/ct_urban/ct_copper_52.bin |
| material_electron_birth_file | data/electrons/proton/birth_bank.bin |
| material_electron_joint_pilot_file | data/electrons/shared/joint.bin |
| material_photon_response_index_file | data/electrons/shared/photon/index.txt |
| material_electron_charged_index_file | data/electrons/shared/charged/index.txt |
