# 08：Neutral-origin 与缺失核过程

## 目标

在charged species对齐后，解释并修复无过滤total剩余缺口。

## 实施步骤

1. 用步骤01 scorer获得TOPAS neutral-origin 3D dose。
2. GPU输出neutron/gamma neutral birth phase space及generation/source。
3. 从TOPAS提取水中energy/direction/depth-conditioned neutral kerma kernel。
4. GPU独立生成neutral-origin dose，与charged/local合并为total。
5. 对nuclear elastic、decay、low-energy nuclear做TOPAS on/off A/B。
6. 仅当total贡献>0.2%或任一species>1%时纳入GPU。
7. elastic recoil及neutral产生的charged track保留正确origin。

## 验收

- [ ] charged、neutral、local、unclassified对total closure明确。
- [ ] neutral kernel在独立reference上积分和深度形状≤2%。
- [ ] 400 MeV/u total缺口不再由untracked neutral energy解释。
- [ ] 缺失过程纳入/排除均有量化A/B证据。
