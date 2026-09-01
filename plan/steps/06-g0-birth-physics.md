# 06：G0 Birth Physics 对齐

## 目标

在无secondary re-interaction时，对齐primary C12 reaction的final-state composition和kinematics。

## 实施步骤

1. 四能量按H/O、collision E/A、depth比较reaction count。
2. 按isotope比较multiplicity、yield、parent survival、birth E/A、角度和横/纵动量。
3. 拆分C10/11/12、B8/10/11、Be7/9/10、He3/4/6及p/d/t。
4. Secondary C峰位偏深时先修birth energy/parent continuation，不改stopping power。
5. He/Be缺口回到package exposure/channel sampling，不使用dose scale。
6. birth observables通过后才运行transport dose。

## 验收

- [ ] reaction fraction/CDF ≤1%。
- [ ] 主要isotope yield与mean birth E/A ≤2%或统计相容。
- [ ] parent survival与local/neutral energy ≤1%。
- [ ] G0 charged-origin species integral ≤3%。
