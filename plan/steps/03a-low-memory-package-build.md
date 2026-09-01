# 03a：低内存 Package 构建

## 问题

现有 Python compiler 同时持有 decoded raw、grouped records、两轮 normalized
records、global-energy dict/list、完整 interaction/product blobs，以及最终完整输出。
12 GB package 因而可能需要超过 100 GB 内存。

## 新构建路径

1. 每个 source projectile 独立生成 raw/exposure shard。
2. 每个 projectile shard 独立编译，限制单次峰值内存。
3. 编译器第一遍流式校验 CRC、identity、energy ledger，并把 normalized 定长记录
   写入磁盘 scratch；内存只保留 cell count 和统计量。
4. scratch 按 `(Zp,Ap,Zt,At,hybrid-bin,E,run,thread,event,track,sequence)` 做外部排序。
5. 第二遍顺序写 interaction array；product array 写独立临时文件，随后块复制到输出。
6. interaction 已按 identity/energy 排序时，global event index 是顺序 permutation；
   只需流式生成 node/offset，禁止构造千万级 Python dictionary。
7. header 先写占位，完成后 seek 回填；SHA256 分块读取，禁止 `read_bytes()`。
8. 多 projectile package 使用二进制流式 merge，禁止重新解码为 Python objects。

## 内存与磁盘门槛

- compiler RSS ≤8 GB，默认目标 ≤4 GB。
- scratch 可占 package 大小约 1.5--2.5 倍，必须位于 `/mnt/sda/wuwei`。
- 不在 `/tmp` 或系统盘存放 production scratch。
- 中断后 scratch manifest 支持恢复；成功后再清理。

## 等价性验收

- synthetic raw：旧/新 compiler loader interaction/product/node moments 完全一致。
- 事件 identity、parent continuation、local/neutral/unsupported ledger 完全一致。
- lookup hit/miss 和固定 seed 选择序列一致。
- package SHA256 不要求一致，但语义 checksum 必须一致。
- 当前 e400 数据做一次双编译，记录 wall time、MaxRSS、文件大小和 inspect diff。
