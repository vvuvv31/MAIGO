# RT07575 Schneider25 / density82 joint experiment

This opt-in experiment uses 25 independently generated Schneider electron joint models and source/selected-node density scaling across the existing82 response nodes. Birth sampling and interface fallback retain the legacy library. FP32 dose, group32,0.25mm charged primary cap and <=0.05mm new-model steps. No dose fitting; unchanged TOPAS weighted DIJ reference.

- `smoke_result.json`: completed1,983,920-primary screen.
- `preview_12batches/paired_result.json`: completed23,807,040-primary paired preview vs the same12 legacy shards.
- `group000.log`: live119-batch process; each `[plan-shard-done]` is one finished shard.
- `campaign_status.json`, `completion.json`: runner state and final process return.
- `results/joint_comparison.json`, `REPORT_zh.md`: generated after236,086,432-primary full plan and comparisons complete.
- `gpu_memory.csv`: sampled device memory/utilization.
- `build.json`: successful frozen FP32 build and source hashes.

The12-batch preview gives71.8741% packet entry into the new model, zero invalid/overflow, BODY dose relativeL1 change0.07826%, pooled peak/valley changes below0.034%. These are first results, not formal accuracy acceptance. The full0.1% plan completed119 batches and236,086,432 primaries in1527.30s; all execution audits passed. Final results are in REPORT_zh.md. New1.3261GiB package supplements the retained legacy bank; no total-memory replacement or speed claim.

Generation and limitations: `../../../docs/schneider_joint_density_20261003.md` in the repository.
