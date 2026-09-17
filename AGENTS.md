# MAIGO Agent Instructions

## GPU performance work

- Before proposing or implementing a performance experiment, search
  `failed.md`. Do not repeat a listed failed route unless its documented
  retry condition is satisfied by new evidence. Append every newly rejected
  candidate to `failed.md` before ending the task.
- Always use FP32 dose scoring/atomics for GPU performance development:
  `CARBON_DOSE_FP32=ON` and `CARBON_DOSE_FP64=OFF`.
- Do not configure, build, run, propose, or use an FP64 dose scorer as a
  performance or validation fallback unless the user explicitly overrides this
  rule in the current request. FP64 scoring is too slow for this project.
- For scheduling optimizations, FP32 atomic accumulation-order differences are
  expected and are not, by themselves, a reason to reject a candidate. Continue
  to require the transport/EM audits, quality checks, energy accounting, and
  queue-overflow checks to pass.
