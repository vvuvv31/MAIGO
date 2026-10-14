# Proton CT broadbeam dense restricted-ionization tail

Primary energy-loss fluctuations remain enabled. For the local-electron-off proton CT broadbeam primary path only, the Glandz explicit tail at Poisson mean p3≥4 uses an aggregate Gamma with shape μ²/v and scale v/μ. For tail lower energy w3 and upper cut c, μ=p3*w3*log(c/w3)/((c-w3)/c), v=p3*w3*c. This preserves its first two moments, not the zero-event atom or higher moments. The low-p3 tail, excitation component, aggregate hard-delta model, nuclear rates, packages, masses, caps and scoring are unchanged. Carbon, water, minibeam, secondary and generic paths retain the original explicit-tail sampler.

RT07575 head optimized-weight full plan, primary fluctuations ON, secondary OFF, electron transport OFF, elastic ON, forced 1mm cap. ABBA plus independent-seed BABA: 36M simulated/24M warm timed per arm, 6M+12M unique dose campaigns. RTX2080Ti paired throughput 555146→571448 primary/s (+2.937%); all four ratios positive. This is an incremental comparison, not a sum with earlier gains; first-batch timings are not whole CLI cold starts.

BODY≥0.4Gy local1%/1mm/search0.1mm: 6M control99.962977%, candidate99.951016%; 12M control99.958990%, candidate99.951301%. Reference is the historical optimized-weight TOPAS spot library, not a new independent replay. Scope is this paired RT07575 comparison, not new acceptance for every CT or GPU. Runtime energy/EM/coverage/completion/zero-overflow audits pass. Source campaigns exact; altered random draws cause interaction count changes disclosed in the experiment source audit.

GPU independent full restricted-loss fixtures (54×262144 draws per arm) have invalid0, maximum mean/variance standardized differences1.570/1.896. These establish first-two-moment consistency statistically, not whole-distribution equality.

Carbon screen +0.292% with mixed paired signs did not justify adopting this approximation. Production integration checks are recorded in tmp/proton_bb_dense_tail_gamma_20261013/. No changes to source normalization or per-spot seed allocation.
