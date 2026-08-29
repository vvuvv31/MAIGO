# P0-1 Eq. 13–16 first fragment

Bug: `R=0` when `running_a_sum==0` ⇒ `k=0.4` ⇒ first fragment `E/A = 0.6 P`.

Fix: implicit formula so first fragment `E/A ≈ P`. Test at 95 MeV/u table kinematics, P=100–400: not 0.6×P.
