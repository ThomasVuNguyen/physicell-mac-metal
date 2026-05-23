#!/usr/bin/env python3
"""
test_hill_function.py — Hill function rule engine numerical tests

Reimplements RuleEngine::evaluateHill and verifies boundary conditions,
special cases, and numerical accuracy. No binary required.

Reference: physicell-mac-metal/src/rules.cpp::RuleEngine::evaluateHill

Hill function:
  increasing: base + (max - base) * s^n / (s^n + K^n)
  decreasing: base + (min - base) * s^n / (s^n + K^n)
"""

import math


def _hill(signal, base, max_val, min_val, half_max, hill_power, increasing):
    """Python reimplementation of RuleEngine::evaluateHill from rules.cpp."""
    K = half_max
    n = hill_power

    if K <= 0.0:
        return max_val if increasing else min_val

    if signal < 0.0:
        signal = 0.0

    s_n = signal ** n
    K_n = K ** n
    denom = s_n + K_n

    if denom <= 0.0:
        return base

    if increasing:
        return base + (max_val - base) * s_n / denom
    else:
        return base + (min_val - base) * s_n / denom


# ─── Boundary value tests ─────────────────────────────────────────────

def test_increasing_at_signal_zero(ctx=None):
    """Increasing Hill at s=0 returns base (s^n=0, so whole numerator=0)."""
    result = _hill(0.0, base=0.1, max_val=1.0, min_val=0.0,
                   half_max=5.0, hill_power=2.0, increasing=True)
    assert abs(result - 0.1) < 1e-12, f"Expected base=0.1, got {result}"
    return True, f"Hill(0) = {result:.4f} == base"


def test_decreasing_at_signal_zero(ctx=None):
    """Decreasing Hill at s=0 returns base."""
    result = _hill(0.0, base=1.0, max_val=1.0, min_val=0.0,
                   half_max=5.0, hill_power=2.0, increasing=False)
    assert abs(result - 1.0) < 1e-12, f"Expected base=1.0, got {result}"
    return True, f"Decreasing Hill(0) = {result:.4f} == base"


def test_increasing_at_half_max(ctx=None):
    """Increasing Hill at s=K returns base + 0.5*(max-base)."""
    K = 5.0
    base, max_val = 0.1, 1.0
    expected = base + 0.5 * (max_val - base)

    for n in [1.0, 2.0, 4.0]:
        result = _hill(K, base=base, max_val=max_val, min_val=0.0,
                       half_max=K, hill_power=n, increasing=True)
        assert abs(result - expected) < 1e-12, (
            f"Hill(K) at n={n}: expected {expected:.6f}, got {result:.6f}"
        )
    return True, f"Hill(K={K}) = {expected:.4f} = base + 0.5*(max-base)"


def test_decreasing_at_half_max(ctx=None):
    """Decreasing Hill at s=K returns base + 0.5*(min-base)."""
    K = 5.0
    base, min_val = 1.0, 0.0
    expected = base + 0.5 * (min_val - base)  # = 0.5

    for n in [1.0, 2.0]:
        result = _hill(K, base=base, max_val=base, min_val=min_val,
                       half_max=K, hill_power=n, increasing=False)
        assert abs(result - expected) < 1e-12, (
            f"Decreasing Hill(K) at n={n}: expected {expected:.6f}, got {result:.6f}"
        )
    return True, f"Decreasing Hill(K={K}) = {expected:.4f}"


def test_increasing_at_large_signal(ctx=None):
    """Increasing Hill at s >> K approaches max_val."""
    base, max_val, K = 0.0, 1.0, 1.0

    for n in [1.0, 2.0, 4.0]:
        result = _hill(1e6, base=base, max_val=max_val, min_val=0.0,
                       half_max=K, hill_power=n, increasing=True)
        assert result > 0.999999, (
            f"Hill(1e6) at n={n}: expected ≈1.0, got {result:.8f}"
        )
    return True, "Hill(s>>K) → max_val"


def test_decreasing_at_large_signal(ctx=None):
    """Decreasing Hill at s >> K approaches min_val."""
    base, min_val, K = 1.0, 0.0, 1.0

    for n in [1.0, 2.0]:
        result = _hill(1e6, base=base, max_val=base, min_val=min_val,
                       half_max=K, hill_power=n, increasing=False)
        assert result < 1e-6, (
            f"Decreasing Hill(1e6) at n={n}: expected ≈0.0, got {result:.8f}"
        )
    return True, "Decreasing Hill(s>>K) → min_val"


def test_n1_is_michaelis_menten(ctx=None):
    """With n=1, Hill = Michaelis-Menten: base + (max-base)*s/(s+K)."""
    base, max_val, K = 0.0, 1.0, 5.0

    for s in [0.0, 1.0, 5.0, 10.0, 100.0]:
        hill_result = _hill(s, base=base, max_val=max_val, min_val=0.0,
                            half_max=K, hill_power=1.0, increasing=True)
        mm_result = base + (max_val - base) * s / (s + K)
        assert abs(hill_result - mm_result) < 1e-12, (
            f"n=1 Hill({s}): Hill={hill_result:.8f}, MM={mm_result:.8f}"
        )
    return True, "n=1 Hill function matches Michaelis-Menten"


def test_n2_steeper_than_n1(ctx=None):
    """Hill with n=2 is steeper than n=1 around s=K (sigmoidal vs hyperbolic)."""
    K, base, max_val = 5.0, 0.0, 1.0
    # At s slightly above K, n=2 should give higher value than n=1
    # because n=2 concentrates response near K
    h1 = _hill(K * 1.5, base=base, max_val=max_val, min_val=0.0,
               half_max=K, hill_power=1.0, increasing=True)
    h2 = _hill(K * 1.5, base=base, max_val=max_val, min_val=0.0,
               half_max=K, hill_power=2.0, increasing=True)
    assert h2 > h1, f"n=2 Hill({K*1.5}) = {h2:.4f} should be > n=1 Hill = {h1:.4f}"
    return True, f"n=2 steeper: Hill(1.5K)={h2:.4f} > n=1: {h1:.4f}"


def test_degenerate_half_max_zero(ctx=None):
    """K=0: step function — returns max_val for increasing."""
    result_inc = _hill(0.0, base=0.1, max_val=1.0, min_val=0.0,
                       half_max=0.0, hill_power=2.0, increasing=True)
    result_dec = _hill(0.0, base=1.0, max_val=1.0, min_val=0.0,
                       half_max=0.0, hill_power=2.0, increasing=False)
    assert result_inc == 1.0, f"K=0 increasing should return max_val=1.0, got {result_inc}"
    assert result_dec == 0.0, f"K=0 decreasing should return min_val=0.0, got {result_dec}"
    return True, "K=0 acts as step function"


def test_negative_signal_clamped(ctx=None):
    """Negative signal is clamped to 0 (same as Hill(0) = base for increasing)."""
    for s_neg in [-1.0, -100.0, -1e10]:
        result = _hill(s_neg, base=0.5, max_val=1.0, min_val=0.0,
                       half_max=5.0, hill_power=2.0, increasing=True)
        expected = _hill(0.0, base=0.5, max_val=1.0, min_val=0.0,
                         half_max=5.0, hill_power=2.0, increasing=True)
        assert abs(result - expected) < 1e-12, (
            f"Hill({s_neg}) = {result:.6f} should equal Hill(0) = {expected:.6f}"
        )
    return True, "Negative signals clamped to Hill(0)"


def test_rule_chaining_independent(ctx=None):
    """Two rules on the same signal produce independent outputs (no cross-talk)."""
    signal = 3.0
    # Rule 1: signal increases proliferation
    out1 = _hill(signal, base=0.0, max_val=0.01, min_val=0.0,
                 half_max=5.0, hill_power=2.0, increasing=True)
    # Rule 2: signal decreases apoptosis
    out2 = _hill(signal, base=0.001, max_val=0.001, min_val=0.0,
                 half_max=1.0, hill_power=2.0, increasing=False)

    # Verify they are independent (different Hill curves)
    assert out1 != out2, "Two different rules should give different outputs"
    assert out1 > 0.0 and out2 > 0.0, "Both outputs should be positive"
    return True, f"Rule1={out1:.6f}, Rule2={out2:.6f}"


def test_hill_consistency_with_physicell_reference(ctx=None):
    """Cross-check against PhysiCell default: oxygen increases proliferation.

    PhysiCell heterogeneity sample default rule:
      oxygen increases cycle entry
      base=0, max=7.0e-4, K=38 mmHg, n=4
    """
    base, max_val, K, n = 0.0, 7.0e-4, 38.0, 4.0

    # At O2=0: rate should be base=0
    r0 = _hill(0.0, base=base, max_val=max_val, min_val=0.0,
               half_max=K, hill_power=n, increasing=True)
    assert abs(r0 - 0.0) < 1e-15, f"At O2=0: rate should be 0, got {r0}"

    # At O2=K=38: rate should be base + 0.5*(max-base) = 3.5e-4
    rK = _hill(K, base=base, max_val=max_val, min_val=0.0,
               half_max=K, hill_power=n, increasing=True)
    assert abs(rK - 3.5e-4) < 1e-15, f"At O2=K: rate should be 3.5e-4, got {rK:.4e}"

    # At O2>>K (s/K=26.3): rate approaches but doesn't exactly equal max=7e-4.
    # Residual error: max * K^n/(s^n+K^n) ≈ 7e-4 * (38/1000)^4 ≈ 1.4e-9
    r_high = _hill(1000.0, base=base, max_val=max_val, min_val=0.0,
                   half_max=K, hill_power=n, increasing=True)
    assert abs(r_high - 7.0e-4) < 1e-7, f"At high O2: rate should ≈7e-4, got {r_high:.4e}"

    return True, f"O2 rule: Hill(0)=0, Hill(K)=3.5e-4, Hill(∞)≈{r_high:.4e}"
