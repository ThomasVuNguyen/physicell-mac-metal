#!/usr/bin/env python3
"""
PhysiCell Metal — Test Runner
Discovers and runs all test modules, compares Metal output against PhysiCell reference.

Usage:
    python3 tests/run_tests.py              # run all tests
    python3 tests/run_tests.py --unit       # unit tests only
    python3 tests/run_tests.py --e2e        # end-to-end tests only
    python3 tests/run_tests.py --bench      # performance benchmarks
    python3 tests/run_tests.py -v           # verbose output
"""

import argparse
import importlib
import os
import sys
import time
import traceback

# ─── Test discovery ───

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
BINARY = os.path.join(PROJECT_DIR, "build", "physicell-metal")
ORIGINAL_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "PhysiCell")

UNIT_MODULES = [
    # Analytical / pure-Python (no binary needed)
    "test_volume_ode",
    "test_hill_function",
    "test_secretion",
    # Binary readback (spawn binary, read .mat output)
    "test_diffusion_readback",
    "test_cycle_death",
    "test_motility",
    "test_config",
    "test_secretion_binary",
    "test_necrosis",
    "test_chemotaxis",
    "test_ki67",
    "test_csv_placement",
    "test_3d",
    "test_multitype",
    # Legacy source-inspection tests (weak but kept for coverage)
    "test_diffusion",
    "test_mechanics",
    "test_phenotype",
    "test_interactions",
]

E2E_MODULES = [
    "test_e2e",
]

PARITY_MODULES = [
    "test_parity",
]


class TestResult:
    """Holds result of a single test."""
    def __init__(self, name, passed, message="", duration=0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        dur = f"({self.duration:.2f}s)" if self.duration > 0.01 else ""
        msg = f"  → {self.message}" if self.message else ""
        return f"  {status} {self.name} {dur}{msg}"


class TestSuite:
    """Collects and runs tests from a module."""

    def __init__(self, module_name, verbose=False):
        self.module_name = module_name
        self.verbose = verbose
        self.results = []

    def run(self):
        """Import the module and run all test_* functions."""
        try:
            sys.path.insert(0, TEST_DIR)
            mod = importlib.import_module(self.module_name)
        except ImportError as e:
            self.results.append(TestResult(
                f"{self.module_name} (import)", False, str(e)))
            return

        # Find all test functions
        test_funcs = sorted([
            name for name in dir(mod)
            if name.startswith("test_") and callable(getattr(mod, name))
        ])

        if not test_funcs:
            self.results.append(TestResult(
                f"{self.module_name} (no tests)", False, "No test_* functions found"))
            return

        # Provide context to tests
        ctx = {
            "project_dir": PROJECT_DIR,
            "binary": BINARY,
            "original_dir": ORIGINAL_DIR,
            "test_dir": TEST_DIR,
            "verbose": self.verbose,
        }

        for func_name in test_funcs:
            func = getattr(mod, func_name)
            test_name = f"{self.module_name}::{func_name}"
            t0 = time.time()
            try:
                # Tests can accept ctx as a kwarg or positional
                try:
                    result = func(ctx)
                except TypeError:
                    result = func()

                if result is None:
                    # If function didn't raise, it passed
                    self.results.append(TestResult(test_name, True, duration=time.time() - t0))
                elif isinstance(result, bool):
                    self.results.append(TestResult(
                        test_name, result,
                        "returned False" if not result else "",
                        duration=time.time() - t0))
                elif isinstance(result, tuple) and len(result) == 2:
                    passed, msg = result
                    self.results.append(TestResult(
                        test_name, passed, msg, duration=time.time() - t0))
                else:
                    self.results.append(TestResult(test_name, True, duration=time.time() - t0))

            except AssertionError as e:
                self.results.append(TestResult(
                    test_name, False, str(e), duration=time.time() - t0))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if self.verbose:
                    msg += "\n" + traceback.format_exc()
                self.results.append(TestResult(
                    test_name, False, msg, duration=time.time() - t0))


def main():
    parser = argparse.ArgumentParser(description="PhysiCell Metal Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--e2e", action="store_true", help="Run E2E tests only")
    parser.add_argument("--parity", action="store_true",
                        help="Run PhysiCell↔Metal parity scenarios only")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Default: run unit + e2e (parity is opt-in because it builds & runs PhysiCell)
    any_explicit = args.unit or args.e2e or args.parity or args.bench
    run_unit = args.unit or not any_explicit
    run_e2e = args.e2e or not any_explicit
    run_parity = args.parity

    print("=" * 60)
    print("  PhysiCell Metal — Test Suite")
    print("=" * 60)
    print(f"  Binary:   {BINARY}")
    print(f"  Original: {ORIGINAL_DIR}")
    print()

    # Check binary exists
    if not os.path.exists(BINARY):
        print(f"❌ Binary not found: {BINARY}")
        print("   Run 'make' first to build the project.")
        sys.exit(1)

    all_results = []
    t_start = time.time()

    if run_unit:
        print("─── Unit Tests ───")
        for mod_name in UNIT_MODULES:
            suite = TestSuite(mod_name, verbose=args.verbose)
            suite.run()
            for r in suite.results:
                print(r)
            all_results.extend(suite.results)
        print()

    if run_e2e:
        print("─── End-to-End Tests ───")
        for mod_name in E2E_MODULES:
            suite = TestSuite(mod_name, verbose=args.verbose)
            suite.run()
            for r in suite.results:
                print(r)
            all_results.extend(suite.results)
        print()

    if run_parity:
        print("─── Parity Scenarios (PhysiCell ↔ Metal) ───")
        for mod_name in PARITY_MODULES:
            suite = TestSuite(mod_name, verbose=args.verbose)
            suite.run()
            for r in suite.results:
                print(r)
            all_results.extend(suite.results)
        print()

    # Summary
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed
    elapsed = time.time() - t_start

    print("=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed  ({elapsed:.1f}s)")
    if failed > 0:
        print()
        print("  Failed tests:")
        for r in all_results:
            if not r.passed:
                print(f"    ❌ {r.name}: {r.message}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
