"""
Comprehensive test suite for Monster RPS AI.
Runs all tests and generates a performance report.
"""

import sys
import importlib

def run_all_tests():
    print("🔥" * 30)
    print("  MONSTER RPS AI - COMPREHENSIVE TEST SUITE")
    print("🔥" * 30)
    print()
    
    results = {}
    
    # Test 1: Random Opponent
    print("\n📊 TEST 1: Random Opponent")
    print("-" * 60)
    try:
        from test_random import test_random
        wr = test_random(300)
        results['Random'] = wr
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results['Random'] = 0
    
    # Test 2: Pattern Opponent
    print("\n📊 TEST 2: Pattern Opponent")
    print("-" * 60)
    try:
        from test_pattern import test_pattern
        wr = test_pattern(300)
        results['Pattern'] = wr
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results['Pattern'] = 0
    
    # Test 3: Counter-Strategy Opponent
    print("\n📊 TEST 3: Counter-Strategy Opponent")
    print("-" * 60)
    try:
        from test_counter import test_counter
        wr = test_counter(300)
        results['Counter'] = wr
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results['Counter'] = 0
    
    # Test 4: Adaptive Opponent
    print("\n📊 TEST 4: Adaptive Opponent")
    print("-" * 60)
    try:
        from test_adaptive import test_adaptive
        wr = test_adaptive(300)
        results['Adaptive'] = wr
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results['Adaptive'] = 0
    
    # Test 5: Mixed Strategy Opponent
    print("\n📊 TEST 5: Mixed Strategy Opponent")
    print("-" * 60)
    try:
        from test_mixed import test_mixed
        wr = test_mixed(300)
        results['Mixed'] = wr
    except Exception as e:
        print(f"❌ Test failed: {e}")
        results['Mixed'] = 0
    
    # Final Report
    print("\n" + "🔥" * 30)
    print("  FINAL PERFORMANCE REPORT")
    print("🔥" * 30)
    print()
    
    for test_name, win_rate in results.items():
        status = "✅" if win_rate >= 45 else ("⚠️" if win_rate >= 38 else "❌")
        print(f"{status} {test_name:12s}: {win_rate:5.2f}%")
    
    avg_wr = sum(results.values()) / len(results) if results else 0
    print()
    print(f"📈 AVERAGE WIN RATE: {avg_wr:.2f}%")
    print()
    
    # Overall assessment
    if avg_wr >= 50:
        print("🏆 MONSTER STATUS: LEGENDARY! This AI is a BEAST!")
    elif avg_wr >= 45:
        print("🔥 MONSTER STATUS: ELITE! Excellent performance!")
    elif avg_wr >= 40:
        print("✅ MONSTER STATUS: STRONG! Good performance!")
    elif avg_wr >= 35:
        print("⚠️ MONSTER STATUS: DECENT. Room for improvement.")
    else:
        print("❌ MONSTER STATUS: NEEDS WORK.")
    
    print()
    print("🔥" * 30)


if __name__ == "__main__":
    run_all_tests()
