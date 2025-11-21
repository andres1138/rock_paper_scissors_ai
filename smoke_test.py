"""
Quick smoke test to verify Monster RPS AI is working correctly.
Tests basic functionality without running full test suite.
"""

from rps_monster import MonsterRPS
import random

def quick_test():
    print("🔥 MONSTER RPS AI - SMOKE TEST 🔥\n")
    
    print("1. Initializing AI...")
    ai = MonsterRPS(save_prefix="smoke_test")
    print(f"   ✅ AI initialized with {ai.K} strategies")
    
    print("\n2. Testing move generation...")
    moves = []
    for i in range(10):
        move = ai.get_move()
        moves.append(move)
        assert move in ai.MOVES, f"Invalid move: {move}"
    print(f"   ✅ Generated 10 valid moves: {', '.join(moves[:5])}...")
    
    print("\n3. Testing learning/update...")
    for i, move in enumerate(moves):
        result = random.choice(['w', 'l', 'd'])
        ai.update(move, result)
    print(f"   ✅ Processed 10 updates successfully")
    print(f"   Total rounds: {ai.total_rounds}")
    
    print("\n4. Testing pattern detection...")
    # Feed a clear pattern
    pattern = ['rock', 'paper', 'scissors'] * 10
    for opp_move in pattern:
        ai_move = ai.get_move()
        result = ai.move_result(ai_move, opp_move)
        ai.update(ai_move, result)
    
    ai._detect_patterns()
    print(f"   ✅ Pattern detection ran successfully")
    print(f"   Patterns detected: {len(ai.patterns)}")
    
    print("\n5. Testing opponent modeling...")
    ai._analyze_opponent_archetype()
    print(f"   ✅ Opponent archetype: {ai.opponent_archetype}")
    print(f"   Confidence: {ai.archetype_confidence:.2f}")
    
    print("\n6. Testing statistics...")
    stats = ai.stats()
    assert len(stats) > 0, "Stats output is empty"
    print("   ✅ Stats generated successfully")
    
    print("\n7. Testing save/load...")
    save_success = ai.save_state()
    assert save_success, "Save failed"
    print("   ✅ State saved successfully")
    
    # Create new AI and load
    ai2 = MonsterRPS(save_prefix="smoke_test")
    print(f"   ✅ State loaded: {ai2.total_rounds} rounds")
    
    print("\n" + "="*50)
    print("✅ ALL SMOKE TESTS PASSED!")
    print("="*50)
    print(f"\nMonster AI is ready to DOMINATE! 🔥")
    print(f"Strategies: {ai.K}")
    print(f"Patterns detected: {len(ai.patterns)}")
    print(f"Experience: {ai.total_rounds} rounds")
    
    return True

if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
