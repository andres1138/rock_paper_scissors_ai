"""
Updated test files to work with Monster RPS AI.
This wrapper maintains compatibility with existing test structure.
"""

# test_random.py wrapper
def test_random(num_games=300):
    """Test Monster AI against pure random opponent."""
    import random
    from rps_monster import MonsterRPS
    
    print("=" * 60)
    print("  TEST: Monster AI vs Pure Random Opponent")
    print("=" * 60)
    print(f"Running {num_games} games...\n")
    
    ai = MonsterRPS(save_prefix="test_random")
    MOVES = ['rock', 'paper', 'scissors']
    
    results = {'w': 0, 'l': 0, 'd': 0}
    
    for i in range(num_games):
        ai_move = ai.get_move()
        opp_move = random.choice(MOVES)
        
        result = ai.move_result(ai_move, opp_move)
        results[result] += 1
        ai.update(ai_move, result)
        
        if (i + 1) % 50 == 0:
            wr = results['w'] / (i + 1) * 100
            print(f"Round {i+1}: Win Rate = {wr:.1f}%")
    
    total = sum(results.values())
    win_rate = results['w'] / total * 100
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Total: {total} | W: {results['w']} | L: {results['l']} | D: {results['d']}")
    print(f"Win Rate: {win_rate:.2f}%")
    print("=" * 60)
    
    if 30 <= win_rate <= 40:
        print("✅ PASS: Appropriate WR against random (33% baseline)")
    elif win_rate > 40:
        print("⚠️ WARNING: WR suspiciously high for pure random")
    else:
        print("❌ FAIL: WR too low")
    
    return win_rate


# test_pattern.py wrapper
def test_pattern(num_games=300):
    """Test Monster AI against pattern-based opponent."""
    from rps_monster import MonsterRPS
    
    print("=" * 60)
    print("  TEST: Monster AI vs Pattern Opponent")
    print("=" * 60)
    print(f"Running {num_games} games...\n")
    
    ai = MonsterRPS(save_prefix="test_pattern")
    pattern = ['rock', 'paper', 'scissors']
    
    results = {'w': 0, 'l': 0, 'd': 0}
    
    for i in range(num_games):
        ai_move = ai.get_move()
        opp_move = pattern[i % len(pattern)]
        
        result = ai.move_result(ai_move, opp_move)
        results[result] += 1
        ai.update(ai_move, result)
        
        if (i + 1) % 50 == 0:
            wr = results['w'] / (i + 1) * 100
            print(f"Round {i+1}: Win Rate = {wr:.1f}%")
    
    total = sum(results.values())
    win_rate = results['w'] / total * 100
    
    # Check recent performance (should be high after pattern detected)
    recent = list(ai.results)[-100:]
    recent_wr = (recent.count('w') / len(recent) * 100) if recent else 0
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Total: {total} | W: {results['w']} | L: {results['l']} | D: {results['d']}")
    print(f"Overall Win Rate: {win_rate:.2f}%")
    print(f"Recent WR (last 100): {recent_wr:.2f}%")
    print("=" * 60)
    
    if win_rate >= 70:
        print("✅ EXCELLENT! Pattern crushed!")
    elif win_rate >= 55:
        print("✅ GOOD! Pattern exploited well.")
    elif win_rate >= 45:
        print("⚠️ ACCEPTABLE - pattern detected but exploitation weak.")
    else:
        print("❌ FAIL: Failed to exploit simple pattern.")
    
    return win_rate


# test_counter.py wrapper
def test_counter(num_games=300):
    """Test Monster AI against counter-strategy opponent."""
    import random
    from rps_monster import MonsterRPS
    
    print("=" * 60)
    print("  TEST: Monster AI vs Counter-Strategy Opponent")
    print("=" * 60)
    print("Opponent plays what beats AI's last move\n")
    print(f"Running {num_games} games...\n")
    
    ai = MonsterRPS(save_prefix="test_counter")
    BEATEN_BY = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    MOVES = list(BEATEN_BY.keys())
    
    results = {'w': 0, 'l': 0, 'd': 0}
    last_ai_move = None
    
    for i in range(num_games):
        ai_move = ai.get_move()
        
        if last_ai_move:
            opp_move = BEATEN_BY[last_ai_move]
        else:
            opp_move = random.choice(MOVES)
        
        last_ai_move = ai_move
        
        result = ai.move_result(ai_move, opp_move)
        results[result] += 1
        ai.update(ai_move, result)
        
        if (i + 1) % 50 == 0:
            wr = results['w'] / (i + 1) * 100
            print(f"Round {i+1}: Win Rate = {wr:.1f}%")
    
    total = sum(results.values())
    win_rate = results['w'] / total * 100
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Total: {total} | W: {results['w']} | L: {results['l']} | D: {results['d']}")
    print(f"Win Rate: {win_rate:.2f}%")
    print("=" * 60)
    
    if win_rate >= 50:
        print("✅ EXCELLENT! Beat counter-predictor!")
    elif win_rate >= 40:
        print("✅ GOOD! Adapted to counter-strategy.")
    elif win_rate >= 33:
        print("⚠️ ACCEPTABLE - held ground against counter.")
    else:
        print("❌ FAIL: Counter-predictor dominated.")
    
    return win_rate


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == 'random':
            test_random()
        elif test_type == 'pattern':
            test_pattern()
        elif test_type == 'counter':
            test_counter()
    else:
        print("Usage: python test_wrappers.py [random|pattern|counter]")
