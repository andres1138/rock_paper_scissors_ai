#!/usr/bin/env python3
"""Test the RPS AI against a pure random opponent."""
import sys
import random
from rps_neo import RealTimeEvolvingRPS_TF

def test_random_opponent(num_games=200):
    """Test AI against completely random opponent."""
    print("=" * 60)
    print("TEST: AI vs Pure Random Opponent")
    print("=" * 60)
    print(f"Running {num_games} games...\n")
    
    ai = RealTimeEvolvingRPS_TF(save_prefix="test_random", seed=42)
    MOVES = ['rock', 'paper', 'scissors']
    
    results = {'w': 0, 'l': 0, 'd': 0}
    
    for i in range(num_games):
        ai_move = ai.get_move()
        opp_move = random.choice(MOVES)  # Pure random
        
        result = ai.move_result(ai_move, opp_move)
        results[result] += 1
        ai.update(ai_move, result)
        
        if (i + 1) % 50 == 0:
            wr = results['w'] / (i + 1) * 100
            print(f"Round {i+1}: Win Rate = {wr:.1f}%")
    
    # Final stats
    total = sum(results.values())
    win_rate = results['w'] / total * 100
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Games: {total}")
    print(f"Wins: {results['w']} ({results['w']/total*100:.1f}%)")
    print(f"Losses: {results['l']} ({results['l']/total*100:.1f}%)")
    print(f"Draws: {results['d']} ({results['d']/total*100:.1f}%)")
    print(f"\nFinal Win Rate: {win_rate:.1f}%")
    
    # Evaluation
    print("\n" + "=" * 60)
    if 28 <= win_rate <= 38:
        print("✅ PASS: Win rate is appropriate for random opponent (30-35% expected)")
    elif win_rate > 40:
        print("⚠️  WARNING: Win rate too high - may be overfitting to noise")
    else:
        print("❌ FAIL: Win rate too low - AI has issues")
    print("=" * 60)
    
    return win_rate

if __name__ == "__main__":
    win_rate = test_random_opponent(200)
    sys.exit(0 if 28 <= win_rate <= 42 else 1)
