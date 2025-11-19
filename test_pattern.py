#!/usr/bin/env python3
"""Test the RPS AI against a pattern-based opponent."""
import sys
from rps_neo import RealTimeEvolvingRPS_TF

def test_pattern_opponent(num_games=150):
    """Test AI against opponent playing repeating pattern."""
    print("=" * 60)
    print("TEST: AI vs Pattern Opponent (R-P-S repeating)")
    print("=" * 60)
    print(f"Running {num_games} games...\n")
    
    ai = RealTimeEvolvingRPS_TF(save_prefix="test_pattern", seed=42)
    pattern = ['rock', 'paper', 'scissors']
    
    results = {'w': 0, 'l': 0, 'd': 0}
    
    for i in range(num_games):
        ai_move = ai.get_move()
        opp_move = pattern[i % len(pattern)]  # Repeating pattern
        
        result = ai.move_result(ai_move, opp_move)
        results[result] += 1
        ai.update(ai_move, result)
        
        if (i + 1) % 30 == 0:
            wr = results['w'] / (i + 1) * 100
            print(f"Round {i+1}: Win Rate = {wr:.1f}%")
    
    # Final stats
    total = sum(results.values())
    win_rate = results['w'] / total * 100
    
    # Calculate win rate after pattern should be detected (after first 20 games)
    recent_results = list(ai.results[-100:])
    recent_wr = recent_results.count('w') / len(recent_results) * 100 if recent_results else 0
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Games: {total}")
    print(f"Wins: {results['w']} ({results['w']/total*100:.1f}%)")
    print(f"Losses: {results['l']} ({results['l']/total*100:.1f}%)")
    print(f"Draws: {results['d']} ({results['d']/total*100:.1f}%)")
    print(f"\nOverall Win Rate: {win_rate:.1f}%")
    print(f"Recent Win Rate (last 100): {recent_wr:.1f}%")
    
    # Evaluation
    print("\n" + "=" * 60)
    if win_rate > 55:
        print("✅ PASS: AI successfully exploited the pattern!")
    elif win_rate > 45:
        print("⚠️  PARTIAL: AI detected pattern but exploitation could be better")
    else:
        print("❌ FAIL: AI failed to exploit simple repeating pattern")
    print("=" * 60)
    
    return win_rate

if __name__ == "__main__":
    win_rate = test_pattern_opponent(150)
    sys.exit(0 if win_rate > 50 else 1)
