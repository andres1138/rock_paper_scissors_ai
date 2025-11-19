#!/usr/bin/env python3
"""Test the RPS AI against a counter-strategy opponent."""
import sys
from rps_neo import RealTimeEvolvingRPS_TF

def test_counter_opponent(num_games=200):
    """Test AI against opponent who counters AI's last move."""
    print("=" * 60)
    print("TEST: AI vs Counter-Strategy Opponent")
    print("(Opponent plays what beats AI's last move)")
    print("=" * 60)
    print(f"Running {num_games} games...\n")
    
    ai = RealTimeEvolvingRPS_TF(save_prefix="test_counter", seed=42)
    BEATEN_BY = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    
    results = {'w': 0, 'l': 0, 'd': 0}
    last_ai_move = None
    
    for i in range(num_games):
        ai_move = ai.get_move()
        
        # Opponent plays what beats AI's LAST move
        if last_ai_move:
            opp_move = BEATEN_BY[last_ai_move]
        else:
            # First move is random
            import random
            opp_move = random.choice(list(BEATEN_BY.keys()))
        
        last_ai_move = ai_move
        
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
    if win_rate > 40:
        print("✅ PASS: AI adapted to counter-strategy effectively")
    elif win_rate > 30:
        print("⚠️  PARTIAL: AI showed some adaptation but could be better")
    else:
        print("❌ FAIL: AI failed to adapt to counter-strategy")
    print("=" * 60)
    
    return win_rate

if __name__ == "__main__":
    win_rate = test_counter_opponent(200)
    sys.exit(0 if win_rate > 35 else 1)
