"""
Test Monster RPS AI against an ADAPTIVE opponent that switches strategies.
This is a tough test - the opponent changes behavior periodically.
"""

import random
import sys
from rps_monster import MonsterRPS

class AdaptiveOpponent:
    """Opponent that switches between different strategies every N rounds."""
    
    MOVES = ['rock', 'paper', 'scissors']
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    BEATEN_BY = {v: k for k, v in BEATS.items()}
    
    def __init__(self, switch_every=30):
        self.switch_every = switch_every
        self.history = []
        self.ai_history = []
        self.round = 0
        self.current_strategy = 0
        
        # Available strategies to cycle through
        self.strategies = [
            'random',
            'pattern_cycle',
            'counter_ai',
            'frequency',
            'anti_frequency'
        ]
    
    def get_current_strategy(self):
        # Switch strategy periodically
        strategy_idx = (self.round // self.switch_every) % len(self.strategies)
        return self.strategies[strategy_idx]
    
    def get_move(self):
        strategy = self.get_current_strategy()
        
        if strategy == 'random':
            return random.choice(self.MOVES)
        
        elif strategy == 'pattern_cycle':
            # Play rock, paper, scissors cycle
            cycle = ['rock', 'paper', 'scissors']
            return cycle[self.round % 3]
        
        elif strategy == 'counter_ai':
            # Try to counter what AI played last
            if len(self.ai_history) >= 1:
                return self.BEATEN_BY[self.ai_history[-1]]
            return random.choice(self.MOVES)
        
        elif strategy == 'frequency':
            # Play most common move from AI
            if len(self.ai_history) >= 5:
                from collections import Counter
                most_common = Counter(self.ai_history[-20:]).most_common(1)[0][0]
                return most_common
            return random.choice(self.MOVES)
        
        elif strategy == 'anti_frequency':
            # Play least common move from AI
            if len(self.ai_history) >= 5:
                from collections import Counter
                counts = Counter(self.ai_history[-20:])
                least_common = min(counts.items(), key=lambda x: x[1])[0]
                return least_common
            return random.choice(self.MOVES)
        
        return random.choice(self.MOVES)
    
    def update(self, opp_move, ai_move):
        self.history.append(opp_move)
        self.ai_history.append(ai_move)
        self.round += 1


def test_adaptive(num_rounds=500):
    print("=" * 60)
    print("  TESTING MONSTER AI vs ADAPTIVE OPPONENT")
    print("=" * 60)
    print("This opponent switches strategies every 30 rounds")
    print("Testing adaptability and shift detection...\n")
    
    ai = MonsterRPS(save_prefix="test_adaptive")
    opponent = AdaptiveOpponent(switch_every=30)
    
    stats = {'w': 0, 'l': 0, 'd': 0}
    
    for round_num in range(1, num_rounds + 1):
        # Get moves
        ai_move = ai.get_move()
        opp_move = opponent.get_move()
        
        # Determine result
        result = ai.move_result(ai_move, opp_move)
        stats[result] += 1
        
        # Update both
        ai.update(ai_move, result)
        opponent.update(opp_move, ai_move)
        
        # Print progress
        if round_num % 50 == 0:
            win_rate = stats['w'] / round_num * 100
            current_strategy = opponent.get_current_strategy()
            print(f"Round {round_num}: WR={win_rate:.1f}% | Opponent Strategy: {current_strategy}")
    
    # Final stats
    total = num_rounds
    win_rate = stats['w'] / total * 100
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Total Rounds: {total}")
    print(f"Wins: {stats['w']} | Losses: {stats['l']} | Draws: {stats['d']}")
    print(f"Win Rate: {win_rate:.2f}%")
    print("=" * 60)
    
    # Evaluation
    if win_rate >= 55:
        print("✅ EXCELLENT! AI adapted to strategy shifts effectively!")
    elif win_rate >= 48:
        print("✅ GOOD! AI handled adaptive opponent well.")
    elif win_rate >= 40:
        print("⚠️ ACCEPTABLE but could improve adaptation speed.")
    else:
        print("❌ NEEDS WORK - AI struggled to adapt to strategy changes.")
    
    return win_rate


if __name__ == "__main__":
    test_adaptive(500)
