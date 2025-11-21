"""
Test Monster RPS AI against a MIXED opponent (60% pattern + 40% random).
This simulates a realistic opponent that's partially predictable.
"""

import random
from rps_monster import MonsterRPS

class MixedOpponent:
    """60% follows a pattern, 40% random - realistic scenario."""
    
    MOVES = ['rock', 'paper', 'scissors']
    
    def __init__(self, pattern_ratio=0.6):
        self.pattern_ratio = pattern_ratio
        self.round = 0
        self.patterns = [
            ['rock', 'paper', 'scissors'],  # Cycle
            ['rock', 'rock', 'paper', 'paper', 'scissors', 'scissors'],  # Double cycle
            ['rock', 'scissors', 'rock', 'scissors'],  # Alternating
        ]
        self.current_pattern = random.choice(self.patterns)
    
    def get_move(self):
        # Switch patterns occasionally
        if self.round > 0 and self.round % 100 == 0:
            self.current_pattern = random.choice(self.patterns)
        
        # 60% pattern, 40% random
        if random.random() < self.pattern_ratio:
            return self.current_pattern[self.round % len(self.current_pattern)]
        else:
            return random.choice(self.MOVES)
    
    def update(self):
        self.round += 1


def test_mixed(num_rounds=500):
    print("=" * 60)
    print("  TESTING MONSTER AI vs MIXED OPPONENT")
    print("=" * 60)
    print("Opponent: 60% pattern-based + 40% random")
    print("This tests pattern detection with noise...\n")
    
    ai = MonsterRPS(save_prefix="test_mixed")
    opponent = MixedOpponent(pattern_ratio=0.6)
    
    stats = {'w': 0, 'l': 0, 'd': 0}
    
    for round_num in range(1, num_rounds + 1):
        # Get moves
        ai_move = ai.get_move()
        opp_move = opponent.get_move()
        
        # Determine result
        result = ai.move_result(ai_move, opp_move)
        stats[result] += 1
        
        # Update
        ai.update(ai_move, result)
        opponent.update()
        
        # Print progress
        if round_num % 50 == 0:
            win_rate = stats['w'] / round_num * 100
            print(f"Round {round_num}: WR={win_rate:.1f}%")
    
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
    if win_rate >= 50:
        print("✅ EXCELLENT! AI extracted patterns despite noise!")
    elif win_rate >= 43:
        print("✅ GOOD! AI handled mixed strategy well.")
    elif win_rate >= 38:
        print("⚠️ ACCEPTABLE - detected some patterns.")
    else:
        print("❌ NEEDS WORK - AI struggled with noisy patterns.")
    
    return win_rate


if __name__ == "__main__":
    test_mixed(500)
