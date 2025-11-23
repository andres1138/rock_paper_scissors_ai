#!/usr/bin/env python3
"""
Test suite for RPS AI - Tests against various opponent types
"""

import sys
import random
from collections import Counter


class OpponentBot:
    """Base class for test opponents."""
    
    MOVES = ['rock', 'paper', 'scissors']
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    
    def __init__(self, name):
        self.name = name
        self.history = []
        self.opponent_history = []
    
    def get_move(self):
        """Get opponent's move - override in subclasses."""
        raise NotImplementedError
    
    def update(self, own_move, opponent_move):
        """Update opponent's history."""
        self.history.append(own_move)
        self.opponent_history.append(opponent_move)
    
    def move_result(self, move1, move2):
        """Determine result from move1's perspective."""
        if move1 == move2:
            return 'd'
        if self.BEATS[move1] == move2:
            return 'w'
        return 'l'


class PatternBot(OpponentBot):
    """Always plays in a fixed pattern: Rock -> Paper -> Scissors."""
    
    def __init__(self):
        super().__init__("Pattern Player (R->P->S)")
        self.sequence = ['rock', 'paper', 'scissors']
        self.index = 0
    
    def get_move(self):
        move = self.sequence[self.index % len(self.sequence)]
        self.index += 1
        return move


class FrequencyBot(OpponentBot):
    """Always plays the move it has played most often."""
    
    def __init__(self):
        super().__init__("Frequency Player")
        self.favorite = random.choice(self.MOVES)
    
    def get_move(self):
        # 70% favorite, 30% random
        if random.random() < 0.7:
            return self.favorite
        return random.choice(self.MOVES)


class CounterBot(OpponentBot):
    """Always counters the AI's previous move."""
    
    def __init__(self):
        super().__init__("Counter Player")
        self.BEATEN_BY = {v: k for k, v in self.BEATS.items()}
    
    def get_move(self):
        if len(self.opponent_history) == 0:
            return random.choice(self.MOVES)
        
        # Play what beats the AI's last move
        return self.BEATEN_BY[self.opponent_history[-1]]


class RandomBot(OpponentBot):
    """Plays completely randomly (Nash equilibrium)."""
    
    def __init__(self):
        super().__init__("Random Player")
    
    def get_move(self):
        return random.choice(self.MOVES)


class MixedBot(OpponentBot):
    """Mixes strategies - mostly pattern with some randomness."""
    
    def __init__(self):
        super().__init__("Mixed Player")
        self.sequence = ['rock', 'rock', 'paper', 'scissors']
        self.index = 0
    
    def get_move(self):
        # 60% pattern, 40% random
        if random.random() < 0.6:
            move = self.sequence[self.index % len(self.sequence)]
            self.index += 1
            return move
        return random.choice(self.MOVES)


def run_test(ai, opponent, num_rounds=100, verbose=False):
    """Run a test of AI vs opponent for num_rounds."""
    
    print(f"\n{'='*60}")
    print(f"Testing against: {opponent.name}")
    print(f"Number of rounds: {num_rounds}")
    print(f"{'='*60}")
    
    wins = 0
    losses = 0
    draws = 0
    
    # Track performance over time
    early_wins = 0  # First 30 rounds
    late_wins = 0   # Last 30 rounds
    
    for round_num in range(1, num_rounds + 1):
        # Get moves
        ai_move = ai.get_move()
        opp_move = opponent.get_move()
        
        # Determine result
        result = opponent.move_result(ai_move, opp_move)
        
        if result == 'w':
            wins += 1
            if round_num <= 30:
                early_wins += 1
            if round_num > num_rounds - 30:
                late_wins += 1
        elif result == 'l':
            losses += 1
        else:
            draws += 1
        
        # Update both
        ai.update(ai_move, result)
        opponent.update(opp_move, ai_move)
        
        # Verbose output
        if verbose and round_num % 10 == 0:
            wr = wins / round_num * 100
            print(f"Round {round_num}: WR = {wr:.1f}% (W:{wins} L:{losses} D:{draws})")
    
    # Final stats
    total = num_rounds
    win_rate = wins / total * 100
    early_wr = early_wins / min(30, num_rounds) * 100
    late_wr = late_wins / min(30, num_rounds) * 100
    
    print(f"\n📊 RESULTS:")
    print(f"   Total: W:{wins} L:{losses} D:{draws}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Early WR (first 30): {early_wr:.1f}%")
    print(f"   Late WR (last 30): {late_wr:.1f}%")
    print(f"   Learning Improvement: {late_wr - early_wr:+.1f}%")
    
    # Verdict
    print(f"\n🎯 VERDICT: ", end="")
    
    if isinstance(opponent, PatternBot):
        # Should dominate pattern players
        if late_wr >= 80:
            print("✅ EXCELLENT - Learned pattern effectively")
            return True
        elif late_wr >= 60:
            print("⚠️  GOOD - Pattern partially learned")
            return True
        else:
            print("❌ POOR - Failed to learn simple pattern")
            return False
    
    elif isinstance(opponent, FrequencyBot):
        # Should beat frequency players
        if late_wr >= 65:
            print("✅ EXCELLENT - Exploited frequency bias")
            return True
        elif late_wr >= 50:
            print("⚠️  GOOD - Partially exploited")
            return True
        else:
            print("❌ POOR - Failed to exploit frequency")
            return False
    
    elif isinstance(opponent, CounterBot):
        # Counter bots are tough, should at least not lose badly
        if late_wr >= 50:
            print("✅ EXCELLENT - Beat the counter-predictor")
            return True
        elif late_wr >= 40:
            print("⚠️  GOOD - Held own against counter-predictor")
            return True
        else:
            print("❌ POOR - Exploited by counter-predictor")
            return False
    
    elif isinstance(opponent, RandomBot):
        # Should stay close to Nash (33%)
        if 28 <= late_wr <= 45:
            print("✅ EXCELLENT - Nash equilibrium maintained")
            return True
        elif 20 <= late_wr <= 50:
            print("⚠️  ACCEPTABLE - Close to Nash")
            return True
        else:
            print("❌ POOR - Deviated too far from Nash")
            return False
    
    elif isinstance(opponent, MixedBot):
        # Should adapt and win
        if late_wr >= 55:
            print("✅ EXCELLENT - Adapted to mixed strategy")
            return True
        elif late_wr >= 45:
            print("⚠️  GOOD - Partially adapted")
            return True
        else:
            print("❌ POOR - Failed to adapt")
            return False
    
    return False


def main():
    """Run all tests."""
    print("🧪" * 30)
    print("      RPS AI TEST SUITE")
    print("🧪" * 30)
    
    # Import the AI
    try:
        from rps_smart import SmartRPS
        ai_class = SmartRPS
        ai_name = "Smart RPS"
    except ImportError:
        try:
            from rps_monster import MonsterRPS
            ai_class = MonsterRPS
            ai_name = "Monster RPS"
        except ImportError:
            print("❌ Could not import any RPS AI!")
            sys.exit(1)
    
    print(f"\n✅ Testing: {ai_name}\n")
    
    # Define test suite
    test_opponents = [
        (PatternBot(), 100, True),      # Should easily learn this
        (FrequencyBot(), 100, True),    # Should exploit frequency
        (CounterBot(), 100, True),      # Should adapt to counter
        (MixedBot(), 150, True),        # Should learn mixed patterns
        (RandomBot(), 100, True),       # Should maintain Nash
    ]
    
    results = []
    
    for opponent, rounds, verbose in test_opponents:
        # Create fresh AI for each test
        ai = ai_class(save_prefix=f"test_{opponent.name.replace(' ', '_')}")
        
        # Run test
        passed = run_test(ai, opponent, rounds, verbose=verbose)
        results.append((opponent.name, passed))
        
        print("\n")
    
    # Summary
    print("=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🏆 PERFECT SCORE! AI is learning effectively!")
    elif total_passed >= total_tests * 0.8:
        print("✅ GOOD! AI is learning well!")
    elif total_passed >= total_tests * 0.6:
        print("⚠️  ACCEPTABLE - AI needs improvement")
    else:
        print("❌ POOR - AI is not learning effectively")


if __name__ == "__main__":
    main()
