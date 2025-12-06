#!/usr/bin/env python3
"""
Test suite for RPS AI Predictor

Tests the AI against different opponent strategies to measure performance.
"""

import random
import argparse
from collections import Counter
from rps_predictor import RPSPredictor


class OpponentStrategy:
    """Base class for opponent strategies."""
    
    MOVES = ['rock', 'paper', 'scissors']
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    
    def get_move(self, history):
        """Get next move based on strategy."""
        raise NotImplementedError
    
    def name(self):
        """Strategy name."""
        raise NotImplementedError


class RandomStrategy(OpponentStrategy):
    """Completely random moves."""
    
    def get_move(self, history):
        return random.choice(self.MOVES)
    
    def name(self):
        return "Random"


class PatternStrategy(OpponentStrategy):
    """Plays a repeating pattern."""
    
    def __init__(self):
        self.pattern = ['rock', 'paper', 'scissors', 'scissors', 'paper']
        self.index = 0
    
    def get_move(self, history):
        move = self.pattern[self.index % len(self.pattern)]
        self.index += 1
        return move
    
    def name(self):
        return "Pattern (R-P-S-S-P)"


class FrequencyStrategy(OpponentStrategy):
    """Always plays rock (frequency bias)."""
    
    def __init__(self):
        self.moves = ['rock'] * 5 + ['paper'] * 2 + ['scissors'] * 1
        self.index = 0
    
    def get_move(self, history):
        move = self.moves[self.index % len(self.moves)]
        self.index += 1
        return move
    
    def name(self):
        return "Frequency (60% rock)"


class WinStayStrategy(OpponentStrategy):
    """Repeats move after winning, changes after losing."""
    
    def __init__(self):
        self.last_move = random.choice(self.MOVES)
        self.last_result = None
    
    def get_move(self, history):
        if not history:
            return self.last_move
        
        # Check if we won last round
        ai_last = history[-1]['ai_move']
        player_last = history[-1]['opponent_move']
        
        if self.BEATS[player_last] == ai_last:
            # We won! Stay with same move
            return player_last
        else:
            # We lost or drew, switch
            others = [m for m in self.MOVES if m != player_last]
            self.last_move = random.choice(others)
            return self.last_move
    
    def name(self):
        return "Win-Stay/Lose-Shift"


class CounterStrategy(OpponentStrategy):
    """Tries to counter AI's most common move."""
    
    BEATEN_BY = {'scissors': 'rock', 'rock': 'paper', 'paper': 'scissors'}
    
    def get_move(self, history):
        if len(history) < 3:
            return random.choice(self.MOVES)
        
        # Count AI moves
        recent_ai = [h['ai_move'] for h in history[-10:]]
        counter = Counter(recent_ai)
        most_common_ai = counter.most_common(1)[0][0]
        
        # Counter it
        return self.BEATEN_BY[most_common_ai]
    
    def name(self):
        return "Counter-AI"


def run_test(strategy: OpponentStrategy, rounds: int = 100, verbose: bool = False):
    """Run a test against a specific strategy."""
    ai = RPSPredictor()
    history = []
    
    print(f"\n{'='*70}")
    print(f"Testing against: {strategy.name()}")
    print(f"Rounds: {rounds}")
    print(f"{'='*70}\n")
    
    wins = 0
    losses = 0
    draws = 0
    
    # Track AI performance over time
    win_rate_checkpoints = []
    
    for round_num in range(1, rounds + 1):
        # Get moves
        ai_move = ai.get_move()
        opponent_move = strategy.get_move(history)
        
        # Record result
        result = ai.record_round(opponent_move, ai_move)
        
        history.append({
            'round': round_num,
            'ai_move': ai_move,
            'opponent_move': opponent_move,
            'result': result
        })
        
        if result == 'w':
            wins += 1
        elif result == 'l':
            losses += 1
        else:
            draws += 1
        
        # Checkpoints
        if round_num in [10, 25, 50, 75, 100]:
            wr = wins / round_num * 100
            win_rate_checkpoints.append((round_num, wr))
        
        # Verbose output
        if verbose and round_num <= 20:
            result_text = {'w': 'AI WIN', 'l': 'LOSS', 'd': 'DRAW'}[result]
            conf = ai.prediction_confidence
            print(f"Round {round_num:3d}: AI={ai_move:8s} Opp={opponent_move:8s} → {result_text:7s} (conf: {conf:.0%})")
    
    # Final stats
    stats = ai.get_stats()
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"AI Record:       {wins}-{losses}-{draws}")
    print(f"AI Win Rate:     {stats['win_rate']:.1f}%")
    print(f"Detected Style:  {stats['detected_behavior']}")
    
    print(f"\nWin Rate Over Time:")
    for checkpoint, wr in win_rate_checkpoints:
        bar = '█' * int(wr / 2)
        print(f"  Round {checkpoint:3d}: {wr:5.1f}% {bar}")
    
    insights = ai.get_insights()
    if insights:
        print(f"\nInsights:")
        for insight in insights:
            if insight:
                print(f"  • {insight}")
    
    print(f"{'='*70}\n")
    
    return stats


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Test RPS AI against different strategies')
    parser.add_argument('--strategy', type=str, choices=['random', 'pattern', 'frequency', 'win-stay', 'counter', 'all'],
                        default='all', help='Strategy to test against')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds to play')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    strategies = {
        'random': RandomStrategy(),
        'pattern': PatternStrategy(),
        'frequency': FrequencyStrategy(),
        'win-stay': WinStayStrategy(),
        'counter': CounterStrategy()
    }
    
    print("🎯" * 35)
    print("      RPS AI PREDICTOR - TEST SUITE")
    print("🎯" * 35)
    
    if args.strategy == 'all':
        results = {}
        for name, strat in strategies.items():
            stats = run_test(strat, args.rounds, args.verbose)
            results[name] = stats['win_rate']
        
        print("\n" + "="*70)
        print("SUMMARY - AI Win Rates")
        print("="*70)
        for name, wr in results.items():
            bar = '█' * int(wr / 2)
            print(f"  {name:12s}: {wr:5.1f}% {bar}")
        print("="*70)
    else:
        run_test(strategies[args.strategy], args.rounds, args.verbose)


if __name__ == "__main__":
    main()
