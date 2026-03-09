#!/usr/bin/env python3
"""
Debug Win-Stay behavior detection
"""

from rps_predictor import RPSPredictor
import random

class WinStayStrategy:
    """Repeats move after winning, changes after losing."""
    
    MOVES = ['rock', 'paper', 'scissors']
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    
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

# Test
ai = RPSPredictor()
strategy = WinStayStrategy()
history = []

print("Testing Win-Stay Detection\n" + "="*80)

for round_num in range(1, 51):
    ai_move = ai.get_move()
    opponent_move = strategy.get_move(history)
    result = ai.record_round(opponent_move, ai_move)
    
    history.append({
        'round': round_num,
        'ai_move': ai_move,
        'opponent_move': opponent_move,
        'result': result
    })
    
    # Print diagnostic info every few rounds
    if round_num in [10, 20, 30, 40, 50]:
        print(f"\nRound {round_num}:")
        print(f"  Win-repeats: {ai.win_repeats}, Win-switches: {ai.win_switches}")
        print(f"  Loss-repeats: {ai.loss_repeats}, Loss-switches: {ai.loss_switches}")
        
        if ai.win_repeats + ai.win_switches > 0:
            win_stay_rate = ai.win_repeats / (ai.win_repeats + ai.win_switches)
            print(f"  Win-Stay rate: {win_stay_rate:.1%}")
        
        if ai.loss_repeats + ai.loss_switches > 0:
            lose_shift_rate = ai.loss_switches / (ai.loss_repeats + ai.loss_switches)
            print(f"  Lose-Shift rate: {lose_shift_rate:.1%}")
        
        stats = ai.get_stats()
        print(f"  AI Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Prediction Accuracy: {stats['prediction_accuracy']:.1f}%")
        print(f"  Last Confidence: {stats['last_confidence']:.2f}")
