#!/usr/bin/env python3
"""
Deep dive into why Win-Stay is failing
"""

from rps_predictor import RPSPredictor
import random

class WinStayStrategy:
    """Repeats move after winning, changes after losing."""
    
    MOVES = ['rock', 'paper', 'scissors']
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    
    def __init__(self):
        self.last_move = random.choice(self.MOVES)
    
    def get_move(self, history):
        if not history:
            return self.last_move
        
        # Check if we won last round
        ai_last = history[-1]['ai_move']
        player_last = history[-1]['opponent_move']
        
        if self.BEATS[player_last] == ai_last:
            # We WON! Stay
            return player_last
        else:
            # We LOST or DREW, switch
            others = [m for m in self.MOVES if m != player_last]
            self.last_move = random.choice(others)
            return self.last_move

# Test
ai = RPSPredictor()
strategy = WinStayStrategy()
history = []

print("Deep Debug: Win-Stay/Lose-Shift\n" + "="*100)
print()

correct_predictions = 0
total_predictions = 0

for round_num in range(1, 51):
    # Get prediction BEFORE opponent makes move
    if round_num > 12:
        pred = ai._predict_player_move()
        pred_move = pred['move']
        pred_conf = pred['confidence']
        pred_strat = pred['strategy']
    else:
        pred_move = None
        pred_conf = None
        pred_strat = None
    
    ai_move = ai.get_move()
    opponent_move = strategy.get_move(history)
    result = ai.record_round(opponent_move, ai_move)
    
    history.append({
        'round': round_num,
        'ai_move': ai_move,
        'opponent_move': opponent_move,
        'result': result
    })
    
    result_text = {'w': 'AI WIN', 'l': 'AI LOSS', 'd': 'DRAW'}[result]
    
    if pred_move:
        pred_correct = '✓' if pred_move == opponent_move else '✗'
        if pred_move == opponent_move:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"R{round_num:2d} | Pred:{pred_move:8s}({pred_strat:20s}) conf:{pred_conf:.2f} | " + 
              f"Opp:{opponent_move:8s} {pred_correct} | AI:{ai_move:8s} | {result_text:7s}")
    else:
        print(f"R{round_num:2d} | (learning phase)".ljust(70) + f" | Opp:{opponent_move:8s} | AI:{ai_move:8s} | {result_text:7s}")
    
    # Status check
    if round_num in [20, 30, 40, 50]:
        print()
        print(f"  Status at Round {round_num}:")
        print(f"    Win-Stay detected: {ai.win_repeats}/{ai.win_repeats + ai.win_switches} = " +
              f"{ai.win_repeats/(ai.win_repeats+ai.win_switches)*100 if ai.win_repeats+ai.win_switches > 0 else 0:.0f}%")
        print(f"    Lose-Shift detected: {ai.loss_switches}/{ai.loss_repeats + ai.loss_switches} = " +
              f"{ai.loss_switches/(ai.loss_repeats+ai.loss_switches)*100 if ai.loss_repeats+ai.loss_switches > 0 else 0:.0f}%")
        if total_predictions > 0:
            print(f"    Prediction accuracy: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions*100:.1f}%")
        stats = ai.get_stats()
        print(f"    AI Win Rate: {stats['win_rate']:.1f}%")
        print()
