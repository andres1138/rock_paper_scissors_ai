#!/usr/bin/env python3
"""Debug Win-Stay predictions."""

from rps_predictor import RPSPredictor
import os

# Monkey-patch to add debug output
original_predict = RPSPredictor._predict_player_move

def debug_predict(self):
    result = original_predict(self)
    
    # Check Win-Stay stats
    total_after_wins = self.win_repeats + self.win_switches
    if total_after_wins >= 3:
        repeat_rate = self.win_repeats / total_after_wins
        if repeat_rate >= 0.65 and len(self.results) > 0:
            last_result = self.results[-1]
            print(f"\n[DEBUG] Win-Stay Detection:")
            print(f"  total_after_wins={total_after_wins}, repeat_rate={repeat_rate:.0%}")
            print(f"  last_result={last_result}")
            print(f"  Should add Win-Stay prediction: {last_result == 'l'}")
            print(f"  Final prediction: {result['move']} (conf={result['confidence']:.0%}, strategy={result['strategy']})")
    
    return result

RPSPredictor._predict_player_move = debug_predict

# Fresh start
if os.path.exists("test_brain.pkl"):
    os.remove("test_brain.pkl")

ai = RPSPredictor(save_file="test_brain.pkl")

# Perfect Win-Stay sequence
moves = [
    ('rock', 'scissors', 'l'),  # Opp wins
    ('rock', 'paper', 'w'),      # Opp repeats → AI wins
    ('paper', 'rock', 'l'),      # Opp wins  
    ('paper', 'scissors', 'w'),  # Opp repeats → AI wins
    ('scissors', 'paper', 'l'),  # Opp wins
    ('scissors', 'rock', 'w'),   # Opp repeats → AI wins
    ('rock', 'scissors', 'l'),   # Opp wins
    ('rock', 'paper', 'w'),      # Opp repeats
]

print("Running perfect Win-Stay sequence...")
for i, (opp, ai_move, expected_result) in enumerate(moves, 1):
    if i > 3:  # After enough data
        print(f"\n=== Round {i} ===")
        print(f"Getting move for next round...")
        pred_move = ai.get_move()
        print(f"AI chose: {pred_move}")
    
    result = ai.record_round(opp, ai_move)
    print(f"Round {i}: Opp={opp}, AI={ai_move}, Result={result}")

print(f"\n{'='*70}")
print(f"Final: win_repeats={ai.win_repeats}, win_switches={ai.win_switches}")

# Cleanup
if os.path.exists("test_brain.pkl"):
    os.remove("test_brain.pkl")
