#!/usr/bin/env python3
"""Enhanced debug for Win-Stay predictions."""

from rps_predictor import RPSPredictor
import os

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
    result = ai.record_round(opp, ai_move)
    print(f"Round {i}: Opp={opp:8s} AI={ai_move:8s} Result={result}")
    
    if i >= 4:  # After enough data
        # Manually check Win-Stay detection
        total_after_wins = ai.win_repeats + ai.win_switches
        if total_after_wins >= 3:
            repeat_rate = ai.win_repeats / total_after_wins
            print(f"  → Win-Stay stats: repeats={ai.win_repeats}, switches={ai.win_switches}, rate={repeat_rate:.0%}")
            
            if repeat_rate >= 0.60:
                print(f"  → Win-Stay DETECTED! (threshold met)")
                if len(ai.results) > 0:
                    last_result = ai.results[-1]
                    print(f"  → Last result: {last_result} (opponent {'WON' if last_result == 'l' else 'LOST' if last_result == 'w' else 'DREW'})")
                    if last_result == 'l':
                        print(f"  → Should predict: opponent will REPEAT {opp}")
            else:
                print(f"  → Win-Stay threshold NOT met yet ({repeat_rate:.0%} < 60%)")

print(f"\n{'='*70}")
print(f"Final: win_repeats={ai.win_repeats}, win_switches={ai.win_switches}")

# Now test a prediction
print(f"\n{'='*70}")
print("Testing prediction after round 8...")
prediction = ai._predict_player_move()
print(f"Prediction: {prediction}")

# Cleanup
if os.path.exists("test_brain.pkl"):
    os.remove("test_brain.pkl")
