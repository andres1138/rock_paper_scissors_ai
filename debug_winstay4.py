#!/usr/bin/env python3
"""
Debug Win-Stay prediction logic
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
            # We won! Stay with same move
            print(f"    [Opponent] I WON with {player_last}, staying with it")
            return player_last
        else:
            # We lost or drew, switch
            others = [m for m in self.MOVES if m != player_last]
            self.last_move = random.choice(others)
            print(f"    [Opponent] I LOST/DREW with {player_last}, switching to {self.last_move}")
            return self.last_move

# Test
ai = RPSPredictor()
strategy = WinStayStrategy()
history = []

print("Detailed Win-Stay Prediction Analysis\n" + "="*80 + "\n")

for round_num in range(1, 31):
    print(f"Round {round_num}:")
    
    # Show what AI will predict
    if round_num > 12:
        prediction = ai._predict_player_move()
        print(f"  [AI] Prediction: {prediction['move']} (conf: {prediction['confidence']:.2f}, strategy: {prediction['strategy']})")
    
    ai_move = ai.get_move()
    print(f"  [AI] Playing: {ai_move}")
    
    opponent_move = strategy.get_move(history)
    print(f"  [Opponent] Played: {opponent_move}")
    
    result = ai.record_round(opponent_move, ai_move)
    result_text = {'w': 'AI WIN', 'l': 'AI LOSS', 'd': 'DRAW'}[result]
    print(f"  Result: {result_text}")
    
    history.append({
        'round': round_num,
        'ai_move': ai_move,
        'opponent_move': opponent_move,
        'result': result
    })
    
    print()

print("\n" + "="*80)
print("Final Stats:")
print(f"Win-repeats: {ai.win_repeats}, Win-switches: {ai.win_switches}")
print(f"Loss-repeats: {ai.loss_repeats}, Loss-switches: {ai.loss_switches}")
if ai.win_repeats + ai.win_switches > 0:
    print(f"Win-Stay detection: {ai.win_repeats / (ai.win_repeats + ai.win_switches):.1%}")
