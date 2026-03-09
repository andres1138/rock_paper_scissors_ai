#!/usr/bin/env python3
"""
Test suite for Beast-Tier RPS AI v4.0

Tests against standard and advanced opponent strategies including
adaptive AI opponents that simulate tough competition.
"""

import random
import argparse
from collections import Counter
from rps_predictor import RPSPredictor, MOVES, BEATS, BEATEN_BY


# ─── Opponent Strategies ─────────────────────────────────────────────────

class OpponentStrategy:
    """Base class for opponent strategies."""
    
    def get_move(self, history):
        raise NotImplementedError
    
    def name(self):
        raise NotImplementedError


class RandomStrategy(OpponentStrategy):
    """Completely random — the baseline."""
    
    def get_move(self, history):
        return random.choice(MOVES)
    
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
    """Heavy frequency bias toward rock."""
    
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
    """Win-Stay / Lose-Shift — classic behavioral strategy."""
    
    def __init__(self):
        self.last_move = random.choice(MOVES)
    
    def get_move(self, history):
        if not history:
            return self.last_move
        
        ai_last = history[-1]['ai_move']
        opp_last = history[-1]['opponent_move']
        
        if BEATS[opp_last] == ai_last:
            # Opponent won — stay
            self.last_move = opp_last
        else:
            # Opponent lost or drew — shift
            others = [m for m in MOVES if m != opp_last]
            self.last_move = random.choice(others)
        
        return self.last_move
    
    def name(self):
        return "Win-Stay / Lose-Shift"


class CounterStrategy(OpponentStrategy):
    """Counters AI's most common recent move."""
    
    def get_move(self, history):
        if len(history) < 3:
            return random.choice(MOVES)
        
        recent_ai = [h['ai_move'] for h in history[-10:]]
        most_common = Counter(recent_ai).most_common(1)[0][0]
        return BEATEN_BY[most_common]
    
    def name(self):
        return "Counter-AI"


class MetaCounterStrategy(OpponentStrategy):
    """Uses its own Markov model to predict AI's moves — tough opponent."""
    
    def __init__(self):
        self.transitions = {}
    
    def get_move(self, history):
        if len(history) < 5:
            return random.choice(MOVES)
        
        # Build Markov model of AI's moves
        self.transitions.clear()
        ai_moves = [h['ai_move'] for h in history]
        for i in range(len(ai_moves) - 1):
            key = ai_moves[i]
            if key not in self.transitions:
                self.transitions[key] = Counter()
            self.transitions[key][ai_moves[i + 1]] += 1
        
        # Predict AI's next move
        last_ai = ai_moves[-1]
        if last_ai in self.transitions and self.transitions[last_ai]:
            predicted_ai = self.transitions[last_ai].most_common(1)[0][0]
            return BEATEN_BY[predicted_ai]
        
        return random.choice(MOVES)
    
    def name(self):
        return "Meta-Counter (Markov AI predictor)"


class PatternSwitcherStrategy(OpponentStrategy):
    """Switches between strategies every 15-25 rounds."""
    
    def __init__(self):
        self.sub_strategies = [
            self._play_rock_heavy,
            self._play_cycle,
            self._play_counter_last,
            self._play_scissors_heavy,
            self._play_anti_repeat,
        ]
        self.current_idx = 0
        self.rounds_in_current = 0
        self.switch_point = random.randint(15, 25)
    
    def _play_rock_heavy(self, history):
        return random.choices(MOVES, weights=[5, 2, 1])[0]
    
    def _play_cycle(self, history):
        return MOVES[len(history) % 3]
    
    def _play_counter_last(self, history):
        if history:
            return BEATEN_BY[history[-1]['ai_move']]
        return random.choice(MOVES)
    
    def _play_scissors_heavy(self, history):
        return random.choices(MOVES, weights=[1, 2, 5])[0]
    
    def _play_anti_repeat(self, history):
        if history:
            last_opp = history[-1]['opponent_move']
            others = [m for m in MOVES if m != last_opp]
            return random.choice(others)
        return random.choice(MOVES)
    
    def get_move(self, history):
        self.rounds_in_current += 1
        if self.rounds_in_current >= self.switch_point:
            self.current_idx = (self.current_idx + 1) % len(self.sub_strategies)
            self.rounds_in_current = 0
            self.switch_point = random.randint(15, 25)
        
        return self.sub_strategies[self.current_idx](history)
    
    def name(self):
        return "Pattern-Switcher (changes every ~20 rounds)"


class RotatingBiasStrategy(OpponentStrategy):
    """Cycles which move is favored every 10 rounds."""
    
    def __init__(self):
        self.round = 0
    
    def get_move(self, history):
        self.round += 1
        phase = (self.round // 10) % 3
        weights = [[6, 2, 1], [1, 6, 2], [2, 1, 6]][phase]
        return random.choices(MOVES, weights=weights)[0]
    
    def name(self):
        return "Rotating Bias (cycles favorites)"


class AdaptiveAIStrategy(OpponentStrategy):
    """Simulates a strong adaptive AI opponent — the Stake simulator."""
    
    def __init__(self):
        self.opp_history = []  # Tracks player (our AI's) moves
        self.transitions_1 = {}
        self.transitions_2 = {}
        self.context_model = {}
    
    def get_move(self, history):
        if len(history) < 6:
            return random.choice(MOVES)
        
        ai_moves = [h['ai_move'] for h in history]
        self.opp_history = ai_moves
        
        # Multi-model prediction of what AI will play
        predictions = Counter()
        
        # Model 1: Markov-1 on AI moves
        for i in range(len(ai_moves) - 1):
            key = ai_moves[i]
            if key not in self.transitions_1:
                self.transitions_1[key] = Counter()
            self.transitions_1[key][ai_moves[i + 1]] += 1
        
        last = ai_moves[-1]
        if last in self.transitions_1 and self.transitions_1[last]:
            pred = self.transitions_1[last].most_common(1)[0][0]
            predictions[pred] += 3
        
        # Model 2: Markov-2 on AI moves
        if len(ai_moves) >= 3:
            for i in range(len(ai_moves) - 2):
                key = (ai_moves[i], ai_moves[i + 1])
                if key not in self.transitions_2:
                    self.transitions_2[key] = Counter()
                self.transitions_2[key][ai_moves[i + 2]] += 1
            
            key2 = (ai_moves[-2], ai_moves[-1])
            if key2 in self.transitions_2 and self.transitions_2[key2]:
                pred = self.transitions_2[key2].most_common(1)[0][0]
                predictions[pred] += 5
        
        # Model 3: Context (move + result)
        results = [h['result'] for h in history]
        if len(results) >= 2:
            for i in range(len(ai_moves) - 1):
                if i < len(results):
                    ckey = (ai_moves[i], results[i])
                    if ckey not in self.context_model:
                        self.context_model[ckey] = Counter()
                    self.context_model[ckey][ai_moves[i + 1]] += 1
            
            if len(results) >= 1:
                ckey = (ai_moves[-1], results[-1])
                if ckey in self.context_model and self.context_model[ckey]:
                    pred = self.context_model[ckey].most_common(1)[0][0]
                    predictions[pred] += 4
        
        if predictions:
            # Counter the most likely AI move
            predicted_ai = predictions.most_common(1)[0][0]
            # 80% exploit, 20% random for safety
            if random.random() < 0.80:
                return BEATEN_BY[predicted_ai]
        
        return random.choice(MOVES)
    
    def name(self):
        return "Adaptive AI (Stake simulator)"


# ─── Test Runner ─────────────────────────────────────────────────────────

def run_test(strategy: OpponentStrategy, rounds: int = 200, verbose: bool = False):
    """Run a test against a specific strategy."""
    ai = RPSPredictor(reset_brain=True)
    history = []
    
    print(f"\n{'=' * 70}")
    print(f"Testing against: {strategy.name()}")
    print(f"Rounds: {rounds}")
    print(f"{'=' * 70}\n")
    
    wins = losses = draws = 0
    win_rate_checkpoints = []
    
    for round_num in range(1, rounds + 1):
        ai_move = ai.get_move()
        opponent_move = strategy.get_move(history)
        
        result = ai.record_round(opponent_move, ai_move)
        
        history.append({
            'round': round_num,
            'ai_move': ai_move,
            'opponent_move': opponent_move,
            'result': result,
        })
        
        if result == 'w':
            wins += 1
        elif result == 'l':
            losses += 1
        else:
            draws += 1
        
        # Checkpoints
        if round_num in [10, 25, 50, 75, 100, 150, 200]:
            wr = wins / round_num * 100
            win_rate_checkpoints.append((round_num, wr))
        
        if verbose and round_num <= 25:
            conf = ai.prediction_confidence
            strat = ai.active_strategy_name[:30]
            result_label = {'w': 'WIN ', 'l': 'LOSS', 'd': 'DRAW'}[result]
            print(f"  R{round_num:3d}: AI={ai_move:8s} Opp={opponent_move:8s} → {result_label} "
                  f"(conf:{conf:.0%}) [{strat}]")
    
    # Results
    stats = ai.get_stats()
    
    print(f"\n{'─' * 70}")
    print("RESULTS")
    print(f"{'─' * 70}")
    print(f"  Record:        {wins}-{losses}-{draws}")
    print(f"  Win Rate:      {stats['win_rate']:.1f}%")
    print(f"  Detected:      {stats['detected_behavior']}")
    
    print(f"\n  Win Rate Over Time:")
    for checkpoint, wr in win_rate_checkpoints:
        bar = '█' * int(wr / 2)
        color = '🟢' if wr > 40 else '🟡' if wr > 33 else '🔴'
        print(f"    Round {checkpoint:3d}: {wr:5.1f}% {color} {bar}")
    
    # Strategy leaderboard
    leaderboard = ai.get_strategy_leaderboard(top_n=3)
    if leaderboard:
        print(f"\n  Top Strategies:")
        for entry in leaderboard:
            print(f"    {entry['name']:28s} score: {entry['score']:+.1f}")
    
    insights = ai.get_insights()
    if insights:
        print(f"\n  Insights:")
        for insight in insights:
            print(f"    • {insight}")
    
    print(f"{'=' * 70}\n")
    return stats


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Test Beast RPS AI v4.0')
    parser.add_argument('--strategy', type=str,
                        choices=['random', 'pattern', 'frequency', 'win-stay', 'counter',
                                 'meta-counter', 'switcher', 'rotating', 'adaptive', 'all'],
                        default='all', help='Strategy to test against')
    parser.add_argument('--rounds', type=int, default=200, help='Number of rounds')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    strategies = {
        'random': RandomStrategy(),
        'pattern': PatternStrategy(),
        'frequency': FrequencyStrategy(),
        'win-stay': WinStayStrategy(),
        'counter': CounterStrategy(),
        'meta-counter': MetaCounterStrategy(),
        'switcher': PatternSwitcherStrategy(),
        'rotating': RotatingBiasStrategy(),
        'adaptive': AdaptiveAIStrategy(),
    }
    
    print("🔥" * 35)
    print("      BEAST RPS AI v4.0 — TEST SUITE")
    print("🔥" * 35)
    
    if args.strategy == 'all':
        results = {}
        for name, strat in strategies.items():
            stats = run_test(strat, args.rounds, args.verbose)
            results[name] = stats['win_rate']
        
        print("\n" + "=" * 70)
        print("SUMMARY — AI Win Rates")
        print("=" * 70)
        for name, wr in results.items():
            bar = '█' * int(wr / 2)
            color = '🟢' if wr > 40 else '🟡' if wr > 33 else '🔴'
            print(f"  {name:15s}: {wr:5.1f}% {color} {bar}")
        
        avg = sum(results.values()) / len(results)
        print(f"\n  {'AVERAGE':15s}: {avg:5.1f}%")
        print("=" * 70)
    else:
        run_test(strategies[args.strategy], args.rounds, args.verbose)


if __name__ == "__main__":
    main()
