#!/usr/bin/env python3
"""
Beast-Tier RPS Predictor v4.0 — Multi-Layer Meta-Strategy Engine

Inspired by competition-winning bots (Iocaine Powder, Dan Egnor's algorithm).
Uses 6 base predictors × 3 meta-levels = 18 competing strategies.
Automatically selects the best counter-level for any opponent.

Architecture:
  - Base Predictors: frequency, markov-1, markov-2, n-gram(3-6), 
    win-stay/lose-shift, history-match, rotation-detect, context-predictor
  - Meta-Levels: P0 (direct predict), P1 (counter their counter), P2 (double counter)
  - Strategy scoring with exponential decay for rapid adaptation
  - Nash equilibrium fallback as anti-exploitation shield
"""

import random
import math
import pickle
import os
from collections import deque, Counter, defaultdict
from typing import Optional, Dict, Tuple, List


# ─── Constants ───────────────────────────────────────────────────────────

MOVES = ['rock', 'paper', 'scissors']
BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
BEATEN_BY = {'scissors': 'rock', 'rock': 'paper', 'paper': 'scissors'}
MOVE_INDEX = {'rock': 0, 'paper': 1, 'scissors': 2}
INDEX_MOVE = {0: 'rock', 1: 'paper', 2: 'scissors'}


def rotate_move(move: str, steps: int) -> str:
    """Rotate a move by steps in the cycle rock->paper->scissors."""
    idx = MOVE_INDEX[move]
    return INDEX_MOVE[(idx + steps) % 3]


def counter(move: str) -> str:
    """Return the move that beats the given move."""
    return BEATEN_BY[move]


def counter_n(move: str, levels: int) -> str:
    """Apply counter `levels` times. 0=same, 1=beat it, 2=beat the counter, etc."""
    m = move
    for _ in range(levels):
        m = BEATEN_BY[m]
    return m


# ─── Base Predictors ─────────────────────────────────────────────────────

class BasePredictor:
    """Base class for all predictors. Predicts opponent's next move."""
    
    def __init__(self, name: str):
        self.name = name
    
    def predict(self, opp_history: list, my_history: list, results: list) -> Optional[str]:
        """Return predicted opponent move, or None if no prediction."""
        raise NotImplementedError
    
    def reset(self):
        """Reset any internal state."""
        pass


class FrequencyPredictor(BasePredictor):
    """Predict based on overall move frequency."""
    
    def __init__(self):
        super().__init__("frequency")
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 5:
            return None
        counts = Counter(opp_history[-30:])
        total = sum(counts.values())
        most_common, count = counts.most_common(1)[0]
        if count / total >= 0.38:
            return most_common
        return None


class RecentFrequencyPredictor(BasePredictor):
    """Predict based on very recent move frequency (last 8 moves)."""
    
    def __init__(self):
        super().__init__("recent_freq")
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 6:
            return None
        counts = Counter(opp_history[-8:])
        most_common, count = counts.most_common(1)[0]
        if count >= 4:
            return most_common
        return None


class Markov1Predictor(BasePredictor):
    """First-order Markov: predict based on last opponent move."""
    
    def __init__(self):
        super().__init__("markov1")
        self.transitions = defaultdict(Counter)
    
    def predict(self, opp_history, my_history, results):
        # Rebuild transitions from history (lightweight)
        if len(opp_history) < 4:
            return None
        
        self.transitions.clear()
        for i in range(len(opp_history) - 1):
            self.transitions[opp_history[i]][opp_history[i + 1]] += 1
        
        last = opp_history[-1]
        if last in self.transitions and sum(self.transitions[last].values()) >= 3:
            most_common, count = self.transitions[last].most_common(1)[0]
            total = sum(self.transitions[last].values())
            if count / total >= 0.38:
                return most_common
        return None
    
    def reset(self):
        self.transitions.clear()


class Markov2Predictor(BasePredictor):
    """Second-order Markov: predict based on last 2 opponent moves."""
    
    def __init__(self):
        super().__init__("markov2")
        self.transitions = defaultdict(Counter)
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 5:
            return None
        
        self.transitions.clear()
        for i in range(len(opp_history) - 2):
            key = (opp_history[i], opp_history[i + 1])
            self.transitions[key][opp_history[i + 2]] += 1
        
        key = (opp_history[-2], opp_history[-1])
        if key in self.transitions and sum(self.transitions[key].values()) >= 2:
            most_common, count = self.transitions[key].most_common(1)[0]
            total = sum(self.transitions[key].values())
            if count / total >= 0.35:
                return most_common
        return None
    
    def reset(self):
        self.transitions.clear()


class NgramPredictor(BasePredictor):
    """N-gram predictor for sequences of length 3-6."""
    
    def __init__(self, n: int):
        super().__init__(f"ngram{n}")
        self.n = n
        self.patterns = defaultdict(Counter)
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < self.n + 2:
            return None
        
        self.patterns.clear()
        for i in range(len(opp_history) - self.n):
            key = tuple(opp_history[i:i + self.n])
            self.patterns[key][opp_history[i + self.n]] += 1
        
        key = tuple(opp_history[-self.n:])
        if key in self.patterns and sum(self.patterns[key].values()) >= 2:
            most_common, count = self.patterns[key].most_common(1)[0]
            total = sum(self.patterns[key].values())
            if count / total >= 0.35:
                return most_common
        return None
    
    def reset(self):
        self.patterns.clear()


class ContextPredictor(BasePredictor):
    """Predict based on (opponent_move, my_move, result) context triples."""
    
    def __init__(self):
        super().__init__("context")
        self.patterns = defaultdict(Counter)
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 5 or len(my_history) < 2 or len(results) < 2:
            return None
        
        self.patterns.clear()
        min_len = min(len(opp_history), len(my_history), len(results))
        for i in range(min_len - 1):
            key = (opp_history[i], my_history[i], results[i])
            self.patterns[key][opp_history[i + 1]] += 1
        
        key = (opp_history[-1], my_history[-1], results[-1])
        if key in self.patterns and sum(self.patterns[key].values()) >= 2:
            most_common, count = self.patterns[key].most_common(1)[0]
            total = sum(self.patterns[key].values())
            if count / total >= 0.35:
                return most_common
        return None
    
    def reset(self):
        self.patterns.clear()


class DoubleContextPredictor(BasePredictor):
    """Predict based on (opp[-2], opp[-1], my[-1], result[-1]) — deeper context."""
    
    def __init__(self):
        super().__init__("double_context")
        self.patterns = defaultdict(Counter)
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 8 or len(my_history) < 3 or len(results) < 3:
            return None
        
        self.patterns.clear()
        min_len = min(len(opp_history), len(my_history), len(results))
        for i in range(1, min_len - 1):
            key = (opp_history[i - 1], opp_history[i], my_history[i], results[i])
            self.patterns[key][opp_history[i + 1]] += 1
        
        key = (opp_history[-2], opp_history[-1], my_history[-1], results[-1])
        if key in self.patterns and sum(self.patterns[key].values()) >= 2:
            most_common, count = self.patterns[key].most_common(1)[0]
            total = sum(self.patterns[key].values())
            if count / total >= 0.35:
                return most_common
        return None
    
    def reset(self):
        self.patterns.clear()


class WinStayLoseShiftPredictor(BasePredictor):
    """Detect and exploit Win-Stay/Lose-Shift behavior."""
    
    def __init__(self):
        super().__init__("winstay")
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 4 or len(results) < 2:
            return None
        
        # Count WSLS behavior
        min_len = min(len(opp_history), len(results))
        win_repeat = win_total = 0
        lose_switch = lose_total = 0
        
        for i in range(1, min_len):
            prev_result = results[i - 1]
            did_repeat = (opp_history[i] == opp_history[i - 1])
            
            if prev_result == 'l':  # AI lost = opponent won
                win_total += 1
                if did_repeat:
                    win_repeat += 1
            elif prev_result == 'w':  # AI won = opponent lost
                lose_total += 1
                if not did_repeat:
                    lose_switch += 1
        
        last_result = results[-1]
        last_opp = opp_history[-1]
        
        # WSLS after opponent won
        if last_result == 'l' and win_total >= 3:
            repeat_rate = win_repeat / win_total
            if repeat_rate >= 0.55:
                return last_opp  # They'll repeat
        
        # WSLS after opponent lost
        if last_result == 'w' and lose_total >= 3:
            switch_rate = lose_switch / lose_total
            if switch_rate >= 0.55:
                # They'll switch — predict which move they switch TO
                other_moves = [m for m in MOVES if m != last_opp]
                # Check if they have a preferred switch target
                switch_targets = Counter()
                for i in range(1, min_len):
                    if results[i - 1] == 'w' and opp_history[i] != opp_history[i - 1]:
                        switch_targets[opp_history[i]] += 1
                
                if switch_targets:
                    return switch_targets.most_common(1)[0][0]
                else:
                    return random.choice(other_moves)
        
        return None


class HistoryMatchPredictor(BasePredictor):
    """Find the longest matching suffix in history and predict next move."""
    
    def __init__(self):
        super().__init__("history_match")
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 8:
            return None
        
        best_match_len = 0
        best_next_move = None
        
        # Try matching suffixes of length 2 to 10
        for match_len in range(2, min(11, len(opp_history))):
            suffix = opp_history[-match_len:]
            
            # Search for this pattern earlier in history
            for i in range(len(opp_history) - match_len - 1):
                if opp_history[i:i + match_len] == suffix:
                    # Found a match! The next move after this pattern
                    next_idx = i + match_len
                    if next_idx < len(opp_history):
                        if match_len > best_match_len:
                            best_match_len = match_len
                            best_next_move = opp_history[next_idx]
        
        if best_match_len >= 3:
            return best_next_move
        return None


class RotationPredictor(BasePredictor):
    """Detect cycling/rotation patterns with periods 2-8."""
    
    def __init__(self):
        super().__init__("rotation")
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 8:
            return None
        
        best_period = None
        best_score = 0.0
        
        for period in range(2, min(9, len(opp_history) // 2)):
            matches = 0
            checks = 0
            for i in range(len(opp_history) - period):
                checks += 1
                if opp_history[i] == opp_history[i + period]:
                    matches += 1
            
            if checks >= period * 2:
                score = matches / checks
                if score > best_score and score >= 0.65:
                    best_score = score
                    best_period = period
        
        if best_period:
            # Predict using the period
            return opp_history[-best_period]
        return None


class AntiMirrorPredictor(BasePredictor):
    """Detect if opponent mirrors or anti-mirrors our moves."""
    
    def __init__(self):
        super().__init__("anti_mirror")
        
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 8 or len(my_history) < 8:
            return None
        
        # Check if opponent copies our previous move
        copy_count = 0
        counter_count = 0  
        total = 0
        
        min_len = min(len(opp_history), len(my_history))
        for i in range(1, min_len):
            total += 1
            if opp_history[i] == my_history[i - 1]:
                copy_count += 1
            if opp_history[i] == counter(my_history[i - 1]):
                counter_count += 1
        
        if total >= 6:
            if copy_count / total >= 0.50:
                # Opponent copies our last move
                return my_history[-1]
            if counter_count / total >= 0.50:
                # Opponent counters our last move
                return counter(my_history[-1])
        
        return None


class ResultContextPredictor(BasePredictor):
    """Predict based only on the last result (what people do after W/L/D)."""
    
    def __init__(self):
        super().__init__("result_context")
        self.after_win = Counter()
        self.after_loss = Counter()
        self.after_draw = Counter()
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 6 or len(results) < 2:
            return None
        
        self.after_win.clear()
        self.after_loss.clear()
        self.after_draw.clear()
        
        min_len = min(len(opp_history), len(results))
        for i in range(1, min_len):
            if results[i - 1] == 'l':  # AI lost = opp won
                self.after_win[opp_history[i]] += 1
            elif results[i - 1] == 'w':  # AI won = opp lost
                self.after_loss[opp_history[i]] += 1
            else:
                self.after_draw[opp_history[i]] += 1
        
        last_result = results[-1]
        
        if last_result == 'l' and sum(self.after_win.values()) >= 3:
            most_common, count = self.after_win.most_common(1)[0]
            if count / sum(self.after_win.values()) >= 0.45:
                return most_common
        elif last_result == 'w' and sum(self.after_loss.values()) >= 3:
            most_common, count = self.after_loss.most_common(1)[0]
            if count / sum(self.after_loss.values()) >= 0.45:
                return most_common
        elif last_result == 'd' and sum(self.after_draw.values()) >= 3:
            most_common, count = self.after_draw.most_common(1)[0]
            if count / sum(self.after_draw.values()) >= 0.45:
                return most_common
        
        return None


class PairHistoryPredictor(BasePredictor):
    """Predict based on (my_move, opp_move) pairs — joint context."""
    
    def __init__(self):
        super().__init__("pair_history")
        self.patterns = defaultdict(Counter)
    
    def predict(self, opp_history, my_history, results):
        if len(opp_history) < 6 or len(my_history) < 2:
            return None
        
        self.patterns.clear()
        min_len = min(len(opp_history), len(my_history))
        for i in range(min_len - 1):
            key = (my_history[i], opp_history[i])
            self.patterns[key][opp_history[i + 1]] += 1
        
        key = (my_history[-1], opp_history[-1])
        if key in self.patterns and sum(self.patterns[key].values()) >= 2:
            most_common, count = self.patterns[key].most_common(1)[0]
            total = sum(self.patterns[key].values())
            if count / total >= 0.40:
                return most_common
        return None
    
    def reset(self):
        self.patterns.clear()


# ─── Meta-Strategy Layer ─────────────────────────────────────────────────

class MetaStrategy:
    """
    A meta-strategy wraps a base predictor with a meta-level.
    
    Meta-levels:
      P0: Predict opponent's move directly, counter it
      P1: Assume opponent predicts our P0 and counters => counter their counter
      P2: Assume opponent counters our P1 => counter their counter-counter
    """
    
    def __init__(self, predictor: BasePredictor, meta_level: int):
        self.predictor = predictor
        self.meta_level = meta_level
        self.name = f"{predictor.name}.P{meta_level}"
        
        # Scoring with exponential decay
        self.score = 0.0
        self.decay = 0.90  # Forget old results fast
        self.predictions_made = 0
        self.recent_correct = 0
        self.recent_total = 0
    
    def get_suggestion(self, opp_history: list, my_history: list, results: list) -> Optional[str]:
        """Get the suggested move to play (what WE should throw)."""
        prediction = self.predictor.predict(opp_history, my_history, results)
        if prediction is None:
            return None
        
        # Apply meta-level rotation
        # P0: counter their predicted move (1 rotation)  
        # P1: they predict we counter, so they play to beat our counter → we beat THAT (2 rotations)
        # P2: 3 rotations
        return counter_n(prediction, self.meta_level + 1)
    
    def get_predicted_opp_move(self, opp_history: list, my_history: list, results: list) -> Optional[str]:
        """Get what we think opponent will play (for scoring)."""
        return self.predictor.predict(opp_history, my_history, results)
    
    def update_score(self, would_have_won: bool, would_have_drawn: bool):
        """Update the strategy's score based on what would have happened."""
        self.score *= self.decay
        
        if would_have_won:
            self.score += 1.0
            self.recent_correct += 1
        elif would_have_drawn:
            self.score += 0.1
        else:
            self.score -= 0.5
        
        self.recent_total += 1
    
    def reset_score(self):
        self.score = 0.0
        self.recent_correct = 0
        self.recent_total = 0


# ─── Main Engine ─────────────────────────────────────────────────────────

class RPSPredictor:
    """
    Beast-Tier RPS AI v4.0 — Multi-Layer Meta-Strategy Engine
    
    Runs multiple prediction strategies in parallel, each at multiple
    meta-levels. Automatically selects the best one based on recent
    performance. Falls back to Nash equilibrium when nothing works.
    """
    
    MOVES = MOVES
    BEATS = BEATS
    BEATEN_BY = BEATEN_BY
    
    def __init__(self, memory_size: int = 200, save_file: str = "rps_ai_brain.pkl", 
                 reset_brain: bool = False):
        self.memory_size = memory_size
        self.save_file = save_file
        
        # Move history (full lists for deep analysis)
        self.opp_history: List[str] = []
        self.my_history: List[str] = []
        self.results: List[str] = []  # 'w' = AI win, 'l' = AI loss, 'd' = draw
        
        # Statistics
        self.total_rounds = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_streak = 0
        
        # State for UI
        self.prediction_confidence = 0.0
        self.last_prediction = None
        self.last_prediction_strategy = None
        self.last_prediction_correct = None
        self.detected_behavior = "unknown"
        self.active_strategy_name = "gathering data"
        
        # Build the strategy roster
        self._build_strategies()
        
        # Track what each strategy WOULD have suggested (for scoring)
        self._last_suggestions: Dict[str, Optional[str]] = {}
        
        # Anti-exploitation tracking
        self._recent_my_moves = deque(maxlen=20)
        self._exploitation_counter = 0
        
        # Nash equilibrium fallback tracking
        self._nash_rounds = 0
        self._nash_cooldown = 0
        
        # Load brain
        if not reset_brain:
            self.load_brain()
    
    def _build_strategies(self):
        """Build the full roster of meta-strategies."""
        base_predictors = [
            FrequencyPredictor(),
            RecentFrequencyPredictor(),
            Markov1Predictor(),
            Markov2Predictor(),
            NgramPredictor(3),
            NgramPredictor(4),
            NgramPredictor(5),
            NgramPredictor(6),
            ContextPredictor(),
            DoubleContextPredictor(),
            WinStayLoseShiftPredictor(),
            HistoryMatchPredictor(),
            RotationPredictor(),
            AntiMirrorPredictor(),
            ResultContextPredictor(),
            PairHistoryPredictor(),
        ]
        
        self.strategies: List[MetaStrategy] = []
        for predictor in base_predictors:
            for meta_level in range(3):  # P0, P1, P2
                self.strategies.append(MetaStrategy(predictor, meta_level))
    
    def get_move(self) -> str:
        """Get AI's next move. This is the main entry point."""
        
        # Phase 1: Gathering data (first 5 rounds — play random)
        if self.total_rounds < 5:
            self.active_strategy_name = "gathering data"
            self.prediction_confidence = 0.0
            self.last_prediction = None
            return random.choice(MOVES)
        
        # Phase 2: Check for Nash fallback
        if self._should_play_nash():
            self.active_strategy_name = "Nash equilibrium (shield)"
            self.prediction_confidence = 0.0
            self.last_prediction = None
            self._nash_rounds += 1
            return random.choice(MOVES)
        
        # Phase 3: Get all strategy suggestions and pick the best
        suggestions = {}
        for strategy in self.strategies:
            suggestion = strategy.get_suggestion(
                self.opp_history, self.my_history, self.results
            )
            if suggestion is not None:
                suggestions[strategy.name] = suggestion
                self._last_suggestions[strategy.name] = suggestion
        
        if not suggestions:
            self.active_strategy_name = "no patterns found"
            self.prediction_confidence = 0.0
            return random.choice(MOVES)
        
        # Find the best-scoring strategy that has a suggestion
        best_strategy = None
        best_score = -float('inf')
        
        for strategy in self.strategies:
            if strategy.name in suggestions and strategy.score > best_score:
                best_score = strategy.score
                best_strategy = strategy
        
        if best_strategy is None or best_score < -2.0:
            # All strategies are performing badly — Nash fallback
            self.active_strategy_name = "Nash fallback (all negative)"
            self.prediction_confidence = 0.0
            return random.choice(MOVES)
        
        # Use the best strategy's suggestion
        chosen_move = suggestions[best_strategy.name]
        self.active_strategy_name = best_strategy.name
        
        # Calculate confidence based on strategy score
        self.prediction_confidence = min(0.95, max(0.0, best_score / 5.0))
        
        # Get the prediction for display
        opp_prediction = best_strategy.get_predicted_opp_move(
            self.opp_history, self.my_history, self.results
        )
        self.last_prediction = opp_prediction
        self.last_prediction_strategy = best_strategy.name
        
        # Anti-repetition: if we're playing the same move too often, inject randomness
        self._recent_my_moves.append(chosen_move)
        if len(self._recent_my_moves) >= 8:
            move_counts = Counter(self._recent_my_moves)
            most_common_count = move_counts.most_common(1)[0][1]
            if most_common_count / len(self._recent_my_moves) > 0.65:
                # We're being too predictable — 30% chance to randomize
                if random.random() < 0.30:
                    chosen_move = random.choice(MOVES)
                    self.active_strategy_name += " (anti-repeat jitter)"
        
        return chosen_move
    
    def record_round(self, opponent_move: str, ai_move: str) -> str:
        """Record the result of a round and update all strategies."""
        result = self._determine_result(ai_move, opponent_move)
        
        # Score ALL strategies based on what they WOULD have done
        for strategy in self.strategies:
            suggestion = strategy.get_suggestion(
                self.opp_history, self.my_history, self.results
            )
            if suggestion is not None:
                # Would this strategy have won?
                would_have_won = (BEATS[suggestion] == opponent_move)
                would_have_drawn = (suggestion == opponent_move)
                strategy.update_score(would_have_won, would_have_drawn)
        
        # Update history
        self.opp_history.append(opponent_move)
        self.my_history.append(ai_move)
        self.results.append(result)
        
        # Trim history if too long (keep last memory_size entries)
        if len(self.opp_history) > self.memory_size:
            excess = len(self.opp_history) - self.memory_size
            self.opp_history = self.opp_history[excess:]
            self.my_history = self.my_history[excess:]
            self.results = self.results[excess:]
        
        # Update stats
        self.total_rounds += 1
        if result == 'w':
            self.wins += 1
            self.current_streak = max(1, self.current_streak + 1)
        elif result == 'l':
            self.losses += 1
            self.current_streak = min(-1, self.current_streak - 1)
        else:
            self.draws += 1
            self.current_streak = 0
        
        # Check prediction correctness
        if self.last_prediction is not None:
            self.last_prediction_correct = (self.last_prediction == opponent_move)
        
        # Detect exploitation
        self._check_exploitation(ai_move, opponent_move, result)
        
        # Update behavior analysis periodically
        if self.total_rounds % 10 == 0:
            self._analyze_behavior()
        
        return result
    
    def _should_play_nash(self) -> bool:
        """Determine if we should fall back to Nash equilibrium."""
        if self.total_rounds < 15:
            return False
        
        # Cool down from forced Nash
        if self._nash_cooldown > 0:
            self._nash_cooldown -= 1
            return True
        
        # Check if any strategy is significantly beating random
        best_score = max(s.score for s in self.strategies)
        
        if best_score < -1.0 and self.total_rounds >= 20:
            # Nothing is working — go Nash for a few rounds
            self._nash_cooldown = 3
            return True
        
        # Check recent win rate
        if self.total_rounds >= 20:
            recent_results = self.results[-15:]
            recent_wins = recent_results.count('w')
            recent_losses = recent_results.count('l')
            
            if recent_losses > 0 and recent_wins / len(recent_results) < 0.20:
                # Getting crushed — Nash reset
                self._nash_cooldown = 5
                # Also reset strategy scores to give everyone a fresh start
                for s in self.strategies:
                    s.score *= 0.3
                return True
        
        return False
    
    def _check_exploitation(self, ai_move: str, opp_move: str, result: str):
        """Check if opponent is exploiting our patterns."""
        if result == 'l':
            self._exploitation_counter += 1
        elif result == 'w':
            self._exploitation_counter = max(0, self._exploitation_counter - 1)
        
        # If being heavily exploited, reset all strategy preferences
        if self._exploitation_counter >= 6:
            for s in self.strategies:
                s.score *= 0.2
            self._exploitation_counter = 0
    
    def _analyze_behavior(self):
        """Classify opponent's playing style."""
        if self.total_rounds < 10:
            self.detected_behavior = "learning"
            return
        
        entropy = self._calculate_entropy()
        
        # Check win rate
        win_rate = self.wins / self.total_rounds if self.total_rounds > 0 else 0
        
        # Check best strategy type
        best_strat = max(self.strategies, key=lambda s: s.score)
        
        if entropy > 0.95:
            self.detected_behavior = "random / unpredictable"
        elif 'winstay' in best_strat.name:
            self.detected_behavior = "win-stay / lose-shift"
        elif 'ngram' in best_strat.name or 'history_match' in best_strat.name:
            self.detected_behavior = "pattern-based"
        elif 'rotation' in best_strat.name:
            self.detected_behavior = "cycling / rotating"
        elif 'context' in best_strat.name or 'pair_history' in best_strat.name:
            self.detected_behavior = "context-adaptive"
        elif 'anti_mirror' in best_strat.name:
            self.detected_behavior = "mirroring / counter-playing"
        elif 'frequency' in best_strat.name:
            self.detected_behavior = "frequency-biased"
        elif 'markov' in best_strat.name:
            self.detected_behavior = "sequential (Markov)"
        elif win_rate < 0.30:
            self.detected_behavior = "advanced adaptive AI"
        else:
            self.detected_behavior = "mixed"
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of recent opponent moves."""
        if len(self.opp_history) < 10:
            return 1.0
        
        recent = self.opp_history[-30:]
        total = len(recent)
        entropy = 0.0
        
        for move in MOVES:
            count = recent.count(move)
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy / math.log2(3)
    
    def _determine_result(self, ai_move: str, player_move: str) -> str:
        """Determine result from AI's perspective."""
        if ai_move == player_move:
            return 'd'
        elif BEATS[ai_move] == player_move:
            return 'w'
        else:
            return 'l'
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        win_rate = (self.wins / self.total_rounds * 100) if self.total_rounds > 0 else 0
        
        return {
            'total_rounds': self.total_rounds,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': win_rate,
            'current_streak': self.current_streak,
            'detected_behavior': self.detected_behavior,
            'last_confidence': self.prediction_confidence,
            'prediction_accuracy': self._get_prediction_accuracy() * 100,
            'entropy': self._calculate_entropy(),
            'active_strategy': self.active_strategy_name,
        }
    
    def _get_prediction_accuracy(self) -> float:
        """Calculate recent prediction accuracy."""
        if self.total_rounds < 10:
            return 0.33
        
        # Check strategy scores — positive scores mean beating random
        positive_strategies = [s for s in self.strategies if s.score > 0]
        if not positive_strategies:
            return 0.33
        
        best = max(positive_strategies, key=lambda s: s.score)
        # Convert score to an accuracy-like metric
        return min(0.85, 0.33 + best.score / 20.0)
    
    def get_insights(self) -> List[str]:
        """Get human-readable insights."""
        insights = []
        
        if self.total_rounds < 5:
            insights.append("Still gathering data...")
            return insights
        
        # Best performing strategy
        best = max(self.strategies, key=lambda s: s.score)
        if best.score > 1.0:
            meta_desc = {0: "direct prediction", 1: "counter-counter", 2: "triple counter"}
            level = meta_desc.get(best.meta_level, "")
            insights.append(f"Best strategy: {best.predictor.name} ({level}, score: {best.score:.1f})")
        
        # Win rate insight
        win_rate = self.wins / self.total_rounds * 100 if self.total_rounds > 0 else 0
        if win_rate > 45:
            insights.append(f"AI is dominating ({win_rate:.0f}% win rate) 🔥")
        elif win_rate > 36:
            insights.append(f"AI has edge ({win_rate:.0f}% win rate)")
        elif win_rate < 28:
            insights.append(f"Opponent is tough ({win_rate:.0f}% win rate)")
        
        # Entropy
        entropy = self._calculate_entropy()
        if entropy < 0.85:
            insights.append(f"Opponent has low randomness (entropy: {entropy:.2f}) — exploitable!")
        elif entropy > 0.96:
            insights.append(f"Opponent plays very randomly (entropy: {entropy:.2f})")
        
        # Behavior
        if self.detected_behavior not in ("unknown", "learning"):
            insights.append(f"Detected style: {self.detected_behavior}")
        
        # Strategy competition info
        active_count = sum(1 for s in self.strategies if s.score > 0)
        insights.append(f"{active_count}/{len(self.strategies)} strategies performing above baseline")
        
        return insights
    
    def get_strategy_leaderboard(self, top_n: int = 5) -> List[Dict]:
        """Get the top performing strategies."""
        sorted_strategies = sorted(self.strategies, key=lambda s: s.score, reverse=True)
        leaderboard = []
        for s in sorted_strategies[:top_n]:
            leaderboard.append({
                'name': s.name,
                'score': s.score,
                'predictor': s.predictor.name,
                'meta_level': s.meta_level,
            })
        return leaderboard
    
    def save_brain(self):
        """Save learned state to disk."""
        brain_data = {
            'version': '4.0',
            'opp_history': self.opp_history[-100:],  # Save last 100 moves
            'my_history': self.my_history[-100:],
            'results': self.results[-100:],
            'total_rounds': self.total_rounds,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'strategy_scores': {s.name: s.score for s in self.strategies},
        }
        
        try:
            with open(self.save_file, 'wb') as f:
                pickle.dump(brain_data, f)
            return True
        except Exception as e:
            print(f"Warning: Could not save brain: {e}")
            return False
    
    def load_brain(self):
        """Load previously learned state from disk."""
        if not os.path.exists(self.save_file):
            return False
        
        try:
            with open(self.save_file, 'rb') as f:
                brain_data = pickle.load(f)
            
            # Only load v4.0 brains
            if brain_data.get('version') != '4.0':
                print("🧠 Old brain format detected — starting fresh with v4.0 engine.")
                return False
            
            self.opp_history = brain_data.get('opp_history', [])
            self.my_history = brain_data.get('my_history', [])
            self.results = brain_data.get('results', [])
            self.total_rounds = brain_data.get('total_rounds', 0)
            self.wins = brain_data.get('wins', 0)
            self.losses = brain_data.get('losses', 0)
            self.draws = brain_data.get('draws', 0)
            
            # Restore strategy scores
            saved_scores = brain_data.get('strategy_scores', {})
            for strategy in self.strategies:
                if strategy.name in saved_scores:
                    strategy.score = saved_scores[strategy.name]
            
            if self.total_rounds > 0:
                print(f"\n🧠 Beast AI brain loaded! Previously learned from {self.total_rounds} rounds.")
                print(f"   Lifetime stats: {self.wins}-{self.losses}-{self.draws}")
                best = max(self.strategies, key=lambda s: s.score)
                if best.score > 0:
                    print(f"   Best strategy: {best.name} (score: {best.score:.1f})")
            
            return True
        except Exception as e:
            print(f"Warning: Could not load brain: {e}")
            return False
    
    def reset(self):
        """Full reset — forget everything."""
        self.opp_history.clear()
        self.my_history.clear()
        self.results.clear()
        self.total_rounds = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_streak = 0
        self._exploitation_counter = 0
        self._nash_rounds = 0
        self._nash_cooldown = 0
        self._recent_my_moves.clear()
        
        for strategy in self.strategies:
            strategy.reset_score()
            strategy.predictor.reset()
        
        print("🧠 Beast AI brain reset — all knowledge forgotten!")
