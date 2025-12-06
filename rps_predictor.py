#!/usr/bin/env python3
"""
Advanced RPS Predictor - Pattern & Behavioral Analysis Engine (v2.0)

Key Improvements:
- Meta-learning with prediction accuracy tracking
- Proper Nash equilibrium fallback
- Confidence calibration based on performance
- Counter-exploitation detection
- Increased sample size requirements
"""

import random
import math
import pickle
import os
from collections import deque, Counter, defaultdict
from typing import Optional, Dict, Tuple, List


class PredictionTracker:
    """Tracks prediction accuracy to enable meta-learning."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.total_predictions = 0
        self.correct_predictions = 0
    
    def record(self, predicted_move: str, actual_move: str, confidence: float):
        """Record a prediction and its outcome."""
        correct = (predicted_move == actual_move)
        self.predictions.append({
            'predicted': predicted_move,
            'actual': actual_move,
            'confidence': confidence,
            'correct': correct
        })
        
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1
    
    def get_accuracy(self) -> float:
        """Get recent prediction accuracy (0.0-1.0)."""
        if not self.predictions:
            return 0.33  # Random baseline
        
        correct = sum(1 for p in self.predictions if p['correct'])
        return correct / len(self.predictions)
    
    def get_lifetime_accuracy(self) -> float:
        """Get overall prediction accuracy."""
        if self.total_predictions == 0:
            return 0.33
        return self.correct_predictions / self.total_predictions
    
    def should_trust_predictions(self) -> bool:
        """Returns True if predictions are better than random."""
        accuracy = self.get_accuracy()
        # Need to be significantly better than random (33.33%) to trust
        return accuracy > 0.40 and len(self.predictions) >= 10


class StrategyPerformance:
    """Tracks performance of individual prediction strategies."""
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.strategy_predictions = defaultdict(lambda: deque(maxlen=window_size))
        self.strategy_total = defaultdict(int)
        self.strategy_correct = defaultdict(int)
    
    def record(self, strategy_name: str, predicted_move: str, actual_move: str):
        """Record a strategy prediction and its outcome."""
        correct = (predicted_move == actual_move)
        self.strategy_predictions[strategy_name].append(correct)
        self.strategy_total[strategy_name] += 1
        if correct:
            self.strategy_correct[strategy_name] += 1
    
    def get_strategy_accuracy(self, strategy_name: str) -> float:
        """Get recent accuracy for a specific strategy."""
        predictions = self.strategy_predictions[strategy_name]
        if not predictions:
            return 0.33  # Random baseline
        return sum(predictions) / len(predictions)
    
    def get_best_strategy(self) -> Optional[str]:
        """Get the currently best-performing strategy."""
        best_strategy = None
        best_accuracy = 0.35  # Must beat random by margin
        
        for strategy_name, predictions in self.strategy_predictions.items():
            if len(predictions) >= 5:  # Need minimum samples
                accuracy = sum(predictions) / len(predictions)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_strategy = strategy_name
        
        return best_strategy
    
    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get confidence multiplier for a strategy based on its performance."""
        predictions = self.strategy_predictions[strategy_name]
        if len(predictions) < 3:
            return 1.0  # Neutral weight for unproven strategies
        
        accuracy = sum(predictions) / len(predictions)
        
        # Convert accuracy to weight multiplier
        if accuracy >= 0.70:  # Excellent
            return 1.4
        elif accuracy >= 0.60:  # Very good
            return 1.25
        elif accuracy >= 0.50:  # Good
            return 1.1
        elif accuracy >= 0.40:  # Okay
            return 1.0
        elif accuracy >= 0.30:  # Below random
            return 0.8
        else:  # Terrible
            return 0.5


class RPSPredictor:
    """Advanced Rock-Paper-Scissors AI that predicts opponent moves."""
    
    MOVES = ['rock', 'paper', 'scissors']
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    BEATEN_BY = {'scissors': 'rock', 'rock': 'paper', 'paper': 'scissors'}
    
    def __init__(self, memory_size: int = 40, save_file: str = "rps_ai_brain.pkl"):
        """Initialize the predictor.
        
        Args:
            memory_size: Number of recent moves to remember (default: 40)
            save_file: Path to save/load learned patterns (default: rps_ai_brain.pkl)
        """
        self.memory_size = memory_size
        self.save_file = save_file
        
        # Move history
        self.player_moves = deque(maxlen=memory_size)
        self.ai_moves = deque(maxlen=memory_size)
        self.results = deque(maxlen=memory_size)  # 'w'=AI win, 'l'=AI loss, 'd'=draw
        
        # Pattern detection (STRICTER requirements)
        self.sequences = defaultdict(Counter)  # (move1, move2, ...) -> Counter(next_move)
        self.transitions = defaultdict(Counter)  # move -> Counter(next_move)
        
        # Behavioral modeling
        self.win_responses = Counter()  # What player does after AI wins (player loses)
        self.loss_responses = Counter()  # What player does after AI loses (player wins)
        self.draw_responses = Counter()  # What player does after draw
        self.move_preferences = Counter()  # Overall move frequencies
        
        # Win-Stay / Lose-Shift specific tracking
        self.win_repeats = 0  # How many times player REPEATS after they WIN
        self.win_switches = 0  # How many times player SWITCHES after they WIN
        self.loss_repeats = 0  # How many times player REPEATS after they LOSE
        self.loss_switches = 0  # How many times player SWITCHES after they LOSE
        
        # Meta-learning
        self.prediction_tracker = PredictionTracker(window_size=20)
        self.strategy_performance = StrategyPerformance(window_size=15)
        
        # Exploitation detection
        self.opponent_counters_ai = 0  # Count of times opponent beat AI's most common move
        self.recent_ai_moves_beaten = deque(maxlen=10)
        
        # State tracking
        self.total_rounds = 0
        self.current_streak = 0
        self.prediction_confidence = 0.0
        self.detected_behavior = "unknown"
        self.last_prediction = None
        self.last_prediction_strategy = None  # Track which strategy made the prediction
        self.last_prediction_correct = None
        
        # Statistics
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        # Try to load previous learning
        self.load_brain()
    
    def get_move(self) -> str:
        """Get AI's next move based on predictions.
        
        Returns:
            The AI's chosen move ('rock', 'paper', or 'scissors')
        """
        # First 8 moves: pure random (gathering data)
        if len(self.player_moves) < 8:
            return random.choice(self.MOVES)
        
        # Check if we're being exploited
        if self._detect_exploitation():
            # Being counter-played! Go random for unpredictability
            return random.choice(self.MOVES)
        
        # Get prediction of what player will do
        prediction = self._predict_player_move()
        
        self.last_prediction = prediction['move']
        self.last_prediction_strategy = prediction['strategy']  # Track strategy for performance analysis
        self.prediction_confidence = prediction['confidence']
        
       # HOT-HAND DETECTION: Boost confidence when a strategy is working
        best_strategy_name = self.strategy_performance.get_best_strategy()
        if best_strategy_name and prediction['strategy'] == best_strategy_name:
            strategy_accuracy = self.strategy_performance.get_strategy_accuracy(best_strategy_name)
            
            # Aggressive exploitation: if we're winning with this strategy, trust it more
            if self.current_streak >= 3 and strategy_accuracy > 0.60:
                # Hot hand! Boost confidence significantly
                prediction['confidence'] *= 1.5
                prediction['confidence'] = min(0.95, prediction['confidence'])
        
        # Check if our predictions are actually working
        prediction_accuracy = self.prediction_tracker.get_accuracy()
        should_trust = self.prediction_tracker.should_trust_predictions()
        
        # AGGRESSIVE confidence thresholds - trust predictions more
        if should_trust and prediction_accuracy > 0.50:
            # Our predictions are working very well - be very aggressive!
            high_conf_threshold = 0.40
            med_conf_threshold = 0.30
        elif should_trust and prediction_accuracy > 0.42:
            # Our predictions are working - be aggressive!
            high_conf_threshold = 0.45
            med_conf_threshold = 0.35
        elif prediction_accuracy > 0.36:
            # Marginal performance - still be somewhat aggressive
            high_conf_threshold = 0.55
            med_conf_threshold = 0.45
        else:
            # Predictions failing - use random
            high_conf_threshold = 0.70
            med_conf_threshold = 0.60
        
        # Decide move based on AGGRESSIVE confidence
        if prediction['confidence'] >= high_conf_threshold:
            # High confidence: counter the prediction AGGRESSIVELY
            ai_move = self.BEATEN_BY[prediction['move']]
            
            # Only enforce diversity if confidence is not extremely high
            if prediction['confidence'] < 0.75 and len(self.ai_moves) >= 5:
                recent_ai = list(self.ai_moves)[-5:]
                if recent_ai.count(ai_move) >= 4:  # Only if 4+ times
                    # 30% chance to switch (less randomness)
                    if random.random() < 0.3:
                        other_moves = [m for m in self.MOVES if m != ai_move]
                        ai_move = random.choice(other_moves)
        
        elif prediction['confidence'] >= med_conf_threshold:
            # Medium confidence: weighted randomization (more aggressive)
            counter_move = self.BEATEN_BY[prediction['move']]
            
            # 75% counter, 25% random
            if random.random() < 0.75:
                ai_move = counter_move
            else:
                ai_move = random.choice(self.MOVES)
        
        else:
            # Low confidence: PURE Nash equilibrium (true random)
            ai_move = random.choice(self.MOVES)
        
        return ai_move
    
    def record_round(self, player_move: str, ai_move: str) -> str:
        """Record the result of a round.
        
        Args:
            player_move: The player's move
            ai_move: The AI's move
            
        Returns:
            Result from AI's perspective: 'w' (win), 'l' (loss), or 'd' (draw)
        """
        result = self._determine_result(ai_move, player_move)
        
        # Track prediction accuracy (BEFORE updating history)
        if self.last_prediction is not None and len(self.player_moves) > 0:
            self.prediction_tracker.record(
                self.last_prediction,
                player_move,
                self.prediction_confidence
            )
            # NEW: Track per-strategy performance
            if self.last_prediction_strategy:
                self.strategy_performance.record(
                    self.last_prediction_strategy,
                    self.last_prediction,
                    player_move
                )
            self.last_prediction_correct = (self.last_prediction == player_move)
        
        # Update history
        self.player_moves.append(player_move)
        self.ai_moves.append(ai_move)
        self.results.append(result)
        
        # Update statistics
        self.total_rounds += 1
        if result == 'w':
            self.wins += 1
            self.current_streak = max(1, self.current_streak + 1)
        elif result == 'l':
            self.losses += 1
            self.current_streak = min(-1, self.current_streak - 1)
            # Track if opponent beat our move
            self.recent_ai_moves_beaten.append(ai_move)
        else:
            self.draws += 1
            self.current_streak = 0
        
        # Learn patterns
        self._learn_patterns(player_move)
        self._learn_behaviors(player_move, result)
        
        # Decay old patterns periodically
        self._decay_old_patterns()
        
        # Analyze opponent style
        if self.total_rounds % 15 == 0:
            self._analyze_opponent_style()
        
        return result
    
    def _predict_player_move(self) -> Dict[str, any]:
        """Predict what the player will do next using ensemble of strategies.
        
        Returns:
            Dict with 'move' (prediction) and 'confidence' (0.0-1.0)
        """
        predictions = []
        
        # Strategy 1: Sequence pattern detection (n-grams) - AGGRESSIVE with recency weighting
        for length in [4, 3, 2]:
            if len(self.player_moves) >= length + 2:
                seq = tuple(list(self.player_moves)[-length:])
                if seq in self.sequences and self.sequences[seq]:
                    total = sum(self.sequences[seq].values())
                    
                    # AGGRESSIVE sample requirements - lower thresholds
                    min_samples = {4: 2, 3: 3, 2: 4}[length]
                    if total >= min_samples:
                        most_common = self.sequences[seq].most_common(1)[0]
                        move, count = most_common
                        
                        # Pattern strength-based confidence
                        raw_conf = count / total
                        # Lower threshold for pattern detection
                        if raw_conf >= 0.40:  # 40% consistency is enough
                            sample_bonus = min(1.0, total / (min_samples * 1.2))
                            # Longer patterns get higher confidence
                            length_bonus = {4: 0.95, 3: 0.90, 2: 0.85}[length]
                            
                            # RECENCY WEIGHTING: boost if pattern is working recently
                            recency_bonus = 1.0
                            if total > min_samples * 2 and len(self.player_moves) >= length + 5:
                                # Check if pattern worked in last few occurrences
                                recent_matches = 0
                                for j in range(min(5, len(self.player_moves) - length)):
                                    check_seq = tuple(list(self.player_moves)[-(length+1+j):-(1+j)])
                                    if check_seq == seq and len(self.player_moves) > (1+j):
                                        actual_next = self.player_moves[-(1+j)] if (1+j) <= len(self.player_moves) else None
                                        if actual_next == move:
                                            recent_matches += 1
                                
                                if recent_matches >= 2:
                                    recency_bonus = 1.15  # 15% boost for recent consistency
                            
                            confidence = raw_conf * sample_bonus * length_bonus * recency_bonus
                            predictions.append(('pattern', move, confidence, f'{length}-gram', total))
        
        # Strategy 2: Transition probability (MARKOV) - AGGRESSIVE
        if len(self.player_moves) >= 4:
            last_move = self.player_moves[-1]
            if last_move in self.transitions and self.transitions[last_move]:
                total = sum(self.transitions[last_move].values())
                if total >= 3:  # Lower requirement
                    most_common = self.transitions[last_move].most_common(1)[0]
                    move, count = most_common
                    
                    raw_conf = count / total
                    if raw_conf >= 0.40:
                        sample_bonus = min(1.0, total / 8)
                        confidence = raw_conf * sample_bonus * 0.85
                        predictions.append(('transition', move, confidence, 'markov', total))
        
        # Strategy 3: Win-Stay / Lose-Shift Detection - COMPLETELY REWRITTEN
        # This is the critical fix for the 39% win rate against Win-Stay opponents
        if len(self.player_moves) >= 3 and len(self.results) >= 2:
            total_after_wins = self.win_repeats + self.win_switches
            total_after_losses = self.loss_repeats + self.loss_switches
            
            # Check for Win-Stay behavior (player repeats after winning)
            if total_after_wins >= 3:  # Have enough data
                repeat_rate_after_wins = self.win_repeats / total_after_wins
                
                # LOWERED threshold from 65% to 60% for earlier detection
                if repeat_rate_after_wins >= 0.60:  # Strong Win-Stay detected
                    last_result = self.results[-1]
                    last_player_move = self.player_moves[-1]
                    
                    if last_result == 'l':  # AI lost, so opponent WON
                        # Opponent will likely REPEAT their winning move
                        # INCREASED confidence from 0.75 to 0.70-0.85 range
                        base_conf = 0.70
                        strength_bonus = min(0.15, (repeat_rate_after_wins - 0.60) * 0.375)
                        sample_bonus = min(0.05, (total_after_wins - 3) * 0.01)
                        confidence = base_conf + strength_bonus + sample_bonus
                        
                        predictions.append(('win-stay', last_player_move, confidence, 
                                          f'repeats-{repeat_rate_after_wins:.0%}', total_after_wins))
            
            # Check for Lose-Shift behavior (player switches after losing)
            if total_after_losses >= 3:  # Have enough data
                switch_rate_after_losses = self.loss_switches / total_after_losses
                
                # LOWERED threshold from 65% to 60% for earlier detection
                if switch_rate_after_losses >= 0.60:  # Strong Lose-Shift detected
                    last_result = self.results[-1]
                    last_player_move = self.player_moves[-1]
                    
                    if last_result == 'w':  # AI won, so opponent LOST
                        # Opponent will likely SWITCH from their losing move
                        other_moves = [m for m in self.MOVES if m != last_player_move]
                        
                        # IMPROVED: Higher confidence, split between two options
                        base_conf = 0.50  # Increased from 0.42
                        strength_bonus = min(0.10, (switch_rate_after_losses - 0.60) * 0.25)
                        confidence = base_conf + strength_bonus
                        
                        # Add both possible switch targets
                        for other_move in other_moves:
                            predictions.append(('lose-shift', other_move, confidence, 
                                              f'switches-{switch_rate_after_losses:.0%}', total_after_losses))

        
        # Strategy 4: Behavioral response - General outcome-based patterns
        # Humans tend to avoid repeating the same move consecutively
        if len(self.player_moves) >= 8:
            # Count consecutive repeats in history
            repeat_count = 0
            for i in range(len(self.player_moves) - 1):
                if self.player_moves[i] == self.player_moves[i+1]:
                    repeat_count += 1
            
            repeat_rate = repeat_count / (len(self.player_moves) - 1)
            
            # If they rarely repeat (anti-repetition bias)
            if repeat_rate < 0.25 and len(self.player_moves) >= 12:
                last_move = self.player_moves[-1]
                other_moves = [m for m in self.MOVES if m != last_move]
                
                # Calculate confidence based on how strongly they avoid repeating
                base_conf = 0.35 + (0.25 - repeat_rate) * 0.8
                split_conf = base_conf / 2  # Split between two possible moves
                
                # Add predictions for both non-repeating moves
                for other_move in other_moves:
                    predictions.append(('anti-repeat', other_move, split_conf, 'avoids-repetition', len(self.player_moves)))
        
        # Strategy 5: Frequency analysis - AGGRESSIVE
        if self.move_preferences:
            total = sum(self.move_preferences.values())
            if total >= 10:  # Lower requirement
                most_common = self.move_preferences.most_common(1)[0]
                move, count = most_common
                
                raw_conf = count / total
                if raw_conf >= 0.42:  # Lower threshold
                    confidence = raw_conf * 0.75  # Higher max
                    predictions.append(('frequency', move, confidence, 'preference', total))
        
        # ENSEMBLE VOTING - IMPROVED with strategy performance weighting
        if predictions:
            move_votes = defaultdict(lambda: {'weighted_conf': 0.0, 'count': 0, 'max_conf': 0.0, 'strategies': []})
            
            for strategy_type, move, conf, detail, samples in predictions:
                # Apply strategy performance weight
                strategy_weight = self.strategy_performance.get_strategy_weight(strategy_type)
                weighted_conf = conf * strategy_weight
                
                # PRIORITY BOOST: Win-Stay and Lose-Shift predictions get extra weight
                # These are behavioral patterns that deserve priority
                if strategy_type in ['win-stay', 'lose-shift']:
                    weighted_conf += 0.15  # Significant boost to compete with other strategies
                
                move_votes[move]['weighted_conf'] += weighted_conf
                move_votes[move]['count'] += 1
                move_votes[move]['max_conf'] = max(move_votes[move]['max_conf'], conf)
                move_votes[move]['strategies'].append((strategy_type, conf, weighted_conf))
            
            # Score each move
            best_move = None
            best_score = 0
            best_strategy = None
            
            for move, votes in move_votes.items():
                # Use WEIGHTED confidence with agreement bonus
                base_conf = votes['max_conf']
                weighted_bonus = (votes['weighted_conf'] - base_conf) * 0.5  # Bonus from other weighted strategies
                agreement_bonus = (votes['count'] - 1) * 0.10  # Agreement bonus
                ensemble_score = base_conf + weighted_bonus + agreement_bonus
                
                if ensemble_score > best_score:
                    best_score = ensemble_score
                    best_move = move
                    # Find the primary strategy (highest confidence)
                    best_strategy = max(votes['strategies'], key=lambda x: x[1])[0]
            
            # Final confidence - HIGHER CAP
            final_confidence = min(0.92, best_score)  # Allow up to 92%
            
            # Apply prediction accuracy calibration
            accuracy = self.prediction_tracker.get_accuracy()
            if accuracy < 0.40:
                # Predictions failing - reduce confidence
                final_confidence *= 0.7
            
            return {
                'move': best_move,
                'confidence': final_confidence,
                'strategy': best_strategy,
                'detail': ''
            }
        else:
            # No patterns detected: random guess
            return {
                'move': random.choice(self.MOVES),
                'confidence': 0.25,  # Very low confidence
                'strategy': 'nash',
                'detail': 'random'
            }
    
    def _learn_patterns(self, player_move: str):
        """Learn patterns from player's move history."""
        self.move_preferences[player_move] += 1
        
        # Learn n-gram sequences (2-4 length)
        for length in range(2, 5):
            if len(self.player_moves) >= length:
                seq = tuple(list(self.player_moves)[-(length+1):-1])
                next_move = player_move
                self.sequences[seq][next_move] += 1
        
        # Learn transition probabilities
        if len(self.player_moves) >= 1:
            prev_move = self.player_moves[-1]
            self.transitions[prev_move][player_move] += 1
    
    def _learn_behaviors(self, player_move: str, result: str):
        """Learn behavioral patterns (how player responds to outcomes)."""
        if len(self.results) >= 2 and len(self.player_moves) >= 2:
            # Look at the PREVIOUS result (before current move was made)
            prev_result = self.results[-2]
            prev_player_move = self.player_moves[-2]
            current_player_move = player_move
            
            # Track general responses
            if prev_result == 'w':
                self.win_responses[current_player_move] += 1
            elif prev_result == 'l':
                self.loss_responses[current_player_move] += 1
            else:
                self.draw_responses[current_player_move] += 1
            
            # Track Win-Stay / Lose-Shift behavior
            did_repeat = (current_player_move == prev_player_move)
            
            if prev_result == 'l':  # AI lost (player WON)
                if did_repeat:
                    self.win_repeats += 1  # Player repeated after winning
                else:
                    self.win_switches += 1  # Player switched after winning
            
            elif prev_result == 'w':  # AI won (player LOST)
                if did_repeat:
                    self.loss_repeats += 1  # Player repeated after losing
                else:
                    self.loss_switches += 1  # Player switched after losing
    
    def _decay_old_patterns(self):
        """Decay old pattern weights to prioritize recent behavior."""
        if self.total_rounds % 25 == 0 and self.total_rounds > 0:
            # Reduce all pattern counts by 10% every 25 rounds
            # This helps AI adapt when opponent changes strategy
            for seq in list(self.sequences.keys()):
                for move in list(self.sequences[seq].keys()):
                    self.sequences[seq][move] = max(1, int(self.sequences[seq][move] * 0.9))
            
            for move_from in list(self.transitions.keys()):
                for move_to in list(self.transitions[move_from].keys()):
                    self.transitions[move_from][move_to] = max(1, int(self.transitions[move_from][move_to] * 0.9))
    
    def _detect_exploitation(self) -> bool:
        """Detect if opponent is counter-playing AI's patterns."""
        if len(self.ai_moves) < 15 or len(self.recent_ai_moves_beaten) < 8:
            return False
        
        # Check if we're losing badly
        if self.total_rounds >= 20:
            win_rate = self.wins / self.total_rounds
            if win_rate < 0.25:  # Less than 25% - being exploited
                return True
        
        # Check if opponent is consistently beating our most common moves
        recent_ai = list(self.ai_moves)[-15:]
        ai_counter = Counter(recent_ai)
        most_common_ai = ai_counter.most_common(1)[0][0]
        
        # Check if this move is being beaten consistently
        beaten_moves = list(self.recent_ai_moves_beaten)
        if beaten_moves.count(most_common_ai) >= 5:
            return True
        
        return False
    
    def _calculate_entropy(self) -> float:
        """Calculate entropy of opponent moves (measure of randomness)."""
        if len(self.player_moves) < 10:
            return 1.0
        
        recent = list(self.player_moves)[-20:]
        total = len(recent)
        
        entropy = 0.0
        for move in self.MOVES:
            count = recent.count(move)
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 (max entropy for 3 options is log2(3) ≈ 1.585)
        max_entropy = math.log2(3)
        return entropy / max_entropy
    
    def _analyze_opponent_style(self):
        """Analyze and classify opponent's playing style."""
        if len(self.player_moves) < 15:
            self.detected_behavior = "learning"
            return
        
        # Calculate entropy (randomness)
        entropy = self._calculate_entropy()
        
        # Check prediction accuracy
        accuracy = self.prediction_tracker.get_accuracy()
        
        # Classify based on entropy and predictability
        if entropy > 0.95:  # High entropy = random
            self.detected_behavior = "random"
        elif accuracy > 0.50:  # We can predict them well
            # Check what type of pattern
            recent_10 = list(self.player_moves)[-10:]
            unique = len(set(recent_10))
            
            if unique <= 3:
                self.detected_behavior = "repetitive"
            else:
                # Check for sequences
                has_pattern = False
                for seq_len in [2, 3, 4]:
                    if len(self.player_moves) >= seq_len * 3:
                        seq = tuple(list(self.player_moves)[-seq_len:])
                        if seq in self.sequences and sum(self.sequences[seq].values()) >= 3:
                            has_pattern = True
                            break
                
                if has_pattern:
                    self.detected_behavior = "pattern-based"
                else:
                    # Check win-stay behavior
                    if sum(self.loss_responses.values()) >= 5:
                        total = sum(self.loss_responses.values())
                        most_common = self.loss_responses.most_common(1)[0][1]
                        if most_common / total > 0.55:
                            self.detected_behavior = "win-stay"
                        else:
                            self.detected_behavior = "frequency-biased"
                    else:
                        self.detected_behavior = "frequency-biased"
        elif accuracy > 0.38:
            self.detected_behavior = "somewhat-predictable"
        else:
            self.detected_behavior = "adaptive"
    
    def _determine_result(self, ai_move: str, player_move: str) -> str:
        """Determine the result of a round from AI's perspective."""
        if ai_move == player_move:
            return 'd'
        elif self.BEATS[ai_move] == player_move:
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
            'prediction_accuracy': self.prediction_tracker.get_accuracy() * 100,
            'entropy': self._calculate_entropy()
        }
    
    def get_insights(self) -> List[str]:
        """Get human-readable insights about the player."""
        insights = []
        
        # Prediction performance
        accuracy = self.prediction_tracker.get_accuracy()
        if self.total_rounds >= 15:
            if accuracy > 0.45:
                insights.append(f"AI predictions are working well ({accuracy*100:.0f}% accurate)")
            elif accuracy > 0.35:
                insights.append(f"AI predictions are marginal ({accuracy*100:.0f}% accurate)")
            else:
                insights.append(f"Opponent plays randomly ({accuracy*100:.0f}% prediction rate)")
        
        # Move preferences
        if self.move_preferences and self.total_rounds >= 15:
            total = sum(self.move_preferences.values())
            for move in self.MOVES:
                count = self.move_preferences.get(move, 0)
                pct = count / total * 100
                if pct >= 45:
                    insights.append(f"You favor {move} ({pct:.0f}%)")
        
        # Behavioral patterns
        if self.detected_behavior != "unknown":
            behavior_descriptions = {
                "pattern-based": "You follow predictable sequences",
                "win-stay": "You repeat moves after winning",
                "repetitive": "You tend to repeat the same move",
                "random": "You play very randomly",
                "adaptive": "You adapt and counter-play",
                "learning": "Still learning your style...",
                "frequency-biased": "You have move preferences",
                "somewhat-predictable": "You're somewhat predictable"
            }
            desc = behavior_descriptions.get(self.detected_behavior, "")
            if desc:
                insights.append(desc)
        
        # Entropy
        entropy = self._calculate_entropy()
        if self.total_rounds >= 15:
            if entropy < 0.85:
                insights.append(f"Your play has low randomness (entropy: {entropy:.2f})")
        
        return insights
    
    def save_brain(self):
        """Save learned patterns to disk for persistence."""
        brain_data = {
            'sequences': dict(self.sequences),
            'transitions': dict(self.transitions),
            'win_responses': dict(self.win_responses),
            'loss_responses': dict(self.loss_responses),
            'draw_responses': dict(self.draw_responses),
            'move_preferences': dict(self.move_preferences),
            'total_rounds': self.total_rounds,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws
        }
        
        try:
            with open(self.save_file, 'wb') as f:
                pickle.dump(brain_data, f)
            return True
        except Exception as e:
            print(f"Warning: Could not save brain: {e}")
            return False
    
    def load_brain(self):
        """Load previously learned patterns from disk."""
        if not os.path.exists(self.save_file):
            return False
        
        try:
            with open(self.save_file, 'rb') as f:
                brain_data = pickle.load(f)
            
            # Load learned patterns
            self.sequences = defaultdict(Counter, brain_data.get('sequences', {}))
            self.transitions = defaultdict(Counter, brain_data.get('transitions', {}))
            self.win_responses = Counter(brain_data.get('win_responses', {}))
            self.loss_responses = Counter(brain_data.get('loss_responses', {}))
            self.draw_responses = Counter(brain_data.get('draw_responses', {}))
            self.move_preferences = Counter(brain_data.get('move_preferences', {}))
            
            # Load lifetime stats (but reset current game stats)
            saved_rounds = brain_data.get('total_rounds', 0)
            if saved_rounds > 0:
                print(f"\n🧠 AI Brain loaded! Previously learned from {saved_rounds} rounds.")
                print(f"   Lifetime stats: {brain_data.get('wins', 0)}-{brain_data.get('losses', 0)}-{brain_data.get('draws', 0)}")
            
            return True
        except Exception as e:
            print(f"Warning: Could not load brain: {e}")
            return False
    
    def reset_brain(self):
        """Reset all learned patterns (forget everything)."""
        self.sequences = defaultdict(Counter)
        self.transitions = defaultdict(Counter)
        self.win_responses = Counter()
        self.loss_responses = Counter()
        self.draw_responses = Counter()
        self.move_preferences = Counter()
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_rounds = 0
        
        # Delete save file
        if os.path.exists(self.save_file):
            os.remove(self.save_file)
            print("🧠 AI brain reset! Starting fresh.")


# Quick test
if __name__ == "__main__":
    print("🎯 RPS Predictor v2.0 - Quick Test")
    print("=" * 50)
    
    ai = RPSPredictor()
    
    # Test 1: Pattern
    print("\n[TEST 1] Pattern: rock → paper → scissors → ...")
    pattern = ['rock', 'paper', 'scissors'] * 10
    
    for i, player_move in enumerate(pattern):
        ai_move = ai.get_move()
        result = ai.record_round(player_move, ai_move)
        
        if i >= 5 and i % 5 == 0:
            result_text = {"w": "WIN", "l": "LOSS", "d": "DRAW"}[result]
            print(f"  Round {i+1}: {result_text} (conf: {ai.prediction_confidence:.0%}, acc: {ai.prediction_tracker.get_accuracy()*100:.0f}%)")
    
    stats = ai.get_stats()
    print(f"\n  Final: {stats['win_rate']:.1f}% win rate, {stats['prediction_accuracy']:.0f}% prediction accuracy")
    
    # Test 2: Random
    print("\n[TEST 2] Random opponent")
    ai2 = RPSPredictor()
    random.seed(42)
    
    for i in range(30):
        player_move = random.choice(ai2.MOVES)
        ai_move = ai2.get_move()
        result = ai2.record_round(player_move, ai_move)
    
    stats2 = ai2.get_stats()
    print(f"  Final: {stats2['win_rate']:.1f}% win rate (should be ~33%)")
    print(f"  Detection: {stats2['detected_behavior']}")
    print(f"  Entropy: {stats2['entropy']:.2f}")
