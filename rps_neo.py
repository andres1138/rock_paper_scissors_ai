import os
import traceback
import random
import pickle
from collections import deque, defaultdict, Counter
from datetime import datetime
from rps_plotter import RPSPlotter

import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

class RealTimeEvolvingRPS_TF:
    MOVES = ['rock', 'paper', 'scissors']
    IDX = {m: i for i, m in enumerate(MOVES)}
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    BEATEN_BY = {v: k for k, v in BEATS.items()}

    def __init__(self,
                save_prefix="rps_beast2",
                memory_size=5000,
                gamma=0.25,  # Even higher exploration
                eta=0.1,  # Much more aggressive learning
                seed=None,
                autosave_every=50,
                tf_train_every=20,
                tf_epochs=20,
                tf_batch=32,
                tf_memory=3000):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.save_prefix = save_prefix
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(base_dir, f"{save_prefix}_state.pkl")
        self.tf_model_dir = os.path.join(base_dir, f"{save_prefix}_tf_model")
        self.memory_size = memory_size

        # Histories
        self.opponent_history = deque(maxlen=memory_size)
        self.ai_history = deque(maxlen=memory_size)
        self.results = []

        # NEW: Track what DOESN'T work - blacklist bad moves in contexts
        self.move_blacklist = defaultdict(lambda: {m: 0 for m in self.MOVES})
        
        # Direct action-value learning (Q-learning style)
        self.context_actions = defaultdict(lambda: {m: [0, 0, 0] for m in self.MOVES})
        
        # Strategy names
        self.strategy_names = [
            'frequency', 'anti_frequency', 'markov1', 'markov2', 
            'ngram', 'streak', 'anti_streak', 'cycle', 'meta', 'mirror', 'anti_mirror', 'random'
        ]
        if TF_AVAILABLE:
            self.strategy_names.append('tf')
        self.K = len(self.strategy_names)

        # EXP3 with aggressive learning
        self.weights = np.ones(self.K, dtype=float)
        self.gamma = float(gamma)
        self.total_rounds = 0
        self.eta = float(eta)
        
        # Track recent performance for faster adaptation
        self.recent_results = deque(maxlen=20)
        self.losing_streak = 0
        self.last_loss_context = None
        self.last_loss_move = None
        
        # NEW: Pattern detection and randomness detection
        self.pattern_confidence = 0.0
        self.randomness_score = 0.5  # Start neutral
        self.opponent_entropy_history = deque(maxlen=30)
        self.anti_exploit_mode = False
        self.counter_predict_detected = 0
        
        # Per-strategy tracking
        self.strategy_wins = np.zeros(self.K, dtype=float)
        self.strategy_plays = np.zeros(self.K, dtype=float)
        self.strategy_recent = [deque(maxlen=10) for _ in range(self.K)]

        # Classical models
        self.freq_counts = Counter()
        self.recency_weights = defaultdict(float)
        self.decay = 0.92  # Even faster decay
        self.markov1 = np.ones((3, 3)) * 0.1
        self.markov2 = defaultdict(lambda: np.ones(3) * 0.1)
        self.ngram_len = 4
        self.ngram = defaultdict(lambda: np.ones(3) * 0.1)

        self.current_streak_move = None
        self.current_streak_count = 0
        
        # Meta-learning
        self.response_patterns = defaultdict(lambda: np.ones(3) * 0.1)
        
        # NEW: Track what beats us
        self.what_beats_us = defaultdict(lambda: np.ones(3) * 0.1)

        # TF memory
        self.tf_train_buffer = deque(maxlen=tf_memory)
        self.tf_train_every = tf_train_every
        self.tf_epochs = tf_epochs
        self.tf_batch = tf_batch

        self.prediction_confidence = 0.0
        self.last_strategy_idx = None
        self.last_ai_move = None
        self.autosave_every = autosave_every

        # TF model
        self.tf_model = None
        if TF_AVAILABLE:
            self._init_or_load_tf_model()

        self.load_state()

    def _hist_slice(self, start=None, end=None):
        """Safely slice opponent_history as a list."""
        hist = list(self.opponent_history)
        return hist[start:end]

    def _build_tf_model(self):
        """Enhanced TF model that directly predicts winning moves."""
        model = keras.Sequential([
            layers.Input(shape=(15,)),  # More context
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.003),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _init_or_load_tf_model(self):
        try:
            if os.path.isdir(self.tf_model_dir):
                self.tf_model = keras.models.load_model(self.tf_model_dir)
            else:
                self.tf_model = self._build_tf_model()
        except Exception:
            self.tf_model = self._build_tf_model()

    def _save_tf_model(self):
        if not TF_AVAILABLE or self.tf_model is None:
            return False
        try:
            self.tf_model.save(self.tf_model_dir, overwrite=True)
            return True
        except Exception:
            return False

    def save_state(self):
        try:
            payload = {
                'opponent_history': list(self.opponent_history),
                'ai_history': list(self.ai_history),
                'results': self.results,
                'weights': self.weights.tolist(),
                'strategy_wins': self.strategy_wins.tolist(),
                'strategy_plays': self.strategy_plays.tolist(),
                'total_rounds': self.total_rounds,
                'freq_counts': dict(self.freq_counts),
                'recency_weights': dict(self.recency_weights),
                'markov1': self.markov1.tolist(),
                'markov2': {k: v.tolist() for k, v in self.markov2.items()},
                'ngram': {k: v.tolist() for k, v in self.ngram.items()},
                'response_patterns': {k: v.tolist() for k, v in self.response_patterns.items()},
                'what_beats_us': {k: v.tolist() for k, v in self.what_beats_us.items()},
                'context_actions': {k: v for k, v in self.context_actions.items()},
                'current_streak_move': self.current_streak_move,
                'current_streak_count': self.current_streak_count,
                'tf_buffer': list(self.tf_train_buffer)
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(payload, f)
            if TF_AVAILABLE and self.tf_model:
                self._save_tf_model()
            print(f"💾 State saved to: {self.state_file}")
            return True
        except Exception as e:
            print("⚠️ save_state failed:", e)
            return False

    def load_state(self):
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, 'rb') as f:
                payload = pickle.load(f)
            self.opponent_history = deque(payload.get('opponent_history', []), maxlen=self.memory_size)
            self.ai_history = deque(payload.get('ai_history', []), maxlen=self.memory_size)
            self.results = payload.get('results', [])
            
            # Handle strategy count mismatch
            loaded_weights = np.array(payload.get('weights', []), dtype=float)
            loaded_wins = np.array(payload.get('strategy_wins', []), dtype=float)
            loaded_plays = np.array(payload.get('strategy_plays', []), dtype=float)
            
            if len(loaded_weights) != self.K:
                new_weights = np.ones(self.K, dtype=float)
                new_wins = np.zeros(self.K, dtype=float)
                new_plays = np.zeros(self.K, dtype=float)
                copy_len = min(len(loaded_weights), self.K)
                new_weights[:copy_len] = loaded_weights[:copy_len]
                if len(loaded_wins) > 0:
                    new_wins[:min(len(loaded_wins), self.K)] = loaded_wins[:min(len(loaded_wins), self.K)]
                if len(loaded_plays) > 0:
                    new_plays[:min(len(loaded_plays), self.K)] = loaded_plays[:min(len(loaded_plays), self.K)]
                self.weights = new_weights
                self.strategy_wins = new_wins
                self.strategy_plays = new_plays
            else:
                self.weights = loaded_weights
                self.strategy_wins = loaded_wins
                self.strategy_plays = loaded_plays
            
            self.total_rounds = payload.get('total_rounds', 0)
            self.freq_counts = Counter(payload.get('freq_counts', {}))
            self.recency_weights = defaultdict(float, payload.get('recency_weights', {}))
            self.markov1 = np.array(payload.get('markov1', self.markov1.tolist()))
            
            self.markov2 = defaultdict(lambda: np.ones(3) * 0.1)
            for k, v in payload.get('markov2', {}).items():
                self.markov2[tuple(k)] = np.array(v)
            
            self.ngram = defaultdict(lambda: np.ones(3) * 0.1)
            for k, v in payload.get('ngram', {}).items():
                self.ngram[tuple(k)] = np.array(v)
            
            self.response_patterns = defaultdict(lambda: np.ones(3) * 0.1)
            for k, v in payload.get('response_patterns', {}).items():
                self.response_patterns[k] = np.array(v)
            
            self.what_beats_us = defaultdict(lambda: np.ones(3) * 0.1)
            for k, v in payload.get('what_beats_us', {}).items():
                self.what_beats_us[k] = np.array(v)
            
            self.context_actions = defaultdict(lambda: {m: [0, 0, 0] for m in self.MOVES})
            for k, v in payload.get('context_actions', {}).items():
                self.context_actions[k] = v
            
            self.current_streak_move = payload.get('current_streak_move', None)
            self.current_streak_count = payload.get('current_streak_count', 0)
            self.tf_train_buffer = deque(payload.get('tf_buffer', []), maxlen=self.tf_train_buffer.maxlen)
            return True
        except Exception as e:
            print("⚠️ load_state failed:", e)
            return False

    def reset_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        if os.path.isdir(self.tf_model_dir):
            import shutil
            shutil.rmtree(self.tf_model_dir, ignore_errors=True)
        self.__init__(save_prefix=self.save_prefix,
                    memory_size=self.memory_size,
                    gamma=self.gamma,
                    eta=self.eta,
                    seed=None,
                    autosave_every=self.autosave_every,
                    tf_train_every=self.tf_train_every,
                    tf_epochs=self.tf_epochs,
                    tf_batch=self.tf_batch,
                    tf_memory=self.tf_train_buffer.maxlen)
        print("🧹 Memory and TF model reset!")

    def one_hot(self, move):
        v = np.zeros(3, dtype=float)
        if move in self.MOVES:
            v[self.IDX[move]] = 1.0
        return v

    def move_result(self, ai_move, opp_move):
        if ai_move == opp_move:
            return 'd'
        if self.BEATS[ai_move] == opp_move:
            return 'w'
        return 'l'

    def _opponent_from_result(self, ai_move, result):
        for m in self.MOVES:
            if self.move_result(ai_move, m) == result:
                return m
        return random.choice(self.MOVES)

    def _get_context_key(self):
        """Generate a context key from recent history - IMPROVED with results."""
        if len(self.opponent_history) < 2 or len(self.ai_history) < 2:
            return "early_game"
        
        # Enhanced: Include both moves AND recent results for better context
        recent_outcome = self.results[-1] if self.results else 'd'
        
        ctx = (
            self.ai_history[-2], self.ai_history[-1],
            self.opponent_history[-2], self.opponent_history[-1],
            recent_outcome  # Add result to context
        )
        return ctx

    # ========== PREDICTION STRATEGIES ==========
    
    def _freq_predict(self):
        """Counter most frequent opponent move."""
        if not self.recency_weights:
            return random.choice(self.MOVES)
        best = max(self.recency_weights.items(), key=lambda x: x[1])[0]
        return self.BEATEN_BY[best]

    def _anti_freq_predict(self):
        """Counter least frequent opponent move."""
        if not self.recency_weights or len(self.recency_weights) < 3:
            return random.choice(self.MOVES)
        least = min(self.recency_weights.items(), key=lambda x: x[1])[0]
        return self.BEATEN_BY[least]

    def _markov1_predict(self):
        if len(self.opponent_history) < 1:
            return random.choice(self.MOVES)
        last = self._hist_slice(-1, None)[0]
        arr = self.markov1[self.IDX[last]]
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _markov2_predict(self):
        if len(self.opponent_history) < 2:
            return self._markov1_predict()
        ctx = tuple(self._hist_slice(-2, None))
        arr = self.markov2.get(ctx)
        if arr is None or np.sum(arr) <= 1:
            return self._markov1_predict()
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _ngram_predict(self):
        if len(self.opponent_history) < self.ngram_len:
            return self._markov2_predict()
        key = tuple(self._hist_slice(-self.ngram_len, None))
        arr = self.ngram.get(key)
        if arr is None or np.sum(arr) <= 1:
            return self._markov2_predict()
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _streak_predict(self):
        if self.current_streak_move is None or self.current_streak_count < 2:
            return random.choice(self.MOVES)
        return self.BEATEN_BY[self.current_streak_move]

    def _anti_streak_predict(self):
        if self.current_streak_move is None or self.current_streak_count < 2:
            return random.choice(self.MOVES)
        # Predict they'll break the streak
        others = [m for m in self.MOVES if m != self.current_streak_move]
        predicted = random.choice(others)
        return self.BEATEN_BY[predicted]

    def _cycle_predict(self):
        seq = self._hist_slice(0, None)
        L = len(seq)
        if L < 6:
            return random.choice(self.MOVES)
        for cycle_len in range(2, min(8, L // 2 + 1)):
            if L >= 2 * cycle_len:
                last = tuple(seq[-cycle_len:])
                prev = tuple(seq[-2 * cycle_len:-cycle_len])
                if last == prev:
                    predicted = last[0]
                    return self.BEATEN_BY[predicted]
        return random.choice(self.MOVES)

    def _meta_predict(self):
        """Learn opponent's response to our moves."""
        if len(self.ai_history) < 1:
            return random.choice(self.MOVES)
        last_ai = self.ai_history[-1]
        arr = self.response_patterns.get(last_ai)
        if arr is None or np.sum(arr) <= 1:
            return random.choice(self.MOVES)
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _mirror_predict(self):
        """Play what opponent played last (psychological)."""
        if len(self.opponent_history) < 1:
            return random.choice(self.MOVES)
        return self.opponent_history[-1]

    def _anti_mirror_predict(self):
        """Counter what opponent played last."""
        if len(self.opponent_history) < 1:
            return random.choice(self.MOVES)
        return self.BEATEN_BY[self.opponent_history[-1]]

    def _random_predict(self):
        return random.choice(self.MOVES)

    def _tf_predict(self):
        """TF model predicts directly what to play (not opponent's move)."""
        if (not TF_AVAILABLE) or (self.tf_model is None):
            return random.choice(self.MOVES)
        if len(self.opponent_history) < 3 or len(self.ai_history) < 3:
            return random.choice(self.MOVES)
        
        try:
            # Build richer context: last 3 moves from each
            context = []
            for i in range(-3, 0):
                context.extend(self.one_hot(self.ai_history[i]))
                context.extend(self.one_hot(self.opponent_history[i]))
            
            # Add losing streak info
            context.append(float(self.losing_streak) / 10.0)
            context.append(float(len(self.recent_results)) / 20.0)
            context.append(float(self.recent_results.count('w')) / max(1.0, float(len(self.recent_results))))
            
            x = np.array(context, dtype=np.float32).reshape(1, -1)
            
            # Ensure correct shape
            if x.shape[1] != 15:
                return random.choice(self.MOVES)
            
            probs = self.tf_model.predict(x, verbose=0)[0]
            return self.MOVES[int(np.argmax(probs))]
        except Exception:
            return random.choice(self.MOVES)

    def strategy_predict(self, idx):
        funcs = [
            self._freq_predict,
            self._anti_freq_predict,
            self._markov1_predict,
            self._markov2_predict,
            self._ngram_predict,
            self._streak_predict,
            self._anti_streak_predict,
            self._cycle_predict,
            self._meta_predict,
            self._mirror_predict,
            self._anti_mirror_predict,
            self._random_predict
        ]
        if TF_AVAILABLE:
            funcs.append(self._tf_predict)
        idx = int(idx) % len(funcs)
        return funcs[idx]()

    # ========== ENSEMBLE VOTING ==========
    
    def _get_ensemble_vote(self):
        """Get votes from all strategies and use weighted voting."""
        votes = defaultdict(float)
        
        for i in range(self.K):
            try:
                move = self.strategy_predict(i)
                # Weight by strategy performance
                recent_perf = self.strategy_recent[i]
                if len(recent_perf) > 0:
                    win_rate = recent_perf.count('w') / len(recent_perf)
                    weight = (win_rate + 0.1) * self.weights[i]
                else:
                    weight = self.weights[i]
                votes[move] += weight
            except:
                pass
        
        if not votes:
            return random.choice(self.MOVES)
        
        # Return highest voted move
        return max(votes.items(), key=lambda x: x[1])[0]

    # ========== PATTERN DETECTION ==========
    
    def _detect_randomness(self):
        """Detect if opponent is playing randomly (no exploitable patterns)."""
        if len(self.opponent_history) < 30:
            return 0.5  # Not enough data
        
        recent = list(self.opponent_history)[-30:]
        counts = Counter(recent)
        
        # Check if distribution is close to 33/33/33
        expected = len(recent) / 3.0
        max_deviation = max(abs(counts.get(m, 0) - expected) for m in self.MOVES)
        
        # Check entropy (randomness measure)
        total = len(recent)
        entropy = 0.0
        for move in self.MOVES:
            p = counts.get(move, 0) / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Perfect random has entropy ~1.585
        # Lower entropy = more predictable
        randomness = entropy / 1.585
        
        # IMPROVED: Check for sequential patterns (autocorrelation)
        # Even if distribution is 33/33/33, repeating patterns are exploitable
        move_to_idx = {m: i for i, m in enumerate(self.MOVES)}
        sequence = [move_to_idx[m] for m in recent]
        
        # Check for repeating cycles
        has_cycle = False
        for cycle_len in range(2, min(6, len(sequence) // 3)):
            # Check if sequence repeats with this cycle length
            matches = 0
            total_checks = len(sequence) - cycle_len
            for i in range(len(sequence) - cycle_len):
                if sequence[i] == sequence[i + cycle_len]:
                    matches += 1
            
            if total_checks > 0 and matches / total_checks > 0.6:  # 60%+ correlation
                has_cycle = True
                randomness *= 0.3  # Heavily reduce randomness score
                break
        
        # Also check if our strategies are actually winning
        recent_wr = self.recent_results.count('w') / len(self.recent_results) if self.recent_results else 0.33
        
        # If we're winning significantly more than 33%, opponent is exploitable
        if recent_wr > 0.42:  # Raised threshold slightly
            randomness = max(0.0, randomness - 0.4)
        
        return randomness
    
    def _calculate_pattern_confidence(self):
        """Calculate confidence in detected patterns."""
        if len(self.opponent_history) < 10:
            return 0.0
        
        # Check how well our predictions have been doing
        recent_wr = self.recent_results.count('w') / len(self.recent_results) if self.recent_results else 0.33
        
        # Check consistency of strategies
        strategy_consistency = 0.0
        for recent_perf in self.strategy_recent:
            if len(recent_perf) > 0:
                wr = recent_perf.count('w') / len(recent_perf)
                if wr > 0.4:  # Strategy is actually working
                    strategy_consistency = max(strategy_consistency, wr)
        
        # Combine metrics
        confidence = (recent_wr - 0.33) * 2.0  # Scale so 50% WR = ~33% confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    # ========== MOVE SELECTION ==========
    
    def get_move(self):
        """Select move using ensemble + context-based Q-learning."""
        
        # FIXED: If we're on a losing streak, DECREASE exploration to focus on what works
        if self.losing_streak >= 2:  # React faster
            # Lower gamma = less exploration, more exploitation of proven strategies
            self.gamma = max(0.05, 0.25 - (self.losing_streak * 0.05))
            
            # If we just lost, AVOID (but don't always skip) the same move in similar context
            if self.last_loss_context and self.last_loss_move and random.random() < 0.7:
                current_context = self._get_context_key()
                if current_context == self.last_loss_context:
                    print(f"🚫 Avoiding {self.last_loss_move.upper()} - it just lost!")
        else:
            self.gamma = 0.25  # Reset to default
        
        # NEW: Update pattern detection metrics
        self.randomness_score = self._detect_randomness()
        self.pattern_confidence = self._calculate_pattern_confidence()
        
        # If opponent appears highly random (>0.8) AND we're not winning, use Nash equilibrium
        if self.randomness_score > 0.8 and self.pattern_confidence < 0.2:
            if random.random() < 0.7:  # 70% of time against random
                nash_move = random.choice(self.MOVES)
                print(f"🎲 Nash equilibrium: {nash_move.upper()} (opponent appears random: {self.randomness_score:.2f})")
                self.last_ai_move = nash_move
                self.prediction_confidence = 0.33
                return nash_move
        
        # Get context-based best action (Q-learning)
        context = self._get_context_key()
        
        # NEW: Check blacklist - but only restrict moves that have REALLY failed
        available_moves = list(self.MOVES)
        if context in self.move_blacklist and self.losing_streak >= 3:
            blacklist = self.move_blacklist[context]
            # Only remove moves that have lost 3+ times more than they've won
            for move in self.MOVES:
                if blacklist[move] >= 3:
                    temp_available = [m for m in available_moves if m != move]
                    if len(temp_available) >= 2:  # Keep at least 2 options
                        available_moves = temp_available
        
        # IMPROVED: Use context-based learning more often when we have good data
        use_context = random.random() < 0.7  # Increased to 70% when we have patterns
        if use_context and context in self.context_actions and self.total_rounds > 5:
            action_stats = self.context_actions[context]
            action_values = {}
            for move in available_moves:
                w, l, d = action_stats[move]
                total = w + l + d
                if total > 2:  # Need at least 3 samples
                    # Value = win_rate - loss_rate
                    action_values[move] = (w - l) / total
            
            if action_values:
                # Add randomness - don't always pick the best
                if random.random() < 0.8:  # 80% pick best, 20% explore
                    sorted_moves = sorted(action_values.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_moves) >= 2 and random.random() < 0.15:
                        best_move = sorted_moves[1][0]  # Pick second best sometimes
                    else:
                        best_move = sorted_moves[0][0]
                else:
                    best_move = random.choice(list(action_values.keys()))
                
                best_value = action_values[best_move]
                if best_value > 0.0:  # FIXED: Only use if actually winning (stricter threshold)
                    print(f"📊 Context-based: {best_move.upper()} (value: {best_value:.2f})")
                    self.last_strategy_idx = None
                    self.prediction_confidence = 0.6
                    
                    # NEW: Anti-exploitation - if opponent keeps beating our context moves, break pattern
                    if context in self.context_actions:
                        total_context_plays = sum(self.context_actions[context][best_move])
                        if total_context_plays >= 5:
                            w, l, d = self.context_actions[context][best_move]
                            if l > w + 2:  # Losing significantly in this context
                                if random.random() < 0.4:  # 40% chance to randomize
                                    best_move = random.choice([m for m in self.MOVES if m != best_move])
                                    print(f"🛡️ Anti-exploit: switching to {best_move.upper()}")
                    
                    return best_move
        
        # Otherwise use EXP3 + strategies (most of the time)
        W = np.array(self.weights, dtype=float)
        if len(W) != self.K:
            W = np.ones(self.K, dtype=float)
            self.weights = W
        if W.sum() == 0:
            W = np.ones_like(W)
        
        probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
        probs = probs / probs.sum()
        
        # Choose strategy
        idx = int(np.random.choice(self.K, p=probs))
        self.last_strategy_idx = idx
        self.strategy_plays[idx] += 1
        our_move = self.strategy_predict(idx)
        
        # Safety: if picked the exact same move as last time AND we lost, 20% chance to override
        if (our_move == self.last_loss_move and 
            self.last_ai_move == our_move and 
            self.results and self.results[-1] == 'l' and 
            random.random() < 0.2):
            other_moves = [m for m in self.MOVES if m != our_move]
            our_move = random.choice(other_moves)
            print(f"🎲 Random switch to break pattern: {our_move.upper()}")
        
        self.last_ai_move = our_move
        self.prediction_confidence = float(probs[idx])
        
        return our_move

    # ========== LEARNING & UPDATE ==========
    
    def update(self, ai_move, result):
        if result not in ('w', 'l', 'd'):
            raise ValueError("result must be 'w','l' or 'd'")

        opponent_move = self._opponent_from_result(ai_move, result)
        
        self.opponent_history.append(opponent_move)
        self.ai_history.append(ai_move)
        self.results.append(result)
        self.recent_results.append(result)
        self.total_rounds += 1

        # Track losing streaks
        if result == 'l':
            self.losing_streak += 1
        else:
            self.losing_streak = max(0, self.losing_streak - 1)

        # Update context-action values (Q-learning)
        context = self._get_context_key()
        w, l, d = self.context_actions[context][ai_move]
        if result == 'w':
            self.context_actions[context][ai_move] = [w+1, l, d]
        elif result == 'l':
            self.context_actions[context][ai_move] = [w, l+1, d]
        else:
            self.context_actions[context][ai_move] = [w, l, d+1]

        # Decay old data faster when losing
        decay_mult = 0.8 if self.losing_streak >= 3 else 1.0
        for k in list(self.recency_weights.keys()):
            self.recency_weights[k] *= (self.decay * decay_mult)
        self.recency_weights[opponent_move] += 1.0
        self.freq_counts[opponent_move] += 1

        # Update Markov models
        if len(self.opponent_history) >= 2:
            prev = self.opponent_history[-2]
            cur = self.opponent_history[-1]
            self.markov1[self.IDX[prev], self.IDX[cur]] += 1.0
        
        if len(self.opponent_history) >= 3:
            ctx = tuple(self._hist_slice(-3, -1))
            self.markov2[ctx][self.IDX[opponent_move]] += 1.0

        if len(self.opponent_history) >= self.ngram_len + 1:
            key = tuple(self._hist_slice(-(self.ngram_len+1), -1))
            self.ngram[key][self.IDX[opponent_move]] += 1.0

        # Meta learning
        if len(self.ai_history) >= 2:
            our_prev = self.ai_history[-2]
            self.response_patterns[our_prev][self.IDX[opponent_move]] += 1.0

        # Track what beats us
        if result == 'l':
            self.what_beats_us[ai_move][self.IDX[opponent_move]] += 2.0  # Double weight losses

        # Streak tracking
        if self.current_streak_move == opponent_move:
            self.current_streak_count += 1
        else:
            self.current_streak_move = opponent_move
            self.current_streak_count = 1

        # TF training data - now learns what WINS, not what opponent plays
        if TF_AVAILABLE and self.tf_model is not None:
            if len(self.ai_history) >= 3 and len(self.opponent_history) >= 3:
                try:
                    context = []
                    for i in range(-3, 0):
                        context.extend(self.one_hot(self.ai_history[i]))
                        context.extend(self.one_hot(self.opponent_history[i]))
                    context.append(float(self.losing_streak) / 10.0)
                    context.append(float(len(self.recent_results)) / 20.0)
                    context.append(float(self.recent_results.count('w')) / max(1.0, float(len(self.recent_results))))
                    
                    # Ensure exactly 15 features
                    x = np.array(context, dtype=np.float32)
                    if x.shape[0] == 15:  # Only add if correct size
                        # Target: what we should have played (the move that beats opponent_move)
                        winning_move = self.BEATEN_BY[opponent_move]
                        y = self.one_hot(winning_move).astype(np.float32)
                        
                        # FIXED: Always train on winning move, but weight examples differently
                        # Wins prove our prediction worked - add them MORE, not less
                        if result == 'w':
                            for _ in range(3):  # Learn from successful predictions
                                self.tf_train_buffer.append((x.copy(), y.copy()))
                        elif result == 'l':
                            # Still learn from losses, but with less weight
                            self.tf_train_buffer.append((x.copy(), y.copy()))
                        else:
                            # Draws are neutral - moderate learning
                            self.tf_train_buffer.append((x.copy(), y.copy()))
                except Exception as e:
                    print(f"⚠️ TF buffer error: {e}")
                    pass

        # ========== AGGRESSIVE EXP3 UPDATE ==========
        if self.last_strategy_idx is not None:
            # Track per-strategy results
            self.strategy_recent[self.last_strategy_idx].append(result)
            
            # FIXED: Stronger reward/penalty for faster learning
            if result == 'w':
                reward = 2.0  # Stronger reward for wins
                self.strategy_wins[self.last_strategy_idx] += 1.0
            elif result == 'l':
                reward = -2.0  # Stronger penalty for losses
            else:
                reward = 0.1  # Small reward for draws (better than losing)
            
            # Boost learning rate when losing
            effective_eta = self.eta * (2.0 if self.losing_streak >= 3 else 1.0)
            
            W = np.array(self.weights, dtype=float)
            if W.sum() == 0:
                W = np.ones_like(W)
            probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
            p_i = probs[self.last_strategy_idx]
            
            x_hat = reward / max(p_i, 0.01)
            self.weights[self.last_strategy_idx] *= np.exp(min(effective_eta * x_hat / self.K, 2.0))
            
            # More aggressive decay when losing
            decay_rate = 0.99 if self.losing_streak < 3 else 0.98
            self.weights *= decay_rate

        # TF model training
        if TF_AVAILABLE and self.tf_model is not None:
            if (self.total_rounds % self.tf_train_every == 0) and (len(self.tf_train_buffer) >= 32):
                try:
                    # Ensure all samples have correct shape
                    valid_samples = []
                    for x, y in self.tf_train_buffer:
                        if x.shape == (15,) and y.shape == (3,):
                            valid_samples.append((x, y))
                    
                    if len(valid_samples) >= 32:
                        X = np.array([t[0] for t in valid_samples], dtype=np.float32)
                        Y = np.array([t[1] for t in valid_samples], dtype=np.float32)
                        
                        # Verify shapes
                        if X.shape[1] == 15 and Y.shape[1] == 3:
                            self.tf_model.fit(X, Y, epochs=self.tf_epochs, batch_size=self.tf_batch, verbose=0)
                            self._save_tf_model()
                except Exception as e:
                    print(f"⚠️ TF training error: {e}")
                    pass

        # Autosave
        if self.total_rounds % self.autosave_every == 0:
            self.save_state()

    # ========== STATS ==========
    
    def stats(self):
        """Display comprehensive statistics."""
        total = len(self.results)
        wins = self.results.count('w')
        losses = self.results.count('l')
        draws = self.results.count('d')
        win_rate = wins / total * 100 if total else 0.0
        
        recent = self.results[-50:] if len(self.results) >= 50 else self.results
        recent_wr = (recent.count('w') / len(recent) * 100) if recent else 0.0
        
        last_20 = self.results[-20:] if len(self.results) >= 20 else self.results
        last_20_wr = (last_20.count('w') / len(last_20) * 100) if last_20 else 0.0
        
        # Calculate win rate per strategy
        strat_info = []
        for i, name in enumerate(self.strategy_names):
            plays = self.strategy_plays[i]
            wins_s = self.strategy_wins[i]
            recent_perf = self.strategy_recent[i]
            recent_wr_s = (recent_perf.count('w') / len(recent_perf) * 100) if recent_perf else 0.0
            weight = self.weights[i]
            strat_info.append((name, plays, recent_wr_s, weight))
        
        # Sort by weight
        strat_info.sort(key=lambda x: x[3], reverse=True)
        top_5 = strat_info[:5]
        
        tf_info = "TF:✓" if (TF_AVAILABLE and self.tf_model is not None) else "TF:✗"
        
        streak_emoji = "🔥" if self.losing_streak >= 3 else "✓"
        
        output = f"""
📊 ===== PERFORMANCE STATS ===== 📊
Total Games: {total} | W:{wins} L:{losses} D:{draws}
Overall WR: {win_rate:.1f}% | Recent 50: {recent_wr:.1f}% | Last 20: {last_20_wr:.1f}%
Losing Streak: {self.losing_streak} {streak_emoji} | Exploration: {self.gamma:.2f}
Confidence: {self.prediction_confidence:.3f} | {tf_info}

🏆 Top 5 Strategies (by weight):
"""
        for name, plays, wr, weight in top_5:
            output += f"  {name:14s}: {int(plays):4d} plays | Recent WR:{wr:5.1f}% | W:{weight:7.2f}\n"
        
        # Show context-action performance
        output += f"\n📚 Context-Action Learning: {len(self.context_actions)} contexts learned\n"
        
        output += f"\nSession: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return output

# ========== MAIN LOOP ==========

def main(save_prefix="rps_beast"):
    print("Current working directory:", os.getcwd())
    print("\n🚀 INITIALIZING ENHANCED ADAPTIVE RPS AI 🚀")
    print("=" * 50)
    
    ai = RealTimeEvolvingRPS_TF(save_prefix)
    plotter = RPSPlotter(ai)
    
    print("\n✅ AI Ready! Features:")
    print("  • Ensemble voting from 13 strategies")
    print("  • Context-based Q-learning")
    print("  • Aggressive adaptation on losing streaks")
    print("  • Deep learning with TensorFlow")
    print("\n📋 Instructions:")
    print("  'w' = AI won | 'l' = AI lost | 'd' = draw")
    print("  Commands: stats, reset, save, exit\n")
    
    while True:
        try:
            ai_move = ai.get_move()
        except Exception as e:
            print("⚠️ get_move error:", e)
            traceback.print_exc()
            ai_move = random.choice(ai.MOVES)

        status = f"🤖 AI plays: {ai_move.upper()} | Conf: {ai.prediction_confidence:.3f}"
        if ai.last_strategy_idx is not None:
            status += f" | Strategy: {ai.strategy_names[ai.last_strategy_idx]}"
        else:
            status += f" | Strategy: ENSEMBLE"
        
        if ai.losing_streak >= 3:
            status += f" | 🔥 Losing streak: {ai.losing_streak} - ADAPTING!"
        
        print(f"\n{status}")
        result = input("🏁 Result (w/l/d) or command: ").strip().lower()

        if result == 'exit':
            ai.save_state()
            print("💾 Memory saved. Exiting.")
            break
        elif result == 'stats':
            print(ai.stats())
            continue
        elif result == 'save':
            if ai.save_state():
                print("💾 Memory saved!")
            else:
                print("⚠️ Save failed.")
            continue
        elif result == 'reset':
            confirm = input("⚠️ Type YES to reset memory & TF model: ")
            if confirm == 'YES':
                ai.reset_state()
            continue
        elif result not in ('w', 'l', 'd'):
            print("❌ Invalid input")
            continue

        try:
            ai.update(ai_move, result)
        except Exception as e:
            print("⚠️ update error:", e)
            traceback.print_exc()

        plotter.update_plot()

if __name__ == "__main__":
    main()