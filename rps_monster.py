import os
import traceback
import random
import pickle
from collections import deque, defaultdict, Counter
from datetime import datetime
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

class MonsterRPS:
    """
    An absolute MONSTER Rock-Paper-Scissors AI that learns aggressively,
    detects patterns, models opponents, and exploits weaknesses relentlessly.
    """
    MOVES = ['rock', 'paper', 'scissors']
    IDX = {m: i for i, m in enumerate(MOVES)}
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    BEATEN_BY = {v: k for k, v in BEATS.items()}

    def __init__(self,
                 save_prefix="rps_monster",
                 memory_size=10000,
                 seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.save_prefix = save_prefix
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(base_dir, f"{save_prefix}_state.pkl")
        self.tf_model_dir = os.path.join(base_dir, f"{save_prefix}_tf_model")
        self.memory_size = memory_size

        # ===== CORE HISTORIES =====
        self.opponent_history = deque(maxlen=memory_size)
        self.ai_history = deque(maxlen=memory_size)
        self.results = deque(maxlen=memory_size)
        
        # ===== PATTERN DETECTION =====
        # Multi-scale pattern detection (2-10 move sequences)
        self.patterns = {}  # pattern -> {count, last_seen, confidence}
        self.pattern_predictions = defaultdict(lambda: np.ones(3) * 0.1)  # pattern -> next move probs
        
        # Cycle detection with phase tracking
        self.detected_cycles = []  # [(cycle_pattern, strength, phase)]
        
        # Frequency with multiple time windows
        self.freq_windows = {
            10: Counter(),
            20: Counter(),
            50: Counter(),
            100: Counter(),
            'all': Counter()
        }
        
        # ===== OPPONENT MODELING =====
        self.opponent_archetype = "unknown"  # random, pattern, counter, adaptive, mixed
        self.archetype_confidence = 0.0
        self.archetype_history = deque(maxlen=100)
        self.counter_predict_evidence = 0  # Tracks if opponent is counter-predicting us
        
        # ===== ENHANCED MARKOV CHAINS =====
        # Deep Markov chains (1st through 5th order)
        self.markov_chains = {
            1: defaultdict(lambda: np.ones(3) * 0.1),
            2: defaultdict(lambda: np.ones(3) * 0.1),
            3: defaultdict(lambda: np.ones(3) * 0.1),
            4: defaultdict(lambda: np.ones(3) * 0.1),
            5: defaultdict(lambda: np.ones(3) * 0.1),
        }
        
        # ===== STRATEGIES =====
        self.strategy_names = [
            'freq_recent', 'freq_all', 'anti_freq',
            'markov1', 'markov2', 'markov3', 'markov4', 'markov5',
            'pattern_2', 'pattern_3', 'pattern_4', 'pattern_5',
            'cycle', 'meta', 'anti_mirror', 'counter_counter',
            'exploit', 'random', 'nash'
        ]
        if TF_AVAILABLE:
            self.strategy_names.extend(['tf_main', 'tf_pattern', 'tf_confidence'])
        
        self.K = len(self.strategy_names)
        
        # EXP3 weights - AGGRESSIVE
        self.weights = np.ones(self.K, dtype=float)
        self.gamma = 0.15  # Start with moderate exploration
        self.base_gamma = 0.15
        self.eta = 0.2  # Aggressive learning rate
        
        # Per-strategy tracking
        self.strategy_wins = np.zeros(self.K)
        self.strategy_losses = np.zeros(self.K)
        self.strategy_plays = np.zeros(self.K)
        self.strategy_recent = [deque(maxlen=20) for _ in range(self.K)]
        
        # ===== LOSS RECOVERY =====
        self.losing_streak = 0
        self.recent_losses = deque(maxlen=5)  # Last 5 losses with context
        self.failed_predictions = defaultdict(int)  # Track what didn't work
        
        # ===== EXPLOITATION MODE =====
        self.exploitation_mode = False
        self.exploitation_strategy = None
        self.exploitation_streak = 0
        
        # ===== COUNTER-INTELLIGENCE =====
        self.being_exploited = False
        self.exploitation_counter = 0
        self.deception_mode = False
        self.last_deception = None
        
        # ===== META-LEARNING =====
        self.response_patterns = defaultdict(lambda: np.ones(3) * 0.1)  # our move -> their response
        self.what_beats_us = defaultdict(lambda: np.ones(3) * 0.1)
        self.what_we_beat = defaultdict(lambda: np.ones(3) * 0.1)
        
        # ===== CONTEXT-BASED Q-LEARNING =====
        self.context_values = defaultdict(lambda: {m: [0, 0, 0] for m in self.MOVES})  # W, L, D
        
        # ===== TENSORFLOW MODELS =====
        self.tf_models = {}
        if TF_AVAILABLE:
            self._init_tf_models()
        
        # Training buffer with prioritized replay
        self.experience_buffer = deque(maxlen=5000)
        self.priority_buffer = deque(maxlen=1000)  # High-priority experiences
        
        # ===== TRACKING =====
        self.total_rounds = 0
        self.last_ai_move = None
        self.last_strategy_idx = None
        self.prediction_confidence = 0.0
        self.recent_win_rate = 0.33
        
        # Load existing state if available
        self.load_state()

    def _init_tf_models(self):
        """Initialize multiple TensorFlow models for different purposes."""
        try:
            # Main prediction model - deep and powerful
            if os.path.isdir(f"{self.tf_model_dir}_main"):
                self.tf_models['main'] = keras.models.load_model(f"{self.tf_model_dir}_main")
            else:
                self.tf_models['main'] = self._build_main_tf_model()
            
            # Pattern specialist - focuses on sequential patterns
            if os.path.isdir(f"{self.tf_model_dir}_pattern"):
                self.tf_models['pattern'] = keras.models.load_model(f"{self.tf_model_dir}_pattern")
            else:
                self.tf_models['pattern'] = self._build_pattern_tf_model()
            
            # Confidence estimator - predicts how certain we should be
            if os.path.isdir(f"{self.tf_model_dir}_confidence"):
                self.tf_models['confidence'] = keras.models.load_model(f"{self.tf_model_dir}_confidence")
            else:
                self.tf_models['confidence'] = self._build_confidence_tf_model()
        except Exception as e:
            print(f"⚠️ TF initialization error: {e}")
            self.tf_models = {}

    def _build_main_tf_model(self):
        """Build a DEEP, POWERFUL main prediction model."""
        model = keras.Sequential([
            layers.Input(shape=(30,)),  # Rich feature vector
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _build_pattern_tf_model(self):
        """Build a model specialized for sequential pattern detection."""
        model = keras.Sequential([
            layers.Input(shape=(20,)),  # Sequence-focused features
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.003),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _build_confidence_tf_model(self):
        """Build a model that estimates prediction confidence."""
        model = keras.Sequential([
            layers.Input(shape=(15,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output: confidence 0-1
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def save_state(self):
        """Save all state to disk."""
        try:
            payload = {
                'opponent_history': list(self.opponent_history),
                'ai_history': list(self.ai_history),
                'results': list(self.results),
                'weights': self.weights.tolist(),
                'strategy_wins': self.strategy_wins.tolist(),
                'strategy_losses': self.strategy_losses.tolist(),
                'strategy_plays': self.strategy_plays.tolist(),
                'total_rounds': self.total_rounds,
                'patterns': self.patterns,
                'freq_windows': {k: dict(v) for k, v in self.freq_windows.items()},
                'markov_chains': {k: {str(key): val.tolist() for key, val in v.items()} 
                                 for k, v in self.markov_chains.items()},
                'opponent_archetype': self.opponent_archetype,
                'archetype_confidence': self.archetype_confidence,
                'context_values': {str(k): v for k, v in self.context_values.items()},
                'response_patterns': {k: v.tolist() for k, v in self.response_patterns.items()},
                'what_beats_us': {k: v.tolist() for k, v in self.what_beats_us.items()},
                'what_we_beat': {k: v.tolist() for k, v in self.what_we_beat.items()},
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(payload, f)
            
            # Save TF models
            if TF_AVAILABLE and self.tf_models:
                for name, model in self.tf_models.items():
                    try:
                        model.save(f"{self.tf_model_dir}_{name}", overwrite=True)
                    except:
                        pass
            
            print(f"💾 Monster state saved!")
            return True
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            return False

    def load_state(self):
        """Load state from disk."""
        if not os.path.exists(self.state_file):
            return False
        try:
            with open(self.state_file, 'rb') as f:
                payload = pickle.load(f)
            
            self.opponent_history = deque(payload.get('opponent_history', []), maxlen=self.memory_size)
            self.ai_history = deque(payload.get('ai_history', []), maxlen=self.memory_size)
            self.results = deque(payload.get('results', []), maxlen=self.memory_size)
            
            loaded_weights = np.array(payload.get('weights', []))
            if len(loaded_weights) == self.K:
                self.weights = loaded_weights
                self.strategy_wins = np.array(payload.get('strategy_wins', np.zeros(self.K)))
                self.strategy_losses = np.array(payload.get('strategy_losses', np.zeros(self.K)))
                self.strategy_plays = np.array(payload.get('strategy_plays', np.zeros(self.K)))
            
            self.total_rounds = payload.get('total_rounds', 0)
            self.patterns = payload.get('patterns', {})
            self.freq_windows = defaultdict(Counter)
            for k, v in payload.get('freq_windows', {}).items():
                self.freq_windows[k] = Counter(v)
            
            # Restore Markov chains
            for order in range(1, 6):
                if order in payload.get('markov_chains', {}):
                    for key_str, val in payload['markov_chains'][order].items():
                        try:
                            key = eval(key_str)
                            self.markov_chains[order][key] = np.array(val)
                        except:
                            pass
            
            self.opponent_archetype = payload.get('opponent_archetype', 'unknown')
            self.archetype_confidence = payload.get('archetype_confidence', 0.0)
            
            # Restore context values
            for key_str, val in payload.get('context_values', {}).items():
                try:
                    key = eval(key_str)
                    self.context_values[key] = val
                except:
                    pass
            
            # Restore meta-learning
            for k, v in payload.get('response_patterns', {}).items():
                self.response_patterns[k] = np.array(v)
            for k, v in payload.get('what_beats_us', {}).items():
                self.what_beats_us[k] = np.array(v)
            for k, v in payload.get('what_we_beat', {}).items():
                self.what_we_beat[k] = np.array(v)
            
            print(f"📂 Monster state loaded! {self.total_rounds} rounds of experience.")
            return True
        except Exception as e:
            print(f"⚠️ Load error: {e}")
            return False

    # ========== PATTERN DETECTION ==========
    
    def _detect_patterns(self):
        """Advanced multi-scale pattern detection."""
        if len(self.opponent_history) < 6:
            return
        
        hist = list(self.opponent_history)
        
        # Detect patterns of length 2-10
        for length in range(2, min(11, len(hist) // 3)):
            pattern = tuple(hist[-length:])
            
            # Count occurrences of this pattern in history
            count = 0
            for i in range(len(hist) - length):
                if tuple(hist[i:i+length]) == pattern:
                    count += 1
            
            if count >= 2:  # Pattern appears at least twice
                confidence = min(1.0, count / 10.0)
                self.patterns[pattern] = {
                    'count': count,
                    'last_seen': self.total_rounds,
                    'confidence': confidence
                }
                
                # Track what usually comes after this pattern
                for i in range(len(hist) - length - 1):
                    if tuple(hist[i:i+length]) == pattern:
                        next_move = hist[i+length]
                        self.pattern_predictions[pattern][self.IDX[next_move]] += 1

    def _detect_cycles(self):
        """Detect repeating cycles with phase tracking."""
        if len(self.opponent_history) < 12:
            return []
        
        hist = list(self.opponent_history)
        detected = []
        
        for cycle_len in range(2, min(10, len(hist) // 3)):
            matches = 0
            comparisons = 0
            
            for i in range(len(hist) - cycle_len):
                if hist[i] == hist[i + cycle_len]:
                    matches += 1
                comparisons += 1
            
            if comparisons > 0:
                strength = matches / comparisons
                if strength > 0.6:  # 60%+ match rate
                    phase = len(hist) % cycle_len
                    detected.append((cycle_len, strength, phase))
        
        return detected

    def _analyze_opponent_archetype(self):
        """Determine what type of opponent we're facing."""
        if len(self.opponent_history) < 20:
            return
        
        hist = list(self.opponent_history)[-50:]  # Analyze recent history
        
        # Check randomness (uniform distribution)
        counts = Counter(hist)
        total = len(hist)
        expected = total / 3.0
        max_dev = max(abs(counts.get(m, 0) - expected) for m in self.MOVES)
        randomness_score = 1.0 - (max_dev / expected)
        
        # Check for patterns
        pattern_score = len([p for p in self.patterns.values() if p['confidence'] > 0.3]) / 5.0
        pattern_score = min(1.0, pattern_score)
        
        # Check if opponent counter-predicts us
        counter_evidence = 0
        if len(self.ai_history) >= 3 and len(self.opponent_history) >= 2:
            for i in range(-min(10, len(self.ai_history)-1), -1):
                if i < -1:
                    # Does opponent play what beats our previous move?
                    if self.opponent_history[i] == self.BEATEN_BY[self.ai_history[i-1]]:
                        counter_evidence += 1
        
        counter_score = min(1.0, counter_evidence / 5.0)
        
        # Classify
        if randomness_score > 0.7 and pattern_score < 0.3:
            self.opponent_archetype = "random"
            self.archetype_confidence = randomness_score
        elif pattern_score > 0.5:
            self.opponent_archetype = "pattern"
            self.archetype_confidence = pattern_score
        elif counter_score > 0.5:
            self.opponent_archetype = "counter"
            self.archetype_confidence = counter_score
        elif randomness_score < 0.5 and pattern_score > 0.2:
            self.opponent_archetype = "mixed"
            self.archetype_confidence = 0.5
        else:
            self.opponent_archetype = "adaptive"
            self.archetype_confidence = 0.4

    # ========== PREDICTION STRATEGIES ==========
    
    def _predict_frequency_recent(self):
        """Counter most common recent move."""
        if len(self.opponent_history) < 3:
            return random.choice(self.MOVES)
        recent = list(self.opponent_history)[-20:]
        most_common = Counter(recent).most_common(1)[0][0]
        return self.BEATEN_BY[most_common]

    def _predict_frequency_all(self):
        """Counter most common overall move."""
        if not self.freq_windows['all']:
            return random.choice(self.MOVES)
        most_common = self.freq_windows['all'].most_common(1)[0][0]
        return self.BEATEN_BY[most_common]

    def _predict_anti_frequency(self):
        """Counter least common move."""
        if not self.freq_windows['all']:
            return random.choice(self.MOVES)
        least_common = self.freq_windows['all'].most_common()[-1][0]
        return self.BEATEN_BY[least_common]

    def _predict_markov(self, order):
        """Predict using Markov chain of given order."""
        if len(self.opponent_history) < order:
            return random.choice(self.MOVES)
        
        context = tuple(list(self.opponent_history)[-order:])
        probs = self.markov_chains[order].get(context)
        
        if probs is None or np.sum(probs) <= 0.5:
            if order > 1:
                return self._predict_markov(order - 1)
            return random.choice(self.MOVES)
        
        probs = probs / np.sum(probs)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _predict_pattern(self, length):
        """Predict using detected patterns of specific length."""
        if len(self.opponent_history) < length:
            return random.choice(self.MOVES)
        
        current_pattern = tuple(list(self.opponent_history)[-length:])
        
        if current_pattern in self.pattern_predictions:
            probs = self.pattern_predictions[current_pattern]
            if np.sum(probs) > 1:
                probs = probs / np.sum(probs)
                predicted = self.MOVES[int(np.argmax(probs))]
                return self.BEATEN_BY[predicted]
        
        return random.choice(self.MOVES)

    def _predict_cycle(self):
        """Predict based on detected cycles."""
        cycles = self._detect_cycles()
        if not cycles:
            return random.choice(self.MOVES)
        
        # Use strongest cycle
        best_cycle = max(cycles, key=lambda x: x[1])
        cycle_len, strength, phase = best_cycle
        
        if len(self.opponent_history) >= cycle_len:
            predicted = list(self.opponent_history)[-cycle_len]
            return self.BEATEN_BY[predicted]
        
        return random.choice(self.MOVES)

    def _predict_meta(self):
        """Predict based on opponent's response to our moves."""
        if len(self.ai_history) < 1:
            return random.choice(self.MOVES)
        
        last_ai = self.ai_history[-1]
        probs = self.response_patterns.get(last_ai)
        
        if probs is None or np.sum(probs) <= 0.5:
            return random.choice(self.MOVES)
        
        probs = probs / np.sum(probs)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _predict_anti_mirror(self):
        """Counter what opponent played last."""
        if len(self.opponent_history) < 1:
            return random.choice(self.MOVES)
        return self.BEATEN_BY[self.opponent_history[-1]]

    def _predict_counter_counter(self):
        """If opponent is counter-predicting us, confuse them."""
        if len(self.ai_history) < 2:
            return random.choice(self.MOVES)
        
        # Play something unexpected
        last_two = [self.ai_history[-2], self.ai_history[-1]]
        # Avoid what we played recently
        candidates = [m for m in self.MOVES if m not in last_two]
        if not candidates:
            candidates = self.MOVES
        return random.choice(candidates)

    def _predict_exploit(self):
        """AGGRESSIVE exploitation of detected weaknesses."""
        if len(self.opponent_history) < 10:
            return random.choice(self.MOVES)
        
        # What have we been beating them with?
        recent_wins = []
        for i in range(-min(20, len(self.results)), 0):
            if self.results[i] == 'w' and abs(i) <= len(self.ai_history):
                recent_wins.append(self.ai_history[i])
        
        if recent_wins:
            # Play what's been winning
            most_winning = Counter(recent_wins).most_common(1)[0][0]
            return most_winning
        
        return random.choice(self.MOVES)

    def _predict_nash(self):
        """Nash equilibrium - pure random."""
        return random.choice(self.MOVES)

    def _predict_tf_main(self):
        """Main TF model prediction."""
        if 'main' not in self.tf_models or len(self.opponent_history) < 5:
            return random.choice(self.MOVES)
        
        try:
            features = self._build_tf_features_main()
            if features is None:
                return random.choice(self.MOVES)
            
            x = np.array(features).reshape(1, -1)
            probs = self.tf_models['main'].predict(x, verbose=0)[0]
            return self.MOVES[int(np.argmax(probs))]
        except:
            return random.choice(self.MOVES)

    def _predict_tf_pattern(self):
        """Pattern-specialized TF model prediction."""
        if 'pattern' not in self.tf_models or len(self.opponent_history) < 5:
            return random.choice(self.MOVES)
        
        try:
            features = self._build_tf_features_pattern()
            if features is None:
                return random.choice(self.MOVES)
            
            x = np.array(features).reshape(1, -1)
            probs = self.tf_models['pattern'].predict(x, verbose=0)[0]
            return self.MOVES[int(np.argmax(probs))]
        except:
            return random.choice(self.MOVES)

    def _predict_tf_confidence(self):
        """Use confidence model to pick best strategy."""
        # This is more of a meta-strategy - use the confidence model to weigh others
        return self._predict_tf_main()  # For now, default to main

    def _build_tf_features_main(self):
        """Build rich feature vector for main TF model."""
        if len(self.opponent_history) < 5 or len(self.ai_history) < 5:
            return None
        
        features = []
        
        # Last 5 moves from each player (one-hot encoded)
        for i in range(-5, 0):
            features.extend(self._one_hot(self.opponent_history[i]))
            features.extend(self._one_hot(self.ai_history[i]))
        
        # Win rate
        recent_results = list(self.results)[-20:]
        if recent_results:
            features.append(recent_results.count('w') / len(recent_results))
        else:
            features.append(0.33)
        
        # Losing streak indicator
        features.append(min(1.0, self.losing_streak / 5.0))
        
        # Archetype confidence
        features.append(self.archetype_confidence)
        
        # Pattern confidence
        if self.patterns:
            max_pattern_conf = max(p['confidence'] for p in self.patterns.values())
            features.append(max_pattern_conf)
        else:
            features.append(0.0)
        
        # Exploration parameter
        features.append(self.gamma)
        
        return features

    def _build_tf_features_pattern(self):
        """Build feature vector focused on sequential patterns."""
        if len(self.opponent_history) < 5:
            return None
        
        features = []
        
        # Last 5 opponent moves (one-hot)
        for i in range(-5, 0):
            features.extend(self._one_hot(self.opponent_history[i]))
        
        # Cycle indicators
        cycles = self._detect_cycles()
        if cycles:
            best_cycle = max(cycles, key=lambda x: x[1])
            features.append(best_cycle[0] / 10.0)  # Cycle length normalized
            features.append(best_cycle[1])  # Strength
            features.append(best_cycle[2] / 10.0)  # Phase normalized
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Pattern count
        features.append(min(1.0, len(self.patterns) / 10.0))
        
        # Archetype indicator
        archetype_encoding = {
            'random': [1, 0, 0, 0],
            'pattern': [0, 1, 0, 0],
            'counter': [0, 0, 1, 0],
            'mixed': [0, 0, 0, 1],
            'adaptive': [0.25, 0.25, 0.25, 0.25],
            'unknown': [0, 0, 0, 0]
        }
        features.extend(archetype_encoding.get(self.opponent_archetype, [0, 0, 0, 0]))
        
        return features

    def _one_hot(self, move):
        """Convert move to one-hot encoding."""
        vec = [0.0, 0.0, 0.0]
        if move in self.MOVES:
            vec[self.IDX[move]] = 1.0
        return vec

    def strategy_predict(self, idx):
        """Execute a specific strategy by index."""
        strategy_map = {
            0: self._predict_frequency_recent,
            1: self._predict_frequency_all,
            2: self._predict_anti_frequency,
            3: lambda: self._predict_markov(1),
            4: lambda: self._predict_markov(2),
            5: lambda: self._predict_markov(3),
            6: lambda: self._predict_markov(4),
            7: lambda: self._predict_markov(5),
            8: lambda: self._predict_pattern(2),
            9: lambda: self._predict_pattern(3),
            10: lambda: self._predict_pattern(4),
            11: lambda: self._predict_pattern(5),
            12: self._predict_cycle,
            13: self._predict_meta,
            14: self._predict_anti_mirror,
            15: self._predict_counter_counter,
            16: self._predict_exploit,
            17: self._predict_nash,
            18: self._predict_nash,  # Default for out of range
        }
        
        tf_offset = 19
        if TF_AVAILABLE and idx >= tf_offset:
            tf_idx = idx - tf_offset
            if tf_idx == 0:
                return self._predict_tf_main()
            elif tf_idx == 1:
                return self._predict_tf_pattern()
            elif tf_idx == 2:
                return self._predict_tf_confidence()
        
        func = strategy_map.get(idx, strategy_map[18])
        return func()

    # ========== MOVE SELECTION ==========
    
    def get_move(self):
        """SELECT THE OPTIMAL MOVE - THIS IS WHERE THE MAGIC HAPPENS."""
        
        # Update analytics
        self._detect_patterns()
        self._analyze_opponent_archetype()
        
        # Calculate recent win rate
        if len(self.results) >= 10:
            recent = list(self.results)[-20:]
            self.recent_win_rate = recent.count('w') / len(recent)
        
        # ===== EXPLOITATION MODE =====
        # If we're winning big, EXPLOIT IT HARD
        if self.recent_win_rate > 0.55 and not self.being_exploited:
            if not self.exploitation_mode:
                print("🔥 EXPLOITATION MODE ACTIVATED! 🔥")
                self.exploitation_mode = True
                
                # Find best strategy
                best_idx = None
                best_wr = 0.0
                for i in range(self.K):
                    recent = self.strategy_recent[i]
                    if len(recent) >= 5:
                        wr = recent.count('w') / len(recent)
                        if wr > best_wr:
                            best_wr = wr
                            best_idx = i
                
                if best_idx is not None:
                    self.exploitation_strategy = best_idx
                    self.gamma = 0.05  # Minimal exploration
            
            # USE EXPLOITATION STRATEGY
            if self.exploitation_strategy is not None:
                move = self.strategy_predict(self.exploitation_strategy)
                self.last_strategy_idx = self.exploitation_strategy
                self.prediction_confidence = 0.9
                self.last_ai_move = move
                print(f"⚡ EXPLOITING with {self.strategy_names[self.exploitation_strategy]}: {move.upper()}")
                return move
        else:
            if self.exploitation_mode:
                print("📊 Exiting exploitation mode")
            self.exploitation_mode = False
            self.exploitation_strategy = None
            self.gamma = self.base_gamma
        
        # ===== LOSS RECOVERY =====
        # If we're losing, GET AGGRESSIVE
        if self.losing_streak >= 2:
            print(f"🚨 LOSS RECOVERY MODE (streak: {self.losing_streak})")
            
            # Blacklist recently failed strategies
            failed_strategies = set()
            for loss_info in self.recent_losses:
                if 'strategy_idx' in loss_info:
                    failed_strategies.add(loss_info['strategy_idx'])
            
            # Try a DIFFERENT strategy that's been working
            available_strategies = [i for i in range(self.K) if i not in failed_strategies]
            if available_strategies:
                # Pick best performing available strategy
                best_idx = max(available_strategies, 
                             key=lambda i: (self.strategy_recent[i].count('w') / len(self.strategy_recent[i]) 
                                          if len(self.strategy_recent[i]) >= 3 else 0.0))
                
                move = self.strategy_predict(best_idx)
                self.last_strategy_idx = best_idx
                self.prediction_confidence = 0.7
                self.last_ai_move = move
                print(f"🔄 SWITCHING to {self.strategy_names[best_idx]}: {move.upper()}")
                return move
        
        # ===== COUNTER-INTELLIGENCE =====
        # If being counter-predicted, use deception
        if self.counter_predict_evidence >= 3:
            if random.random() < 0.5:  # 50% chance to deceive
                print("🎭 DECEPTION MODE - Playing unpredictably")
                self.deception_mode = True
                move = random.choice(self.MOVES)
                self.last_ai_move = move
                self.prediction_confidence = 0.5
                self.last_strategy_idx = None
                return move
        
        # ===== ARCHETYPE-BASED SELECTION =====
        if self.opponent_archetype == "random" and self.archetype_confidence > 0.7:
            # Against random, use Nash equilibrium
            move = random.choice(self.MOVES)
            print(f"🎲 NASH vs Random: {move.upper()}")
            self.last_ai_move = move
            self.prediction_confidence = 0.33
            return move
        
        # ===== NORMAL EXP3 STRATEGY SELECTION =====
        # Weighted strategy selection
        W = self.weights.copy()
        if W.sum() == 0:
            W = np.ones(self.K)
        
        # Adjust gamma based on confidence
        if self.archetype_confidence > 0.6:
            self.gamma = max(0.05, self.base_gamma - 0.1)  # Less exploration when confident
        else:
            self.gamma = self.base_gamma
        
        probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
        probs = probs / probs.sum()
        
        # Choose strategy
        idx = int(np.random.choice(self.K, p=probs))
        self.last_strategy_idx = idx
        self.strategy_plays[idx] += 1
        
        move = self.strategy_predict(idx)
        self.prediction_confidence = float(probs[idx])
        self.last_ai_move = move
        
        return move

    # ========== LEARNING & UPDATE ==========
    
    def update(self, ai_move, result):
        """Learn from the outcome - THIS IS WHERE WE GET SMARTER."""
        if result not in ('w', 'l', 'd'):
            raise ValueError("result must be 'w', 'l', or 'd'")
        
        # Infer opponent move
        opponent_move = None
        for m in self.MOVES:
            if self.move_result(ai_move, m) == result:
                opponent_move = m
                break
        
        if opponent_move is None:
            opponent_move = random.choice(self.MOVES)
        
        # Update histories
        self.opponent_history.append(opponent_move)
        self.ai_history.append(ai_move)
        self.results.append(result)
        self.total_rounds += 1
        
        # Update losing streak
        if result == 'l':
            self.losing_streak += 1
            self.recent_losses.append({
                'move': ai_move,
                'opponent_move': opponent_move,
                'strategy_idx': self.last_strategy_idx,
                'round': self.total_rounds
            })
            
            # Detect being exploited
            if self.losing_streak >= 3:
                self.being_exploited = True
                self.exploitation_mode = False  # Exit exploitation if losing
            
            # Detect counter-predicting
            if len(self.ai_history) >= 2:
                if opponent_move == self.BEATEN_BY.get(self.ai_history[-2], None):
                    self.counter_predict_evidence += 1
        else:
            self.losing_streak = max(0, self.losing_streak - 1)
            if result == 'w':
                self.counter_predict_evidence = max(0, self.counter_predict_evidence - 1)
                self.being_exploited = False
        
        # Update frequency windows
        for window_size in [10, 20, 50, 100]:
            if len(self.opponent_history) >= window_size:
                recent = list(self.opponent_history)[-window_size:]
                self.freq_windows[window_size] = Counter(recent)
        self.freq_windows['all'][opponent_move] += 1
        
        # Update Markov chains
        for order in range(1, 6):
            if len(self.opponent_history) > order:
                context = tuple(list(self.opponent_history)[-(order+1):-1])
                self.markov_chains[order][context][self.IDX[opponent_move]] += 1.0
        
        # Update meta-learning
        if len(self.ai_history) >= 2:
            prev_ai = self.ai_history[-2]
            self.response_patterns[prev_ai][self.IDX[opponent_move]] += 1.0
        
        if result == 'l':
            self.what_beats_us[ai_move][self.IDX[opponent_move]] += 2.0
        elif result == 'w':
            self.what_we_beat[ai_move][self.IDX[opponent_move]] += 2.0
        
        # Update strategy performance
        if self.last_strategy_idx is not None:
            self.strategy_recent[self.last_strategy_idx].append(result)
            
            if result == 'w':
                self.strategy_wins[self.last_strategy_idx] += 1
                reward = 2.0
            elif result == 'l':
                self.strategy_losses[self.last_strategy_idx] += 1
                reward = -2.0
            else:
                reward = 0.2
            
            # AGGRESSIVE EXP3 UPDATE
            eta = self.eta
            if self.losing_streak >= 2:
                eta *= 2.0  # Learn faster when losing
            
            W = self.weights.copy()
            if W.sum() == 0:
                W = np.ones(self.K)
            probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
            p_i = probs[self.last_strategy_idx]
            
            x_hat = reward / max(p_i, 0.01)
            self.weights[self.last_strategy_idx] *= np.exp(min(eta * x_hat / self.K, 3.0))
            
            # Decay weights
            decay = 0.995 if self.losing_streak < 2 else 0.985
            self.weights *= decay
            
            # Ensure no weight goes to zero
            self.weights = np.maximum(self.weights, 0.01)
        
        # TensorFlow training with prioritized experience replay
        if TF_AVAILABLE and self.tf_models:
            self._add_experience(ai_move, opponent_move, result)
            
            # Train periodically
            if self.total_rounds % 25 == 0:
                self._train_tf_models()
        
        # Auto-save
        if self.total_rounds % 50 == 0:
            self.save_state()

    def _add_experience(self, ai_move, opponent_move, result):
        """Add experience to training buffer."""
        try:
            # Main model features
            features_main = self._build_tf_features_main()
            if features_main and len(features_main) == 30:
                winning_move = self.BEATEN_BY[opponent_move]
                target = self._one_hot(winning_move)
                
                experience = {
                    'features_main': features_main,
                    'target': target,
                    'result': result
                }
                
                # Prioritize losses and wins
                if result in ['w', 'l']:
                    self.priority_buffer.append(experience)
                
                self.experience_buffer.append(experience)
        except:
            pass

    def _train_tf_models(self):
        """Train TensorFlow models with prioritized replay."""
        if not self.experience_buffer:
            return
        
        try:
            # Combine priority and regular experiences
            experiences = list(self.priority_buffer) + list(self.experience_buffer)
            
            if len(experiences) < 32:
                return
            
            # Sample experiences
            sample_size = min(128, len(experiences))
            sample = random.sample(experiences, sample_size)
            
            # Prepare training data
            X = np.array([exp['features_main'] for exp in sample if len(exp['features_main']) == 30])
            y = np.array([exp['target'] for exp in sample if len(exp['features_main']) == 30])
            
            if len(X) >= 32 and 'main' in self.tf_models:
                self.tf_models['main'].fit(X, y, epochs=3, batch_size=32, verbose=0)
                
            # Save models
            if self.total_rounds % 100 == 0:
                for name, model in self.tf_models.items():
                    try:
                        model.save(f"{self.tf_model_dir}_{name}", overwrite=True)
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ TF training error: {e}")

    def move_result(self, ai_move, opp_move):
        """Determine result of a match."""
        if ai_move == opp_move:
            return 'd'
        if self.BEATS[ai_move] == opp_move:
            return 'w'
        return 'l'

    # ========== STATS ==========
    
    def stats(self):
        """Display comprehensive statistics."""
        total = len(self.results)
        if total == 0:
            return "No games played yet."
        
        wins = list(self.results).count('w')
        losses = list(self.results).count('l')
        draws = list(self.results).count('d')
        win_rate = wins / total * 100
        
        recent_50 = list(self.results)[-50:]
        recent_wr = (recent_50.count('w') / len(recent_50) * 100) if recent_50 else 0
        
        last_20 = list(self.results)[-20:]
        last_20_wr = (last_20.count('w') / len(last_20) * 100) if last_20 else 0
        
        # Top strategies
        strat_info = []
        for i, name in enumerate(self.strategy_names):
            if self.strategy_plays[i] > 0:
                recent = self.strategy_recent[i]
                recent_wr_s = (recent.count('w') / len(recent) * 100) if recent else 0
                strat_info.append((name, self.strategy_plays[i], recent_wr_s, self.weights[i]))
        
        strat_info.sort(key=lambda x: x[3], reverse=True)
        top_5 = strat_info[:5]
        
        output = f"""
🔥 ===== MONSTER RPS AI STATS ===== 🔥
Total Games: {total} | W:{wins} L:{losses} D:{draws}
Overall WR: {win_rate:.1f}% | Last 50: {recent_wr:.1f}% | Last 20: {last_20_wr:.1f}%
Losing Streak: {self.losing_streak} | Exploitation Mode: {'ON 🔥' if self.exploitation_mode else 'OFF'}
Opponent Type: {self.opponent_archetype.upper()} (confidence: {self.archetype_confidence:.2f})
Patterns Detected: {len(self.patterns)} | Counter-Predict Evidence: {self.counter_predict_evidence}

🏆 Top 5 Strategies:
"""
        for name, plays, wr, weight in top_5:
            output += f"  {name:18s}: {int(plays):4d} plays | Recent WR:{wr:5.1f}% | W:{weight:7.2f}\n"
        
        output += f"\n⚡ Gamma: {self.gamma:.3f} | Confidence: {self.prediction_confidence:.3f}"
        output += f"\nSession: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return output


def main():
    """Main game loop."""
    print("🔥" * 25)
    print("      MONSTER RPS AI - UNBEATABLE MODE")
    print("🔥" * 25)
    
    ai = MonsterRPS()
    
    print("\n✅ Monster AI Ready!")
    print("  • 20+ Advanced Strategies")
    print("  • Multi-scale Pattern Detection")
    print("  • Opponent Modeling & Classification")
    print("  • Counter-Intelligence & Deception")
    print("  • Aggressive Loss Recovery")
    print("  • Exploitation Mode")
    print("  • Deep Learning (TensorFlow)")
    print("\n📋 Instructions:")
    print("  'w' = AI won | 'l' = AI lost | 'd' = draw")
    print("  Commands: stats, save, exit\n")
    
    while True:
        try:
            ai_move = ai.get_move()
        except Exception as e:
            print(f"⚠️ Error in get_move: {e}")
            traceback.print_exc()
            ai_move = random.choice(ai.MOVES)
        
        status = f"\n🤖 AI plays: {ai_move.upper()}"
        if ai.last_strategy_idx is not None and ai.last_strategy_idx < len(ai.strategy_names):
            status += f" | Strategy: {ai.strategy_names[ai.last_strategy_idx]}"
        status += f" | Confidence: {ai.prediction_confidence:.2f}"
        
        print(status)
        result = input("Result (w/l/d) or command: ").strip().lower()
        
        if result == 'exit':
            ai.save_state()
            print("💾 Monster saved. Until next time! 🔥")
            break
        elif result == 'stats':
            print(ai.stats())
            continue
        elif result == 'save':
            ai.save_state()
            continue
        elif result not in ('w', 'l', 'd'):
            print("❌ Invalid input")
            continue
        
        try:
            ai.update(ai_move, result)
        except Exception as e:
            print(f"⚠️ Error in update: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
