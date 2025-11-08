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
                gamma=0.15,  # Increased exploration
                eta=None,
                seed=None,
                autosave_every=50,
                tf_train_every=25,  # Train more frequently
                tf_epochs=15,  # More epochs
                tf_batch=32,  # Smaller batches for better gradients
                tf_memory=2000):  # Larger memory
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

        # Strategy tracking - now includes win/loss per strategy
        self.strategy_names = [
            'frequency', 'anti_frequency', 'markov1', 'markov2', 
            'ngram', 'streak', 'anti_streak', 'cycle', 'meta_predict', 'random'
        ]
        if TF_AVAILABLE:
            self.strategy_names.append('tf')
        self.K = len(self.strategy_names)

        # Enhanced EXP3 with win/loss tracking
        self.weights = np.ones(self.K, dtype=float)
        self.gamma = float(gamma)
        self.total_rounds = 0
        self.eta = eta if eta is not None else np.sqrt(np.log(max(2, self.K)) / (max(1, self.K) * 500.0))
        
        # Track performance per strategy
        self.strategy_wins = np.zeros(self.K, dtype=float)
        self.strategy_plays = np.zeros(self.K, dtype=float)

        # Classical models with decay
        self.freq_counts = Counter()
        self.recency_weights = defaultdict(float)
        self.decay = 0.97  # Faster decay for adaptation
        self.markov1 = np.ones((3, 3))
        self.markov2 = defaultdict(lambda: np.ones(3))
        self.ngram_len = 4  # Longer patterns
        self.ngram = defaultdict(lambda: np.ones(3))

        self.current_streak_move = None
        self.current_streak_count = 0
        
        # Meta-learning: track what opponent does after our moves
        self.response_patterns = defaultdict(lambda: np.ones(3))

        # TF memory and training
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
        """Enhanced TF model with more capacity and dropout."""
        model = keras.Sequential([
            layers.Input(shape=(12,)),  # Larger input: last 2 moves from each player
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.002),
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
                'current_streak_move': self.current_streak_move,
                'current_streak_count': self.current_streak_count,
                'tf_buffer': list(self.tf_train_buffer)
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(payload, f)
            if TF_AVAILABLE and self.tf_model:
                self._save_tf_model()
            print(f"💾 State saved successfully to: {self.state_file}")
            return True
        except Exception as e:
            print("⚠️ save_state failed:", e)
            traceback.print_exc()
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
            
            # Handle strategy count mismatch between saved state and current code
            loaded_weights = np.array(payload.get('weights', []), dtype=float)
            loaded_wins = np.array(payload.get('strategy_wins', []), dtype=float)
            loaded_plays = np.array(payload.get('strategy_plays', []), dtype=float)
            
            # If sizes don't match, resize arrays
            if len(loaded_weights) != self.K:
                print(f"⚠️ Strategy count mismatch: saved={len(loaded_weights)}, current={self.K}. Resizing...")
                new_weights = np.ones(self.K, dtype=float)
                new_wins = np.zeros(self.K, dtype=float)
                new_plays = np.zeros(self.K, dtype=float)
                
                # Copy old values where possible
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
            
            self.markov2 = defaultdict(lambda: np.ones(3))
            for k, v in payload.get('markov2', {}).items():
                self.markov2[tuple(k)] = np.array(v)
            
            self.ngram = defaultdict(lambda: np.ones(3))
            for k, v in payload.get('ngram', {}).items():
                self.ngram[tuple(k)] = np.array(v)
            
            self.response_patterns = defaultdict(lambda: np.ones(3))
            for k, v in payload.get('response_patterns', {}).items():
                self.response_patterns[k] = np.array(v)
            
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
        """Return 'w' if AI wins, 'l' if AI loses, 'd' for draw."""
        if ai_move == opp_move:
            return 'd'
        if self.BEATS[ai_move] == opp_move:
            return 'w'
        return 'l'

    def _opponent_from_result(self, ai_move, result):
        """Reconstruct opponent move from AI move and result."""
        for m in self.MOVES:
            if self.move_result(ai_move, m) == result:
                return m
        return random.choice(self.MOVES)

    # ========== PREDICTION STRATEGIES ==========
    
    def _freq_predict(self):
        """Predict most frequent move, then counter it."""
        if not self.opponent_history or not self.recency_weights:
            return random.choice(self.MOVES)
        # Weight recent moves more heavily
        best = max(self.recency_weights.items(), key=lambda x: x[1])[0]
        return self.BEATEN_BY[best]  # Return counter move

    def _anti_freq_predict(self):
        """Assume opponent avoids their most frequent move."""
        if not self.opponent_history or not self.recency_weights:
            return random.choice(self.MOVES)
        least = min(self.recency_weights.items(), key=lambda x: x[1])[0]
        return self.BEATEN_BY[least]

    def _markov1_predict(self):
        """1st order Markov: predict based on last opponent move."""
        if len(self.opponent_history) < 1:
            return random.choice(self.MOVES)
        last = self._hist_slice(-1, None)[0]
        arr = self.markov1[self.IDX[last]]
        total = np.sum(arr)
        if total <= 3:  # Not enough data
            return random.choice(self.MOVES)
        probs = arr / total
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _markov2_predict(self):
        """2nd order Markov: predict based on last 2 opponent moves."""
        if len(self.opponent_history) < 2:
            return self._markov1_predict()
        ctx = tuple(self._hist_slice(-2, None))
        arr = self.markov2.get(ctx)
        if arr is None or np.sum(arr) <= 3:
            return self._markov1_predict()
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _ngram_predict(self):
        """N-gram prediction with longer sequences."""
        if len(self.opponent_history) < self.ngram_len:
            return self._markov2_predict()
        key = tuple(self._hist_slice(-self.ngram_len, None))
        arr = self.ngram.get(key)
        if arr is None or np.sum(arr) <= 3:
            return self._markov2_predict()
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _streak_predict(self):
        """Predict opponent continues streak."""
        if self.current_streak_move is None or self.current_streak_count < 2:
            return random.choice(self.MOVES)
        return self.BEATEN_BY[self.current_streak_move]

    def _anti_streak_predict(self):
        """Predict opponent breaks streak."""
        if self.current_streak_move is None or self.current_streak_count < 2:
            return random.choice(self.MOVES)
        others = [m for m in self.MOVES if m != self.current_streak_move]
        predicted = random.choice(others)
        return self.BEATEN_BY[predicted]

    def _cycle_predict(self):
        """Detect cyclic patterns."""
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
        """Predict based on opponent's response to our moves."""
        if len(self.ai_history) < 1:
            return random.choice(self.MOVES)
        last_ai = self.ai_history[-1]
        arr = self.response_patterns.get(last_ai)
        if arr is None or np.sum(arr) <= 3:
            return random.choice(self.MOVES)
        probs = arr / np.sum(arr)
        predicted = self.MOVES[int(np.argmax(probs))]
        return self.BEATEN_BY[predicted]

    def _random_predict(self):
        """Pure random baseline."""
        return random.choice(self.MOVES)

    def _tf_predict(self):
        """TensorFlow deep learning prediction."""
        if (not TF_AVAILABLE) or (self.tf_model is None):
            return random.choice(self.MOVES)
        if len(self.opponent_history) < 2 or len(self.ai_history) < 2:
            return random.choice(self.MOVES)
        
        # Use last 2 moves from each player as context
        last_ai_2 = self.one_hot(self.ai_history[-2])
        last_ai_1 = self.one_hot(self.ai_history[-1])
        last_opp_2 = self.one_hot(self.opponent_history[-2])
        last_opp_1 = self.one_hot(self.opponent_history[-1])
        x = np.concatenate([last_ai_2, last_ai_1, last_opp_2, last_opp_1]).reshape(1, -1)
        
        try:
            probs = self.tf_model.predict(x, verbose=0)[0]
            predicted = self.MOVES[int(np.argmax(probs))]
            return self.BEATEN_BY[predicted]
        except Exception:
            return random.choice(self.MOVES)

    def strategy_predict(self, idx):
        """Execute a strategy by index."""
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
            self._random_predict
        ]
        if TF_AVAILABLE:
            funcs.append(self._tf_predict)
        idx = int(idx) % len(funcs)
        return funcs[idx]()

    # ========== EXP3 STRATEGY SELECTION ==========
    
    def get_move(self):
        """Select strategy using EXP3 and get move."""
        W = np.array(self.weights, dtype=float)
        
        # Ensure weights array matches strategy count
        if len(W) != self.K:
            print(f"⚠️ Weight array size mismatch. Resizing from {len(W)} to {self.K}")
            W = np.ones(self.K, dtype=float)
            self.weights = W
        
        if W.sum() == 0:
            W = np.ones_like(W)
        
        # EXP3 probability distribution
        probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
        
        # Ensure probs sum to 1 (numerical stability)
        probs = probs / probs.sum()
        
        idx = int(np.random.choice(self.K, p=probs))
        
        self.last_strategy_idx = idx
        self.strategy_plays[idx] += 1
        
        # Get the move from selected strategy
        our_move = self.strategy_predict(idx)
        self.last_ai_move = our_move
        self.prediction_confidence = float(probs[idx])
        
        return our_move

    # ========== LEARNING & UPDATE ==========
    
    def update(self, ai_move, result):
        """Update all models based on game result."""
        if result not in ('w', 'l', 'd'):
            raise ValueError("result must be 'w','l' or 'd' (from AI's perspective)")

        opponent_move = self._opponent_from_result(ai_move, result)

        self.opponent_history.append(opponent_move)
        self.ai_history.append(ai_move)
        self.results.append(result)
        self.total_rounds += 1

        # Update frequency models with decay
        for k in list(self.recency_weights.keys()):
            self.recency_weights[k] *= self.decay
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

        # Update n-gram model
        if len(self.opponent_history) >= self.ngram_len + 1:
            key = tuple(self._hist_slice(-(self.ngram_len+1), -1))
            self.ngram[key][self.IDX[opponent_move]] += 1.0

        # Update response patterns (opponent's response to our moves)
        if len(self.ai_history) >= 2:
            our_prev = self.ai_history[-2]
            self.response_patterns[our_prev][self.IDX[opponent_move]] += 1.0

        # Update streak tracking
        if self.current_streak_move == opponent_move:
            self.current_streak_count += 1
        else:
            self.current_streak_move = opponent_move
            self.current_streak_count = 1

        # TF training data collection
        if TF_AVAILABLE and self.tf_model is not None:
            if len(self.ai_history) >= 2 and len(self.opponent_history) >= 2:
                last_ai_2 = self.one_hot(self.ai_history[-2])
                last_ai_1 = self.one_hot(self.ai_history[-1])
                last_opp_2 = self.one_hot(self.opponent_history[-2])
                last_opp_1 = self.one_hot(self.opponent_history[-1])
                x = np.concatenate([last_ai_2, last_ai_1, last_opp_2, last_opp_1]).astype(float)
                y = self.one_hot(opponent_move).astype(float)
                self.tf_train_buffer.append((x, y))

        # ========== IMPROVED EXP3 UPDATE ==========
        if self.last_strategy_idx is not None:
            # Direct reward from game outcome
            if result == 'w':
                reward = 1.0
                self.strategy_wins[self.last_strategy_idx] += 1.0
            elif result == 'l':
                reward = -0.5  # Penalize losses
            else:
                reward = 0.1  # Small reward for draws
            
            # Calculate EXP3 update
            W = np.array(self.weights, dtype=float)
            if W.sum() == 0:
                W = np.ones_like(W)
            probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
            p_i = probs[self.last_strategy_idx]
            
            # Estimated reward (importance sampling)
            x_hat = reward / p_i if p_i > 0 else 0.0
            
            # Update weight with clipping to prevent explosion
            self.weights[self.last_strategy_idx] *= np.exp(min(self.eta * x_hat / self.K, 5.0))
            
            # Gentle decay to allow adaptation
            self.weights *= 0.995

        # TF model training
        if TF_AVAILABLE and self.tf_model is not None:
            if (self.total_rounds % self.tf_train_every == 0) and (len(self.tf_train_buffer) >= 16):
                try:
                    X = np.array([t[0] for t in self.tf_train_buffer])
                    Y = np.array([t[1] for t in self.tf_train_buffer])
                    self.tf_model.fit(X, Y, epochs=self.tf_epochs, batch_size=self.tf_batch, verbose=0)
                    self._save_tf_model()
                except Exception as e:
                    print(f"⚠️ TF training error: {e}")

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
        
        # Calculate win rate per strategy
        strat_info = []
        for i, name in enumerate(self.strategy_names):
            plays = self.strategy_plays[i]
            wins_s = self.strategy_wins[i]
            wr = (wins_s / plays * 100) if plays > 0 else 0.0
            weight = self.weights[i]
            strat_info.append((name, plays, wr, weight))
        
        # Sort by weight
        strat_info.sort(key=lambda x: x[3], reverse=True)
        top_4 = strat_info[:4]
        
        tf_info = "TF:✓" if (TF_AVAILABLE and self.tf_model is not None) else "TF:✗"
        
        output = f"""
📊 ===== STATS ===== 📊
Games: {total} | W:{wins} L:{losses} D:{draws}
Overall WR: {win_rate:.1f}% | Recent 50 WR: {recent_wr:.1f}%
Confidence: {self.prediction_confidence:.3f} | {tf_info}

🏆 Top Strategies:
"""
        for name, plays, wr, weight in top_4:
            output += f"  {name:14s}: {int(plays):4d} plays | {wr:5.1f}% WR | weight:{weight:7.2f}\n"
        
        output += f"\nSession: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return output

# ========== MAIN LOOP ==========

def main(save_prefix="rps_beast"):
    print("Current working directory:", os.getcwd())
    ai = RealTimeEvolvingRPS_TF(save_prefix)
    plotter = RPSPlotter(ai)
    print("🎮 Enhanced Evolving RPS AI Ready!")
    print("AI suggests moves. Report results from AI's perspective:")
    print("  'w' = AI won | 'l' = AI lost | 'd' = draw")
    print("Commands: stats, reset, save, exit\n")
    
    while True:
        try:
            ai_move = ai.get_move()
        except Exception as e:
            print("⚠️ get_move error:", e)
            traceback.print_exc()
            ai_move = random.choice(ai.MOVES)

        print(f"\n🤖 AI plays: {ai_move.upper()} | Confidence: {ai.prediction_confidence:.3f}", end="")
        if ai.last_strategy_idx is not None:
            print(f" | Strategy: {ai.strategy_names[ai.last_strategy_idx]}")
        else:
            print()
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