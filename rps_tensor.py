# rps_tf_evolving_fixed.py  (patched)
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
    # BEATS maps a move to the move it beats (canonical)
    BEATS = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    # BEATEN_BY maps a move to the move that beats it (inverse of BEATS)
    BEATEN_BY = {v: k for k, v in BEATS.items()}

    def __init__(self,
                save_prefix="rps_beast2",
                memory_size=5000,
                gamma=0.12,
                eta=None,
                seed=None,
                autosave_every=50,
                tf_train_every=50,
                tf_epochs=10,
                tf_batch=64,
                tf_memory=1000):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.save_prefix = save_prefix
        # Always save to the same directory as this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_file = os.path.join(base_dir, f"{save_prefix}_state.pkl")
        self.tf_model_dir = os.path.join(base_dir, f"{save_prefix}_tf_model")

        self.tf_model_dir = f"{save_prefix}_tf_model"
        self.memory_size = memory_size

        # histories
        self.opponent_history = deque(maxlen=memory_size)
        self.ai_history = deque(maxlen=memory_size)
        self.results = []

        # strategies
        self.strategy_names = [
            'frequency', 'markov1', 'markov2', 'ngram', 'streak', 'cycle', 'random'
        ]
        if TF_AVAILABLE:
            self.strategy_names.append('tf')
        self.K = len(self.strategy_names)

        # EXP3
        self.weights = np.ones(self.K, dtype=float)
        self.gamma = float(gamma)
        self.total_rounds = 0
        self.eta = eta if eta is not None else np.sqrt(np.log(max(2, self.K)) / (max(1, self.K) * 1000.0))

        # classical models
        self.freq_counts = Counter()
        self.recency_weights = defaultdict(float)
        self.decay = 0.985
        self.markov1 = np.ones((3, 3))
        self.markov2 = defaultdict(lambda: np.ones(3))
        self.ngram_len = 3
        self.ngram = defaultdict(lambda: np.ones(3))

        self.current_streak_move = None
        self.current_streak_count = 0

        # TF memory
        self.tf_train_buffer = deque(maxlen=tf_memory)
        self.tf_train_every = tf_train_every
        self.tf_epochs = tf_epochs
        self.tf_batch = tf_batch

        self.prediction_confidence = 0.0
        self.last_strategy_idx = None
        self.autosave_every = autosave_every

        # TF model
        self.tf_model = None
        if TF_AVAILABLE:
            self._init_or_load_tf_model()

        self.load_state()

    # -----------------------------
    # deque slicing helper
    # -----------------------------
    def _hist_slice(self, start=None, end=None):
        """Safely slice opponent_history as a list."""
        hist = list(self.opponent_history)
        return hist[start:end]

    # -----------------------------
    # TensorFlow helpers
    # -----------------------------
    def _build_tf_model(self):
        model = keras.Sequential([
            layers.Input(shape=(6,)),
            layers.Dense(48, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy', metrics=[])
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

    # -----------------------------
    # Persistence
    # -----------------------------
    def save_state(self):
        try:
            payload = {
                'opponent_history': list(self.opponent_history),
                'ai_history': list(self.ai_history),
                'results': self.results,
                'weights': self.weights.tolist(),
                'total_rounds': self.total_rounds,
                'freq_counts': dict(self.freq_counts),
                'recency_weights': dict(self.recency_weights),
                'markov1': self.markov1.tolist(),
                'markov2': {k: v.tolist() for k, v in self.markov2.items()},
                'ngram': {k: v.tolist() for k, v in self.ngram.items()},
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
            self.weights = np.array(payload.get('weights', self.weights.tolist()), dtype=float)
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

    # -----------------------------
    # Helpers
    # -----------------------------
    def one_hot(self, move):
        v = np.zeros(3, dtype=float)
        if move in self.MOVES:
            v[self.IDX[move]] = 1.0
        return v

    def move_result(self, ai_move, opp_move):
        """Return 'w' if AI wins, 'l' if AI loses, 'd' for draw."""
        if ai_move == opp_move:
            return 'd'
        # If BEATS[ai_move] == opp_move, then ai_move beats opp_move.
        if self.BEATS[ai_move] == opp_move:
            return 'w'
        return 'l'

    def _opponent_from_result(self, ai_move, result):
        """
        Reconstruct the opponent move given what the AI played and the result
        (result is with respect to the AI: 'w' if AI won, 'l' if AI lost, 'd' draw).
        """
        for m in self.MOVES:
            if self.move_result(ai_move, m) == result:
                return m
        # fallback (shouldn't happen)
        return random.choice(self.MOVES)

    # -----------------------------
    # Prediction strategies
    # -----------------------------
    def _freq_predict(self):
        if not self.opponent_history or not self.freq_counts:
            return random.choice(self.MOVES)
        if self.recency_weights:
            best = max(self.recency_weights.items(), key=lambda x: x[1])[0]
            return best
        mc = self.freq_counts.most_common(1)
        if not mc:
            return random.choice(self.MOVES)
        return mc[0][0]

    def _markov1_predict(self):
        if len(self.opponent_history) < 1:
            return random.choice(self.MOVES)
        last = self._hist_slice(-1, None)[0]
        arr = self.markov1[self.IDX[last]]
        total = np.sum(arr)
        if total <= 0:
            return random.choice(self.MOVES)
        probs = arr / total
        return self.MOVES[int(np.argmax(probs))]

    def _markov2_predict(self):
        if len(self.opponent_history) < 2:
            return self._markov1_predict()
        ctx = tuple(self._hist_slice(-2, None))
        arr = self.markov2.get(ctx)
        if arr is None:
            return self._markov1_predict()
        total = np.sum(arr)
        if total <= 0:
            return self._markov1_predict()
        probs = arr / total
        return self.MOVES[int(np.argmax(probs))]

    def _ngram_predict(self):
        if len(self.opponent_history) < self.ngram_len:
            return random.choice(self.MOVES)
        key = tuple(self._hist_slice(-self.ngram_len, None))
        arr = self.ngram.get(key)
        if arr is None:
            return self._markov2_predict()
        total = np.sum(arr)
        if total <= 0:
            return self._markov2_predict()
        probs = arr / total
        return self.MOVES[int(np.argmax(probs))]

    def _streak_predict(self):
        if self.current_streak_move is None:
            return random.choice(self.MOVES)
        if self.current_streak_count >= 3:
            others = [m for m in self.MOVES if m != self.current_streak_move]
            return random.choice(others)
        return self.current_streak_move

    def _cycle_predict(self):
        seq = self._hist_slice(0, None)
        L = len(seq)
        if L < 3:
            return random.choice(self.MOVES)
        for cycle_len in range(2, min(7, max(3, L))):
            if L >= 2 * cycle_len:
                last = tuple(seq[-cycle_len:])
                prev = tuple(seq[-2 * cycle_len:-cycle_len])
                if last == prev:
                    return last[0]
        return random.choice(self.MOVES)

    def _random_predict(self):
        return random.choice(self.MOVES)

    def _tf_predict(self):
        if (not TF_AVAILABLE) or (self.tf_model is None):
            return random.choice(self.MOVES)
        if len(self.opponent_history) < 1 or len(self.ai_history) < 1:
            return random.choice(self.MOVES)
        last_ai = self.ai_history[-1]
        last_opp = self.opponent_history[-1]
        x = np.concatenate([self.one_hot(last_ai), self.one_hot(last_opp)]).reshape(1, -1)
        try:
            probs = self.tf_model.predict(x, verbose=0)[0]
            return self.MOVES[int(np.argmax(probs))]
        except Exception:
            return random.choice(self.MOVES)

    def strategy_predict(self, idx):
        funcs = [
            self._freq_predict,
            self._markov1_predict,
            self._markov2_predict,
            self._ngram_predict,
            self._streak_predict,
            self._cycle_predict,
            self._random_predict
        ]
        if TF_AVAILABLE:
            funcs.append(self._tf_predict)
        idx = int(idx) % len(funcs)
        return funcs[idx]()

    # -----------------------------
    # EXP3 selection
    # -----------------------------
    def get_move(self):
        W = np.array(self.weights, dtype=float)
        if W.sum() == 0:
            W = np.ones_like(W)
        probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
        idx = int(np.random.choice(self.K, p=probs))
        self.last_strategy_idx = idx
        predicted_opponent = self.strategy_predict(idx)
        # pick the move that beats the predicted opponent move
        our_move = self.BEATEN_BY[predicted_opponent]
        self.prediction_confidence = float(probs[idx])
        return our_move

    # -----------------------------
    # Update
    # -----------------------------
    def update(self, ai_move, result):
        if result not in ('w', 'l', 'd'):
            raise ValueError("result must be 'w','l' or 'd' (from AI's perspective)")

        # reconstruct actual opponent move (result is relative to the AI)
        opponent_move = self._opponent_from_result(ai_move, result)

        self.opponent_history.append(opponent_move)
        self.ai_history.append(ai_move)
        self.results.append(result)
        self.total_rounds += 1

        for k in list(self.recency_weights.keys()):
            self.recency_weights[k] *= self.decay
        self.recency_weights[opponent_move] += 1.0
        self.freq_counts[opponent_move] += 1

        if len(self.opponent_history) >= 2:
            prev = self.opponent_history[-2]
            cur = self.opponent_history[-1]
            self.markov1[self.IDX[prev], self.IDX[cur]] += 1.0
        if len(self.opponent_history) >= 3:
            ctx = tuple(self._hist_slice(-3, -1))
            self.markov2[ctx][self.IDX[self.opponent_history[-1]]] += 1.0

        if len(self.opponent_history) >= self.ngram_len + 1:
            key = tuple(self._hist_slice(-(self.ngram_len+1), -1))
            self.ngram[key][self.IDX[self.opponent_history[-1]]] += 1.0

        if self.current_streak_move == opponent_move:
            self.current_streak_count += 1
        else:
            self.current_streak_move = opponent_move
            self.current_streak_count = 1

        if TF_AVAILABLE and self.tf_model is not None:
            if len(self.ai_history) >= 1 and len(self.opponent_history) >= 1:
                last_ai = self.ai_history[-1]
                last_opp = self.opponent_history[-1]
                x = np.concatenate([self.one_hot(last_ai), self.one_hot(last_opp)]).astype(float)
                y = self.one_hot(opponent_move).astype(float)
                self.tf_train_buffer.append((x, y))

        if self.last_strategy_idx is not None:
            pred = self.strategy_predict(self.last_strategy_idx)
            # reward is whether that strategy predicted the opponent correctly
            reward = 1.0 if pred == opponent_move else 0.0
            W = np.array(self.weights, dtype=float)
            if W.sum() == 0:
                W = np.ones_like(W)
            probs = (1 - self.gamma) * (W / W.sum()) + self.gamma / self.K
            p_i = probs[self.last_strategy_idx]
            x_hat = reward / p_i if p_i > 0 else 0.0
            self.weights[self.last_strategy_idx] *= np.exp(self.eta * x_hat / self.K)
            self.weights *= 0.9999

        if TF_AVAILABLE and self.tf_model is not None:
            if (self.total_rounds % self.tf_train_every == 0) and (len(self.tf_train_buffer) >= 8):
                try:
                    X = np.array([t[0] for t in self.tf_train_buffer])
                    Y = np.array([t[1] for t in self.tf_train_buffer])
                    self.tf_model.fit(X, Y, epochs=self.tf_epochs, batch_size=self.tf_batch, verbose=0)
                    try:
                        self._save_tf_model()
                    except Exception:
                        pass
                except Exception:
                    pass

        if self.total_rounds % self.autosave_every == 0:
            self.save_state()

    # -----------------------------
    # Stats
    # -----------------------------
    def stats(self):
        total = len(self.results)
        wins = self.results.count('w')
        losses = self.results.count('l')
        draws = self.results.count('d')
        win_rate = wins / total * 100 if total else 0.0
        recent = self.results[-20:] if len(self.results) >= 20 else self.results
        recent_wr = (recent.count('w') / len(recent) * 100) if recent else 0.0
        top_strats = sorted([(n, float(w)) for n, w in zip(self.strategy_names, self.weights)],
                            key=lambda x: x[1], reverse=True)[:4]
        tf_info = "TF:OK" if (TF_AVAILABLE and self.tf_model is not None) else "TF:Missing"
        return f"""
📊 Stats 📊
Games: {total} | W:{wins} L:{losses} D:{draws}
Overall WR: {win_rate:.1f}% | Recent 20 WR: {recent_wr:.1f}%
Confidence: {self.prediction_confidence:.3f} | {tf_info}
Top Strategies: {[(s, f'{v:.2f}') for s, v in top_strats]}
Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# -----------------------------
# Main interactive loop
# -----------------------------
def main(save_prefix="rps_beast"):
    print("Current working directory:", os.getcwd())
    ai = RealTimeEvolvingRPS_TF(save_prefix)
    plotter = RPSPlotter(ai)
    print("🎮 Evolving RPS AI + TensorFlow Ready!")
    print("AI suggests moves. **IMPORTANT:** report results from the AI's perspective:")
    print("  type 'w' if the AI WON, 'l' if the AI LOST, 'd' for draw. Commands: stats, reset, save, exit\n")
    while True:
        try:
            ai_move = ai.get_move()
        except Exception as e:
            print("⚠️ get_move error:", e)
            ai_move = random.choice(ai.MOVES)

        print(f"\n🤖 Play: {ai_move.upper()} | Confidence: {ai.prediction_confidence:.3f}")
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

        plotter.update_plot()

if __name__ == "__main__":
    main()
