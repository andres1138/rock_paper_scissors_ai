# Rock Paper Scissors - Prediction AI

An advanced Rock Paper Scissors AI that learns your patterns and predicts your moves.

## 🎯 Features

- **Fast Pattern Recognition** - Detects sequences in just 5-10 rounds
- **Behavioral Analysis** - Understands human psychology (win-stay/lose-shift, anti-repetition bias)
- **Confidence-Based Predictions** - Only acts on high-confidence predictions
- **Smart Fallback** - Uses Nash equilibrium (random) when uncertain
- **No Repetition** - AI avoids spamming the same move
- **Real-time Insights** - See what patterns the AI detects in your play

## 🚀 Quick Start

### Get Move Suggestions (Coaching Mode)

The AI learns your opponent's patterns and tells you what to play:

```bash
python3 play_rps.py
```

**How it works:**
1. AI suggests a move for you (e.g., "Play ROCK")
2. You play that move against your opponent
3. You tell the AI if you won, lost, or drew (`w`/`l`/`d`)
4. AI learns and gets smarter!

For verbose mode (see AI's predictions about your opponent):
```bash
python3 play_rps.py --verbose
```

### Run Tests

Test the AI against different strategies:

```bash
# Test against all strategies
python3 test_ai.py --strategy all --rounds 100

# Test against specific strategy
python3 test_ai.py --strategy pattern --rounds 50

# Available strategies: random, pattern, frequency, win-stay, counter
```

## 📊 Performance

Tested over 100 rounds against different opponent types:

| Opponent Strategy | AI Win Rate | Description |
|------------------|-------------|-------------|
| **Pattern-based** | **93%** 🔥 | Repeating sequences (R-P-S-S-P) |
| **Frequency bias** | **73%** 🎯 | Favors rock 60% of the time |
| **Counter-AI** | **61%** ✅ | Tries to counter AI's patterns |
| **Random** | **38%** ⚖️ | Completely random (expected ~33%) |
| **Win-Stay** | **38%** ⚖️ | Repeats after winning, shifts after losing |

## 🧠 How It Works

The AI uses multiple prediction strategies ranked by confidence:

### 1. Pattern Detection (Primary)
- **N-gram sequences** - Detects patterns of length 2-4
- **Transition analysis** - What move follows what
- **Cycle detection** - Repeating sequences

### 2. Behavioral Modeling (Secondary)
- **Win-stay/Lose-shift** - How you respond to outcomes
- **Anti-repetition bias** - Humans avoid playing the same move repeatedly
- **Frequency analysis** - Which moves you favor

### 3. Nash Equilibrium (Fallback)
- Pure random when confidence < 40%
- Prevents exploitation

## 🎮 Example Session

```
Round 5
🤖 AI suggests you play: ROCK

What happened? (w=won / l=lost / d=draw): w
✅ YOU WIN! (Opponent played scissors)
🎯 AI predicted correctly!

Pattern detected: Opponent favors scissors after losing
Streak: You winning 3 in a row! 🔥

Score: You 9 | Losses 5 | Draws 1
```

## 📁 Files

- **`rps_predictor.py`** - Core AI engine (~400 lines)
- **`play_rps.py`** - AI Coach interface (suggests moves for you)
- **`test_ai.py`** - Test suite with multiple opponent strategies

## 🔧 Architecture

```python
class RPSPredictor:
    # Pattern Recognition
    - sequences: N-gram pattern storage
    - transitions: Markov chain probabilities
    
    # Behavioral Analysis
    - win_responses: How you react after winning
    - loss_responses: How you react after losing
    - move_preferences: Overall frequency bias
    
    # Prediction System
    - _predict_player_move(): Multi-strategy prediction
    - confidence scoring and voting
```

## 🆚 vs Previous Versions

### Why This is Better

| Old AI (rps_smart.py) | New AI (rps_predictor.py) |
|----------------------|---------------------------|
| 11+ complex strategies | 5 focused strategies |
| Reactive (learns after losing) | **Predictive** (predicts before you move) |
| 549 lines of complexity | 400 lines, clean and focused |
| Repeats same move | Smart anti-repetition |
| 50+ rounds to learn | **5-10 rounds to learn** |
| No confidence scoring | Confidence-based decisions |

### Test Comparison

Against pattern-based opponent (R-P-S-S-P, 100 rounds):

- **Old AI**: ~45-55% win rate
- **New AI**: **93% win rate** ✨

## 🎓 Tips for Using the AI Coach

1. **Follow the AI's suggestions** - The AI is learning your opponent's patterns
2. **Report accurately** - Tell the AI the real result (won/lost/drew)
3. **Watch the insights** - The AI tells you what patterns it detected in your opponent
4. **Give it time** - AI needs 5-10 rounds to learn patterns
5. **Use verbose mode** - See the AI's predictions and confidence levels
6. **Opponent changing strategy?** - The AI will adapt as it sees new patterns

## 🤝 Contributing

The AI is designed to be simple and extensible. Key areas for improvement:

- Add more behavioral patterns (e.g., streak-based responses)
- Implement meta-learning (learning which strategies work best)
- Add persistent state saving (currently stateless)

## 📝 License

MIT License - Feel free to use and modify!

---

Built with ❤️ using pattern recognition and behavioral analysis. No machine learning libraries required!