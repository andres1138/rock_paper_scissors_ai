# Rock Paper Scissors AI - Persistence Feature

## 🧠 AI Brain Saves Automatically!

Your AI now **remembers** everything it learns between sessions. It saves its "brain" (all learned patterns) to `rps_ai_brain.pkl`.

## How It Works

### Auto-Save
The AI automatically saves every **5 rounds** and when you **quit**:
- Pattern sequences (n-grams)
- Transition probabilities (Markov chains)  
- Behavioral responses (Win-Stay, Lose-Shift)
- Move frequency preferences
- Lifetime statistics

### Auto-Load
When you start a new game, the AI automatically loads its previous learning:
```
🧠 AI Brain loaded! Previously learned from 150 rounds.
   Lifetime stats: 95-42-13
```

## Commands

### Play with persistent learning
```bash
python3 play_rps.py
```

The AI will load its brain and continue getting smarter!

### Reset the AI's brain
If you want the AI to forget everything and start fresh:

```python
from rps_predictor import RPSPredictor

ai = RPSPredictor()
ai.reset_brain()  # Forgets everything, deletes save file
```

### Use custom save file
```python
ai = RPSPredictor(save_file="my_custom_brain.pkl")
```

## Example Session

**First time playing:**
```
🤖 AI suggests you play: ROCK
(No previous learning)
```

**After 20 rounds, you quit:**
```
💾 Learning saved!
```

**Next time you play:**
```
🧠 AI Brain loaded! Previously learned from 20 rounds.
   Lifetime stats: 12-6-2

🤖 AI suggests you play: PAPER
   (AI predicts opponent will play: ROCK, confidence: 75%)
```

The AI **remembers** your opponent's patterns!

## Benefits

✅ **Gets smarter over time** - Each game makes it better  
✅ **Learns your style** - Adapts to how you play  
✅ **Persistent knowledge** - Never forgets patterns  
✅ **No manual saves** - Automatic after every 5 rounds  

## File Location

The brain file `rps_ai_brain.pkl` is saved in the same directory as `play_rps.py`.

You can delete this file to reset the AI's memory.
