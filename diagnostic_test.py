"""
Diagnostic script to test if AI has inverted logic bug.
"""

from rps_monster import MonsterRPS

# Create fresh AI
ai = MonsterRPS(save_prefix="diagnostic_test")

print("Testing AI logic...")
print("="*60)

# Manually test predictions
print("\n1. Testing BEATS logic:")
print(f"   rock BEATS {ai.BEATS['rock']}")  # Should be scissors
print(f"   paper BEATS {ai.BEATS['paper']}")  # Should be rock
print(f"   scissors BEATS {ai.BEATS['scissors']}")  # Should be paper

print("\n2. Testing BEATEN_BY logic:")
print(f"   rock BEATEN_BY {ai.BEATEN_BY['rock']}")  # Should be paper
print(f"   paper BEATEN_BY {ai.BEATEN_BY['paper']}")  # Should be scissors
print(f"   scissors BEATEN_BY {ai.BEATEN_BY['scissors']}")  # Should be rock

print("\n3. Testing move_result logic:")
# AI plays rock, opponent plays scissors -> AI wins
result = ai.move_result('rock', 'scissors')
print(f"   AI:rock vs OPP:scissors -> {result} (should be 'w')")

# AI plays rock, opponent plays paper -> AI loses
result = ai.move_result('rock', 'paper')
print(f"   AI:rock vs OPP:paper -> {result} (should be 'l')")

# AI plays rock, opponent plays rock -> draw
result = ai.move_result('rock', 'rock')
print(f"   AI:rock vs OPP:rock -> {result} (should be 'd')")

print("\n4. Testing frequency prediction logic:")
# Feed opponent history: paper, paper, paper
for _ in range(5):
    ai.opponent_history.append('paper')

# Frequency should predict paper, so should return what beats paper (scissors)
prediction = ai._predict_frequency_recent()
print(f"   Opponent played: paper x5")
print(f"   AI predicts opponent will play: paper")
print(f"   AI should play: scissors (to beat paper)")
print(f"   AI actually plays: {prediction}")

if prediction == 'scissors':
    print("   ✅ CORRECT!")
else:
    print(f"   ❌ BUG! Should be scissors, got {prediction}")

print("\n5. Testing if AI learns correctly:")
# Simulate: AI plays rock, opponent plays scissors (AI wins)
ai_move = 'rock'
opp_move = 'scissors'
result = ai.move_result(ai_move, opp_move)
print(f"   Test: AI plays {ai_move}, opponent plays {opp_move}")
print(f"   Result: {result} (should be 'w')")

# Now update the AI
ai.update(ai_move, result)

# Check if it learned correctly
print(f"   AI history last: {ai.ai_history[-1]} (should be 'rock')")
print(f"   Opponent history last: {ai.opponent_history[-1]} (should be 'scissors')")
print(f"   Results last: {ai.results[-1]} (should be 'w')")

if (ai.ai_history[-1] == ai_move and 
    ai.opponent_history[-1] == opp_move and 
    ai.results[-1] == result):
    print("   ✅ Learning is CORRECT!")
else:
    print("   ❌ BUG in learning!")

print("\n" + "="*60)
print("Diagnostic complete. Check results above for bugs.")
