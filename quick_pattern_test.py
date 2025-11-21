"""
Quick pattern test - verify Monster AI can crush simple patterns.
"""

from rps_monster import MonsterRPS

def quick_pattern_test(rounds=100):
    print("🔥 QUICK PATTERN TEST 🔥\n")
    print("Testing against Rock-Paper-Scissors repeating pattern")
    print(f"Rounds: {rounds}\n")
    
    ai = MonsterRPS(save_prefix="quick_pattern_test")
    pattern = ['rock', 'paper', 'scissors']
    
    results = {'w': 0, 'l': 0, 'd': 0}
    
    for i in range(rounds):
        ai_move = ai.get_move()
        opp_move = pattern[i % 3]
        
        result = ai.move_result(ai_move, opp_move)
        results[result] += 1
        
        ai.update(ai_move, result)
        
        # Show progress every 20 rounds
        if (i + 1) % 20 == 0:
            wr = results['w'] / (i + 1) * 100
            print(f"Round {i+1:3d}: WR = {wr:5.1f}% | Archetype: {ai.opponent_archetype} | Patterns: {len(ai.patterns)}")
    
    # Final stats
    total = rounds
    wr = results['w'] / total * 100
    
    # Last 20 rounds win rate (after pattern should be detected)
    recent = list(ai.results)[-20:]
    recent_wr = (recent.count('w') / len(recent) * 100) if recent else 0
    
    print("\n" + "="*60)
    print(f"FINAL: W:{results['w']} L:{results['l']} D:{results['d']}")
    print(f"Overall WR: {wr:.1f}%")
    print(f"Last 20 WR: {recent_wr:.1f}%")
    print(f"Opponent Type: {ai.opponent_archetype} (conf: {ai.archetype_confidence:.2f})")
    print(f"Patterns Detected: {len(ai.patterns)}")
    print("="*60)
    
    if recent_wr >= 70:
        print("✅ CRUSHING IT! Pattern fully exploited! 🔥")
    elif wr >= 55:
        print("✅ GOOD! Pattern detected and exploited.")
    elif wr >= 40:
        print("⚠️ Pattern detected but exploitation needs work.")
    else:
        print("❌ Failed to exploit simple pattern.")
    
    return wr

if __name__ == "__main__":
    quick_pattern_test(100)
