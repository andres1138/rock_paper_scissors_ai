#!/usr/bin/env python3
"""
RPS AI Coach v4.0 — Beast Mode

The AI learns your opponent's patterns and suggests what YOU should play.
Now powered by a multi-layer meta-strategy engine with 48 competing strategies.
"""

import sys
from rps_predictor import RPSPredictor


def confidence_indicator(confidence: float) -> str:
    """Return a color-coded confidence indicator."""
    if confidence >= 0.60:
        return "🟢 HIGH"
    elif confidence >= 0.30:
        return "🟡 MED"
    elif confidence > 0.0:
        return "🔴 LOW"
    else:
        return "⚪ RANDOM"


def print_header():
    """Print game header."""
    print("🔥" * 30)
    print("    ROCK PAPER SCISSORS — BEAST AI COACH v4.0")
    print("🔥" * 30)
    print()
    print("  48 strategies competing in parallel.")
    print("  Multi-level counter-prediction engine.")
    print("  Exponential decay for rapid adaptation.")
    print()
    print("How it works:")
    print("  1. AI suggests a move for you")
    print("  2. You play that move against your opponent")
    print("  3. You tell AI the result (w/l/d)")
    print("  4. All 48 strategies learn and compete!")
    print()
    print("Commands:")
    print("  w = you won       l = you lost       d = draw")
    print("  stats = show statistics")
    print("  board = show strategy leaderboard")
    print("  undo  = undo last round")
    print("  quit  = exit")
    print()


def print_stats(ai: RPSPredictor):
    """Print statistics."""
    stats = ai.get_stats()
    
    print("\n" + "=" * 60)
    print("📊 STATISTICS")
    print("=" * 60)
    print(f"Total Rounds:     {stats['total_rounds']}")
    print(f"Your Wins:        {stats['wins']}")
    print(f"Your Losses:      {stats['losses']}")
    print(f"Draws:            {stats['draws']}")
    print(f"Your Win Rate:    {stats['win_rate']:.1f}%")
    
    if stats['current_streak'] > 0:
        print(f"Current Streak:   You winning {stats['current_streak']} in a row! 🔥")
    elif stats['current_streak'] < 0:
        print(f"Current Streak:   Losing {abs(stats['current_streak'])} in a row")
    
    print(f"\nActive Strategy:  {stats.get('active_strategy', 'N/A')}")
    print(f"Opponent Style:   {stats['detected_behavior']}")
    
    insights = ai.get_insights()
    if insights:
        print("\n💡 INSIGHTS:")
        for insight in insights:
            if insight:
                print(f"   • {insight}")
    
    print("=" * 60)


def print_leaderboard(ai: RPSPredictor):
    """Print strategy leaderboard."""
    leaderboard = ai.get_strategy_leaderboard(top_n=8)
    
    print("\n" + "=" * 60)
    print("🏆 STRATEGY LEADERBOARD")
    print("=" * 60)
    
    meta_labels = {0: "P0-direct", 1: "P1-counter²", 2: "P2-counter³"}
    
    for i, entry in enumerate(leaderboard):
        rank = i + 1
        score_bar = "█" * max(0, int(entry['score'] * 2))
        meta = meta_labels.get(entry['meta_level'], '?')
        print(f"  #{rank} {entry['name']:28s} score: {entry['score']:+6.1f}  [{meta}] {score_bar}")
    
    print("=" * 60)


def main():
    """Main game loop."""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print_header()
    
    ai = RPSPredictor()
    round_num = 0
    undo_stack = []  # For undo support
    
    while True:
        round_num += 1
        print(f"\n{'─' * 60}")
        print(f"  ROUND {round_num}")
        print(f"{'─' * 60}")
        
        # AI suggests a move
        suggested_move = ai.get_move()
        conf = ai.prediction_confidence
        conf_label = confidence_indicator(conf)
        
        print(f"\n  🤖 Play: {suggested_move.upper()}  [{conf_label}]")
        
        if verbose and ai.last_prediction:
            print(f"     Predicts opponent: {ai.last_prediction.upper()}")
            print(f"     Strategy: {ai.active_strategy_name}")
        
        # Get result
        result_input = input("\n  Result? (w/l/d): ").strip().lower()
        
        # Handle commands
        if result_input in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for playing!")
            ai.save_brain()
            print_stats(ai)
            break
        
        if result_input == 'stats':
            print_stats(ai)
            round_num -= 1
            continue
        
        if result_input == 'board':
            print_leaderboard(ai)
            round_num -= 1
            continue
        
        if result_input == 'undo':
            if undo_stack:
                last = undo_stack.pop()
                # Reverse the last round's effects
                if ai.opp_history:
                    ai.opp_history.pop()
                if ai.my_history:
                    ai.my_history.pop()
                if ai.results:
                    last_result = ai.results.pop()
                    ai.total_rounds -= 1
                    if last_result == 'w':
                        ai.wins -= 1
                    elif last_result == 'l':
                        ai.losses -= 1
                    else:
                        ai.draws -= 1
                round_num -= 2  # Will be incremented at top of loop
                print("  ↩️  Last round undone!")
            else:
                print("  ❌ Nothing to undo")
                round_num -= 1
            continue
        
        if result_input not in ['w', 'l', 'd']:
            print("  ❌ Invalid! Use w, l, or d")
            round_num -= 1
            continue
        
        # Infer opponent's move from result
        if result_input == 'w':
            opponent_move = ai.BEATS[suggested_move]
            print(f"  ✅ YOU WIN! (Opponent: {opponent_move})")
        elif result_input == 'l':
            opponent_move = ai.BEATEN_BY[suggested_move]
            print(f"  ❌ YOU LOSE (Opponent: {opponent_move})")
        else:
            opponent_move = suggested_move
            print(f"  🤝 DRAW (Opponent: {opponent_move})")
        
        # Save undo info
        undo_stack.append({
            'opponent_move': opponent_move,
            'ai_move': suggested_move,
            'result_input': result_input
        })
        
        # Record the round
        ai.record_round(opponent_move, suggested_move)
        
        # Show prediction hit
        if verbose and ai.last_prediction_correct is not None:
            if ai.last_prediction_correct:
                print("  🎯 Prediction correct!")
            else:
                print("  ❌ Prediction missed")
        
        # Periodic stats
        if round_num % 5 == 0:
            stats = ai.get_stats()
            print(f"\n  📈 {stats['wins']}-{stats['losses']}-{stats['draws']} "
                  f"(Win Rate: {stats['win_rate']:.1f}%) "
                  f"| Strategy: {ai.active_strategy_name}")
            
            if ai.save_brain():
                print("  💾 Brain saved!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
