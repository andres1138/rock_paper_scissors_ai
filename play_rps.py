#!/usr/bin/env python3
"""
RPS AI Coach - Get move suggestions and report results

The AI learns your opponent's patterns and suggests what YOU should play.
"""

import sys
from rps_predictor import RPSPredictor


def print_header():
    """Print game header."""
    print("🎯" * 30)
    print("      ROCK PAPER SCISSORS - AI Coach")
    print("🎯" * 30)
    print()
    print("The AI will suggest moves for YOU to play.")
    print("After each round, tell the AI if you won, lost, or drew.")
    print()
    print("How it works:")
    print("  1. AI suggests a move for you")
    print("  2. You play that move against your opponent")
    print("  3. You tell AI the result (w/l/d)")
    print("  4. AI learns and gets smarter!")
    print()
    print("Commands:")
    print("  w = you won")
    print("  l = you lost")
    print("  d = draw")
    print("  stats = show statistics")
    print("  quit = exit")
    print()


def print_stats(ai: RPSPredictor):
    """Print statistics."""
    stats = ai.get_stats()
    
    print("\n" + "="*60)
    print("📊 STATISTICS")
    print("="*60)
    print(f"Total Rounds:    {stats['total_rounds']}")
    print(f"Your Wins:       {stats['wins']}")
    print(f"Your Losses:     {stats['losses']}")
    print(f"Draws:           {stats['draws']}")
    print(f"Your Win Rate:   {stats['win_rate']:.1f}%")
    
    if stats['current_streak'] > 0:
        print(f"Current Streak:  You winning {stats['current_streak']} in a row! 🔥")
    elif stats['current_streak'] < 0:
        print(f"Current Streak:  Losing {abs(stats['current_streak'])} in a row")
    
    print(f"\nOpponent Style:  {stats['detected_behavior']}")
    
    insights = ai.get_insights()
    if insights:
        print("\n💡 INSIGHTS ABOUT YOUR OPPONENT:")
        for insight in insights:
            if insight:
                print(f"   • {insight}")
    
    print("="*60)


def main():
    """Main game loop."""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print_header()
    
    ai = RPSPredictor()
    round_num = 0
    
    while True:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")
        
        # AI suggests a move
        suggested_move = ai.get_move()
        
        print(f"\n🤖 AI suggests you play: {suggested_move.upper()}")
        
        if verbose and ai.prediction_confidence > 0:
            print(f"   (AI predicts opponent will play: {ai.last_prediction.upper()}, confidence: {ai.prediction_confidence:.0%})")
        
        print("\n   Play this move against your opponent!")
        
        # Get result from user
        result_input = input("\nWhat happened? (w=won / l=lost / d=draw): ").strip().lower()
        
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
        
        # Validate result
        if result_input not in ['w', 'l', 'd']:
            print("❌ Invalid input! Use w, l, or d")
            round_num -= 1
            continue
        
        # Infer opponent's move from result
        opponent_move = None
        if result_input == 'w':
            # We won, so opponent played what our move beats
            opponent_move = ai.BEATS[suggested_move]
            print(f"✅ YOU WIN! (Opponent played {opponent_move})")
        elif result_input == 'l':
            # We lost, so opponent played what beats our move
            opponent_move = ai.BEATEN_BY[suggested_move]
            print(f"❌ YOU LOSE (Opponent played {opponent_move})")
        else:  # draw
            # We drew, so opponent played same move
            opponent_move = suggested_move
            print(f"🤝 DRAW (Opponent played {opponent_move})")
        
        # Record the round (from AI's perspective, it suggested the move)
        ai.record_round(opponent_move, suggested_move)
        
        # Show prediction accuracy
        if ai.last_prediction and verbose:
            if ai.last_prediction_correct:
                print("   🎯 AI predicted correctly!")
            else:
                print("   ❌ AI prediction was off")
        
        # Show stats every 5 rounds
        if round_num % 5 == 0:
            stats = ai.get_stats()
            print(f"\n📈 Quick Stats: {stats['wins']}-{stats['losses']}-{stats['draws']} (Win Rate: {stats['win_rate']:.1f}%)")
            
            insights = ai.get_insights()
            if insights and insights[0]:
                print(f"💡 {insights[0]}")
            
            # Save AI's learning
            if ai.save_brain():
                print("💾 Learning saved!")



if __name__ == "__main__":
    ai = None
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Game interrupted. Goodbye!")
        # Try to save if AI was created
        try:
            if 'ai' in locals():
                ai.save_brain()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
