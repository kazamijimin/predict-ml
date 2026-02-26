"""
Emotion Prediction Terminal Interface
Run this script to interact with the emotion prediction model.
"""

import os
import sys

# Add the parent directory to path to import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_model import EmotionModel


def print_banner():
    """Print the application banner."""
    print("\n" + "="*60)
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║          EMOTION PREDICTION SYSTEM                     ║
    ║                                                        ║ 
    ║   Enter text and I'll predict your emotion!           ║
    ║   Detects: Stress, Joy, Sadness, Anger, Fear,         ║
    ║            Surprise, Love, and Neutral                 ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    print("="*60 + "\n")


def print_result(emotion, probabilities, emotion_info):
    """Print the prediction results in a formatted way."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    color = emotion_info['color']
    
    print("\n" + "-"*50)
    print(f"{BOLD}PREDICTION RESULT{RESET}")
    print("-"*50)
    
    # Main emotion with color
    print(f"\n{color}{BOLD}Detected Emotion: {emotion.upper()}{RESET}")
    
    # Description and suggestion
    print(f"\n{emotion_info['description']}")
    print(f"\n💡 Suggestion: {emotion_info['suggestion']}")
    
    # Probability breakdown
    print(f"\n{BOLD}Confidence Levels:{RESET}")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for emo, prob in sorted_probs:
        bar_length = int(prob * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        indicator = " ← DETECTED" if emo == emotion else ""
        print(f"  {emo:12} [{bar}] {prob*100:5.1f}%{indicator}")
    
    print("-"*50 + "\n")


def check_stress_warning(emotion, probabilities):
    """Display special warning if concerning emotions are detected."""
    concerning_emotions = ['Stress', 'Sadness', 'Fear']
    
    if emotion in concerning_emotions:
        print("\n" + "!"*50)
        print("⚠️  MENTAL HEALTH CHECK ⚠️")
        print("!"*50)
        print("""
        We detected emotions that may indicate you're stressed.
        
        Here are some helpful suggestions:
        • Take 5 deep breaths (inhale 4 sec, hold 4 sec, exhale 4 sec)
        • Step away from your screen for a few minutes
        • Reach out to a friend, family member, or counselor
        • Write down what's bothering you
        • Consider professional support if feelings persist
        
        Remember: It's okay to not be okay. You are not alone.
        """)
        print("!"*50 + "\n")


def main():
    """Main function to run the emotion prediction interface."""
    print_banner()
    
    # Initialize the model
    model = EmotionModel()
    
    # Try to load existing model, or train if not available
    print("Initializing emotion model...")
    if not model.load_model():
        print("\nTraining new model...")
        model.train()
        print("\nModel is ready!")
    
    print("\nCommands:")
    print("  - Type any text to analyze emotions")
    print("  - Type 'quit' or 'exit' to close the program")
    print("  - Type 'retrain' to retrain the model")
    print("  - Type 'help' for more information")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("📝 Enter text to analyze: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Emotion Prediction System!")
                print("Take care of your mental health! Goodbye! 👋\n")
                break
            
            # Check for help command
            if user_input.lower() == 'help':
                print("""
                ╔═══════════════════════════════════════════════════════╗
                ║                       HELP                             ║
                ╠═══════════════════════════════════════════════════════╣
                ║ This system analyzes text to detect emotions.          ║
                ║                                                        ║
                ║ Supported Emotions:                                    ║
                ║   • STRESS - Work pressure, overwhelm                  ║
                ║   • JOY - Happiness, excitement, gratitude             ║
                ║   • SADNESS - Grief, feeling down, emptiness           ║
                ║   • ANGER - Frustration, irritation, rage              ║
                ║   • FEAR - Worry, anxiety, terror                      ║
                ║   • SURPRISE - Shock, amazement, unexpected            ║
                ║   • LOVE - Affection, appreciation, caring             ║
                ║   • NEUTRAL - Factual, calm statements                 ║
                ║                                                        ║
                ║ Commands:                                              ║
                ║   quit/exit/q - Close the program                     ║
                ║   retrain    - Retrain the model                      ║
                ║   help       - Show this message                      ║
                ╚═══════════════════════════════════════════════════════╝
                """)
                continue
            
            # Check for retrain command
            if user_input.lower() == 'retrain':
                print("\nRetraining the model...")
                model.train()
                print("Model retrained successfully!\n")
                continue
            
            # Skip empty input
            if not user_input:
                print("Please enter some text to analyze.\n")
                continue
            
            # Make prediction
            emotion, probabilities = model.predict(user_input)
            emotion_info = model.get_emotion_info(emotion)
            
            # Print results
            print_result(emotion, probabilities, emotion_info)
            
            # Check for negative emotions
            check_stress_warning(emotion, probabilities)
            
        except KeyboardInterrupt:
            print("\n\nProgram interrupted. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with different text.\n")


if __name__ == '__main__':
    main()
