import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# Mock data for testing
def get_mock_current_quiz():
    """Mock data for the current quiz."""
    return {
        "questions": [
            {"id": 1, "topic": "Biology", "difficulty": "Easy", "selected_option": 2, "correct_option": 2},
            {"id": 2, "topic": "Chemistry", "difficulty": "Medium", "selected_option": 3, "correct_option": 1},
            {"id": 3, "topic": "Physics", "difficulty": "Hard", "selected_option": 1, "correct_option": 1},
        ]
    }

def get_mock_historical_quizzes():
    """Mock data for historical quizzes."""
    return [
        {
            "quiz_id": 1,
            "questions": [
                {"id": 1, "topic": "Biology", "difficulty": "Easy", "selected_option": 2, "correct_option": 2},
                {"id": 2, "topic": "Chemistry", "difficulty": "Medium", "selected_option": 3, "correct_option": 1},
                {"id": 3, "topic": "Physics", "difficulty": "Hard", "selected_option": 1, "correct_option": 1},
            ]
        },
        {
            "quiz_id": 2,
            "questions": [
                {"id": 4, "topic": "Biology", "difficulty": "Easy", "selected_option": 1, "correct_option": 2},
                {"id": 5, "topic": "Chemistry", "difficulty": "Medium", "selected_option": 2, "correct_option": 2},
                {"id": 6, "topic": "Physics", "difficulty": "Hard", "selected_option": 3, "correct_option": 3},
            ]
        },
    ]

# Data analysis functions
def calculate_accuracy_by_topic(df):
    """Calculate accuracy by topic."""
    df["is_correct"] = df["selected_option"] == df["correct_option"]
    return df.groupby("topic")["is_correct"].mean().reset_index()

def calculate_accuracy_by_difficulty(df):
    """Calculate accuracy by difficulty level."""
    df["is_correct"] = df["selected_option"] == df["correct_option"]
    return df.groupby("difficulty")["is_correct"].mean().reset_index()

def identify_weak_areas(topic_accuracy, difficulty_accuracy):
    """Identify weak topics and difficulty levels."""
    weak_topics = topic_accuracy[topic_accuracy["is_correct"] < 0.5]
    weak_difficulties = difficulty_accuracy[difficulty_accuracy["is_correct"] < 0.5]
    return weak_topics, weak_difficulties

def generate_recommendations(weak_topics, weak_difficulties):
    """Generate personalized recommendations."""
    recommendations = []
    if not weak_topics.empty:
        recommendations.append(f"Focus on improving in these topics: {', '.join(weak_topics['topic'])}.")
    if not weak_difficulties.empty:
        recommendations.append(f"Practice more {', '.join(weak_difficulties['difficulty'])} level questions.")
    return recommendations

# Bonus: Student Persona
def define_student_persona(topic_accuracy):
    """Define a student persona based on topic performance."""
    best_topic = topic_accuracy.loc[topic_accuracy["is_correct"].idxmax()]["topic"]
    worst_topic = topic_accuracy.loc[topic_accuracy["is_correct"].idxmin()]["topic"]
    return f"Your strengths lie in {best_topic}, but you need to work on {worst_topic}."

# Bonus: Rank Prediction (Placeholder)
def predict_neet_rank(quiz_scores, historical_neet_data):
    """Predict NEET rank based on quiz performance."""
    X = np.array(quiz_scores).reshape(-1, 1)  # Reshape to 2D array
    y = historical_neet_data["rank"]  # Target (NEET rank)
    model = LogisticRegression()
    model.fit(X, y)
    return model.predict(X[-1].reshape(1, -1))  # Predict rank for the latest quiz

# Main function
def main():
    # Fetch mock data
    current_quiz_data = get_mock_current_quiz()
    historical_quiz_data = get_mock_historical_quizzes()

    # Convert to DataFrames
    current_df = pd.DataFrame(current_quiz_data["questions"])
    historical_df = pd.concat([pd.DataFrame(quiz["questions"]) for quiz in historical_quiz_data])

    # Analyze data
    topic_accuracy = calculate_accuracy_by_topic(historical_df)
    difficulty_accuracy = calculate_accuracy_by_difficulty(historical_df)

    # Identify weak areas
    weak_topics, weak_difficulties = identify_weak_areas(topic_accuracy, difficulty_accuracy)

    # Generate recommendations
    recommendations = generate_recommendations(weak_topics, weak_difficulties)

    # Bonus: Define student persona
    persona = define_student_persona(topic_accuracy)

    # Bonus: Predict NEET rank (placeholder)
    quiz_scores = [0.8, 0.7, 0.6]  # Example quiz scores
    historical_neet_data = pd.DataFrame({"rank": [100, 200, 300]})  # Example historical data
    predicted_rank = predict_neet_rank(quiz_scores, historical_neet_data)

    # Output results
    print("===== Personalized Recommendations =====")
    for rec in recommendations:
        print(rec)
    print("\n===== Student Persona =====")
    print(persona)
    print("\n===== Predicted NEET Rank =====")
    print(f"Your predicted rank is: {predicted_rank[0]}")

# Run the script
if __name__ == "__main__":
    main()