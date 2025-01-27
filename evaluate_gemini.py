import os
import json
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

# Configure API key (debemos poner en la terminal una vez cargado el environment: !export GEMINI_API_KEY=<api key>)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    assert len(data['questions']) == len(data['answers']), "Mismatched questions/answers"
    return data['questions'], data['answers']

# Initialize model with function calling
def initialize_model():
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config={
            "temperature": 0.5,  # Reduced for more deterministic answers
            "max_output_tokens": 200,
        },
        tools=[genai.protos.Tool(
            function_declarations=[genai.protos.FunctionDeclaration(
                name="submit_answer",
                description="Submit the multiple-choice answer",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "selection": content.Schema(
                            type=content.Type.STRING,
                            description="The selected answer (A-D)"
                        )
                    },
                    required=["selection"]
                )
            )]
        )],
        tool_config={'function_calling_config': 'ANY'}
    )

# Process questions and evaluate answers
def evaluate_model(model, questions, answers):
    correct = 0
    results = []
    
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        try:
            chat = model.start_chat()
            response = chat.send_message(f"{question}\nAnswer with the correct letter only.")
            
            # Extract function call response
            selection = None
            for part in response.parts:
                if part.function_call and part.function_call.name == "submit_answer":
                    selection = part.function_call.args.get("selection", "").upper()
                    break
            
            # Evaluate response
            is_correct = selection == answer.upper()
            if is_correct:
                correct += 1
                
            results.append({
                "question": question,
                "expected": answer.upper(),
                "received": selection,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"Error processing question {idx+1}: {str(e)}")
            results.append({
                "question": question,
                "error": str(e)
            })
    
    return correct, results

# Main execution
if __name__ == "__main__":
    # Load data and model
    questions, answers = load_dataset("<aqui pondremos el path al dataset>")
    model = initialize_model()
    
    # Run evaluation
    correct_count, detailed_results = evaluate_model(model, questions, answers)
    
    # Print summary
    accuracy = (correct_count / len(questions)) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(questions)})")
    
    # Print detailed results
    print("\nDetailed Results:")
    for i, result in enumerate(detailed_results):
        if 'error' in result:
            print(f"Q{i+1}: ERROR - {result['error']}")
        else:
            status = "✓" if result['correct'] else "✗"
            print(f"Q{i+1}: {status} Expected {result['expected']}, Got {result['received']}")