import os
import json
import time

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

# Configure API key (debemos poner en la terminal una vez cargado el environment: !export GEMINI_API_KEY=<api key>)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Load dataset
def load_dataset(dir_path):
    all_questions = []
    all_answers = []
    all_options = []
    for root, _, files in os.walk(dir_path):
        # Ordenar archivos numéricamente
        files = sorted(
            [f for f in files if f.endswith('.json')],  # Filtrar solo JSON
            key=lambda x: int(''.join(filter(str.isdigit, x)))  # Extraer números del nombre
        )
        for file_name in files:
            if not file_name.endswith('.json'):
                continue  # Skip non-JSON files
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
                if not root == 'data/logic':
                    assert len(data['questions']) == len(data['answers']), f"Mismatch in {file_path}"
                all_questions.append(data['questions'])
                all_answers.append(data['answers'])
                if 'options' in data.keys():
                    all_options.append(data['options'])
    return all_questions, all_answers, all_options

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
                description="Submit the answer",
                parameters=content.Schema(
                    type=content.Type.OBJECT,
                    properties={
                        "selection": content.Schema(
                            type=content.Type.ARRAY,
                            description="The selected(ed) answer(s) (a number or an array of numbers depending on wether it's one question or an array of questions)",
                            items=content.Schema(
                                type=content.Type.INTEGER,
                            ),
                        )
                    },
                    required=["selection"]
                )
            )]
        )],
        tool_config={'function_calling_config': 'ANY'}
    )

# Process questions and evaluate answers
def evaluate_model(model, questions, answers, options):
    correct = 0
    results = []
    print(answers)
    
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        try:
            chat = model.start_chat()
            time.sleep(1)
            if len(options) > 0:
                response = chat.send_message(f"""{question}\n
                {options[idx]}\n
                Answer with the correct number or the array with the correct numbers.
                """)
            else:
                response = chat.send_message(f"""{question}\n
                Answer with the correct number or the array with the correct numbers.
                """)
            
            # Extract function call response
            selection = None
            for part in response.parts:
                if part.function_call and part.function_call.name == "submit_answer":
                    selection = part.function_call.args.get("selection", "")
                    break
            print(selection, answer)
            # Evaluate response
            if answer is int or answer is float:
                is_correct = selection - answer == 0
            else:
                is_correct = True
                for inner_index in range(len(answer)):
                    if not answer[inner_index] - selection[inner_index] == 0:
                        is_correct = False
                        break

            if is_correct:
                correct += 1
                
            results.append({
                "question": question,
                "expected": answer,
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
    questions, answers, options = load_dataset("data/logic")
    model = initialize_model()
    
    # Run evaluation
    correct_count, detailed_results = evaluate_model(model, questions, answers, options)
    
    # Print summary
    accuracy = (correct_count / len(questions)) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(questions)})")
    
    # Print detailed results
    print("\nDetailed Results:")
    for i, result in enumerate(detailed_results):
        if 'error' in result:
            print(f"Q{i+1}: ERROR - {result['error']}")
        else:
            status = "✓ OK" if result['correct'] else "✗ FAIL"
            print(f"Q{i+1}: {status} Expected {result['expected']}, Got {result['received']}")