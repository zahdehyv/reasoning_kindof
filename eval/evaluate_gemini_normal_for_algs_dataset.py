import os
import json
import time
import re
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from openai import api_key

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
            key=lambda x: int(''.join(filter(str.isdigit, x))
                              )  # Extraer números del nombre
        )
        for file_name in files:
            if not file_name.endswith('.json'):
                continue  # Skip non-JSON files
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf8') as f:
                data = json.load(f)
                if not root == 'data/logic':
                    assert len(data['questions']) == len(
                        data['answers']), f"Mismatch in {file_path}"
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
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        },
        system_instruction="""
        Muy Importante lo siguiente :Devuelve el resultado final en una lista de enteros donde el valor de la respuesta este entre los tags < answer > </answer > . """
    )

# Process questions and evaluate answers


def evaluate_model(model, questions, answers, options):
    correct = 0
    results = []
    print(answers)

    for idx, (question, answer) in enumerate(zip(questions, answers)):
        try:
            print(f"...processing question {idx+1}: {question}")
            chat = model.start_chat()
            time.sleep(8)
            if len(options) > 0:
                response = chat.send_message(f"""{question}\n
                {options[idx]}\n
                Answer each question separately between <answer></answer> tags, only with the corresponding INTEGER VALUE.
                """)
            else:
                response = chat.send_message(f"""{question}\n
                Answer each question separately between <answer></answer> tags, only with the corresponding INTEGER VALUE.
                """)

            # Extract the response text
            response_text = response.text.strip()

            patron = r"<answer>(.*?)</answer>"
            resultados = re.findall(patron, response_text)

            # Try to parse the response as a number or list of numbers
            try:
                selection = json.loads(resultados[0])

            except (json.JSONDecodeError, ValueError):
                selection = resultados[0]

            # Normalize to list for comparison
            if not isinstance(selection, list):
                selection = [selection]
            if not isinstance(answer, list):
                answer = [answer]

            # Convert tuples to lists for comparison
            selection = [list(item) if isinstance(item, tuple)
                         else item for item in selection]
            answer = [list(item) if isinstance(item, tuple)
                      else item for item in answer]

            # Evaluate response
            is_correct = True
            if len(selection) != len(answer):
                is_correct = False
            else:
                for inner_index in range(len(answer)):
                    try:
                        if float(selection[inner_index]) != float(answer[inner_index]):
                            is_correct = False
                            break
                    except (ValueError, TypeError):
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
    questions, answers, options = load_dataset("data/algs")
    model = initialize_model()

    # Run evaluation
    correct_count, detailed_results = evaluate_model(
        model, questions, answers, options)

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
            print(
                f"Q{i+1}: {status} Expected {result['expected']}, Got {result['received']}")
