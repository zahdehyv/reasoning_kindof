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
        ejemplo positivo:
        pregunta:
        En la Isla de los truhanes y los caballeros los habitantes A, B y C hacen las siguientes
declaraciones:
A: B es caballero
B: Si A es caballero entonces C tambien lo es.
Determine que son A, B y C. Demuestrelo formalmente.

Buen razonamiento:

Tenemos como premisa que en la isla de truhanes y caballeros toda persona solo tiene dos posibilidades, ser truhan o ser caballero, por tanto si eres truhan entonces no eres caballero y si eres caballero entonces no eres truhan.

Tenemos tambien segun el ejercicio los siguientes planteamientos los cuales seran nuestras actuales premisas ( una premisa es como un axioma, es algo que es irrefutablemente verdad en dicho contexto ):
Planteamiento 1 : A dice que B es caballero.
Planteamiento 2 : B dice que Si A es caballero entonces C tambien lo es. 

Un primer acercamiento a la respuesta podria ser pensar que sucederia si A fuera caballero. Por tanto asumiremos que A es caballero:
como A dice que B es caballero en el Planteamiento 1, y tenemos que los caballeros dicen la verdad ,entonces dado que A es caballero, lo que dice es verdad.
Por tanto tenemos como verdadero el hecho de que B es caballero y tambien tenemos que A es caballero ya que fue asumido.

Ahora dado que B es caballero ,lo que dice es verdad, luego lo que dice es que " si A es caballero entonces C tambien lo es " en el Planteamiento 2,entonces al tener dicha proposicion, como tenemos que A es caballero ,podemos decir que C tambien es caballero.
Luego llegamos a que todos son caballero A ,B y C. Pero lamentablemente no demostramos el ejercicio ya que tenemos que llegar a una contradiccion para poder demostrar realmente algo. Lo que hicimos fue asumir un conjunto de cosas y llegar a que nada se rompe, es decir no llegamos a ninguna contradiccion. 

Entonces la informacion que podemos sacar de todo esto es que no debemos volver a asumir que A es caballero ya que no llegamos a ninguna contradiccion, ojo, eso no significa que A no sea realmente caballero, esto solo significa que si asumimos que A es caballero inicialmente, no llegaremos a nada contundente.

Luego vamos a asumir que A es truhan ,quizas lleguemos a una contradiccion:

Asumamos que A es truhan, luego como los truhanes dicen mentira , lo que dice A es mentira lo que significa que el significado contrario de lo que dice es verdad.
A dice que B es caballero por tanto como A es truhan, entonces B no es caballero. Pero si B no es caballero entonces B es truhan ya que no puede ser ninguna otra cosa.
Luego aplicamos la misma ideologia, lo que dice B en el Planteamiento 2 es falso o mentira, luego lo que dice es "Si A es caballero entonces C tambien lo es " es falso , pero ¿cuándo es falsa esa afirmacion? Pues es falsa si A es caballero y C no lo es, por tanto tenemos que A es caballero y C no es caballero, que es lo mismo que A es caballero y C es truhan, pero acabamos de llegar a una contradiccion porque habiamos asumido que A es truhan y ahora llegamos a que A es caballero lo cual es una contradiccion . Al llegar a una contradiccion podemos decir que lo que asumimos es falso, como lo que asumimos es A es truhan, entonces A es truhan es falso ,luego A no es truhan que es lo mismo que A es caballero.

Luego llegamos a que A es caballero,¿significa que llegamoss a que A es caballero ? Pues al llegar a una contradiccion, podemos agregar lo contrario que asumimos que llego a la contradiccion a nuestras premisas las cuales siempre seran verdad en nuestro contexto.
Por tanto tenemos ahora en nuestras premisas Planteamiento 1, Planteamiento 2 y A es caballero.

Nos falta demostrar que es B y C.

como A es caballero, lo que dice es verdad ,luego lo que dice es que B es caballero en el Planteamiento 1.
Luego tenemos que B es caballero,y como lo que dice es verdad, lo que dice es que si A es caballero entonces C tambien lo es en el Planteamiento 2, luego como A es caballero , tenemos que C es caballero tambien.

Luego demostramos que A es caballero, B es caballero y C es caballero.

        Muy Importante lo siguiente :Devuelve el resultado final en una lista de enteros donde el valor de la respuesta este entre los tags < answer > </answer > . Seleccionar multiples opciones como respuesta esta mal, solo una de las opciones es la respuesta correcta """
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
    questions, answers, options = load_dataset("data/logic")
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
