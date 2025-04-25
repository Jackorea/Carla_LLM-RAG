# fusion_query.py

from openai import OpenAI
from image_llm import analyze_image
from query_data import query_rag

def main():
    print("🔥 fusion_query.py started")

    image_path = "Carla_image/Town01_001020.png"
    question = "Quel est le type de panneau de signalisation sur cette image ?"

    # ✅ Initialize client here
    client = OpenAI(
        api_key="FXMLo8lTdaSqsmDDRuJ3mT55xcJCryQy",
        base_url="https://llm.intellisphere.fr:9081/v1",
        timeout=60
    )
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    try:
        print("🔍 Analyse de l'image via LLM...")
        image_response = analyze_image(image_path, question, client, model_name)
        print("\n🧠 Réponse du modèle sur l’image :")
        print(image_response)

        print("\n📚 Recherche contextuelle via RAG...")
        rag_response = query_rag(question)
        print("\n📎 Réponse RAG :")
        print(rag_response)

    except Exception as e:
        print(f"💥 Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()
