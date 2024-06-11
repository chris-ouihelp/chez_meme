import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def get_model(model_name="gpt-4o"):
    return ChatOpenAI(model=model_name)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


PROMPT = """en accord avec "ma prime adapte" française, je vais te donner une image qui 
représente une pièce d'une salle de bain avec divers équipements, je voudrais que 
tu me répondes uniquement en me renvoyant un raw json au format suivant avec 
les booleans qui sont à true si l'équipement te parait manquant et false 
si il est déja présent. Voici le format pour les différentes pieces, tu 
dois uniquement m'en renvoyé un d'entre eux en fonction de la piece que tu 
reconnais : 
- salle de bain : 
{
    italian_showers: boolean,
    raised_toilets: boolean,
    grab_bars_and_handrails: boolean,
    automatic_lighting_systems: boolean,
    non_slip_flooring: boolean
  }"""

if __name__ == "__main__":
    image = encode_image("/Users/chtr/Dev/ouihelp/starter-genai/salle_de_bain.png")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT,
            },
            {
                "type": "image_url",
                # "image_url": image_input_data,
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            },
        ],
    )
    model = get_model()
    response = model.invoke(
        [message],
    )
    breakpoint()
    print(response)
