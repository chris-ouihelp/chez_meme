import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def get_model(model_name="gpt-4o"):
    return ChatOpenAI(model=model_name).bind(response_format={"type": "json_object"})


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


PROMPT = """Dans le contexte de bénéficiare étant des personnes agées ou handicapées qui souhaitent adapté leur salle de bain à leur situation
en accord avec "ma prime adapt" française.

voici des liens qui détaillent le sujet :
Home safety environment
- https://www.cdc.gov/falls/index.html
- https://www.who.int/ageing/projects/age-friendly-environments/en/
- https://www.nia.nih.gov/health/home-safety-and-fall-prevention-older-adults
- [Personnes âgées et adaptation du logement](https://www.cairn.info/revue-gerontologie-et-societe1-2011-1-page-141.htm)

Ma prime Adapt’
- https://solidarites.gouv.fr/maprimeadapt-nouvelle-aide-pour-adapter-son-logement-la-perte-dautonomie

en utilisant ce contexte je vais te donner une image qui représente une salle de bain avec divers équipements,
Répond moi seulement et uniquement avec un json au format suivant

{
  "room": [
    {
      "item": "Grab bars",
      "recommendation": "Install grab bars near the bathtub and next to the toilet to help with balance and stabx@ility.",
      "status": "idk"
    },
}

avec

"room" : qui correspond à une pièce de maison que tu auras reconnu
"item" : un objet précis
"recommendation" : un texte explication sur l'utilité
"status" : qui vaut yes si l'objet est présent dans l'image, no s'il ne l'est pas et idk si tu ne sais pas {yes, no, idk}"""

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
    print(response.pretty_print())
