import base64
import io
import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def get_model(model_name="gpt-4o"):
    return ChatOpenAI(model=model_name).bind(response_format={"type": "json_object"})


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Fonction pour convertir une image en base64
def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


PROMPT = """Dans le contexte de personnes agées ou handicapées qui souhaitent adapter leur habitat à leur situation

voici des liens qui détaillent le sujet :
Home safety environment
- https://www.cdc.gov/falls/index.html
- https://www.who.int/ageing/projects/age-friendly-environments/en/
- https://www.nia.nih.gov/health/home-safety-and-fall-prevention-older-adults
- [Personnes âgées et adaptation du logement](https://www.cairn.info/revue-gerontologie-et-societe1-2011-1-page-141.htm)

Ma prime Adapt’
- https://solidarites.gouv.fr/maprimeadapt-nouvelle-aide-pour-adapter-son-logement-la-perte-dautonomie

en utilisant ce contexte je vais te donner des images qui représentent des pièces de maison avec divers équipements,
je souhaite avoir un maximum d'amélioration par pièce
Répond moi seulement et uniquement avec un json au format suivant

{
  "rooms": [
    {
        "room": "cuisine",
        "items": [
            {
            "item": "plan de travail",
            "recommendation": "Le plan de travail doit être à une hauteur accessible, éventuellement ajustable, pour les personnes en fauteuil roulant ou pour celles ayant des difficultés à se baisser.",
            "comment": "Il semble que le plan de travail soit à une hauteur standard. Il serait bénéfique de vérifier s'il est ajustable ou de le remplacer par un modèle réglable.",
            "status": "replace"
            },
        ]
    },
}

avec 

"room" : le type de pièce dans la maison {chambre, cuisine, salle de bain, ...}
"item" : un objet précis
"recommendation" : un texte explication sur l'utilité
"comment" : ton avis sur le changement à faire
"status" : qui vaut ok si l’élément est présent et bien adapter, replace si l’élément à besoin d’être adapter ou changer, absent s’il n’est pas présent et idk si tu ne sais pas {ok, replace, absent, idk},
"""

if __name__ == "__main__":

    base_message = [
            {
                "type": "text",
                "text": PROMPT,
            }
        ]

    image = encode_image("/Users/antoinebasset/Downloads/image3.jpeg")
    images = [
        "/Users/antoinebasset/Downloads/image3.jpeg",
        "/Users/antoinebasset/Downloads/image2.jpg",
        "/Users/antoinebasset/Downloads/image.png",
        "/Users/antoinebasset/Downloads/image4.jpeg"
    ]
    
    for image_path in images:
        image_base64 = image_to_base64(image_path)
        base_message.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}           
            },
        )

    message = HumanMessage(base_message)

    model = get_model()
    response = model.invoke(
        [message],
    )
    print(response.pretty_print())
