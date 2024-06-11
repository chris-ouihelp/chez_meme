import base64

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def get_model(model_name="gpt-4o"):
    return ChatOpenAI(model=model_name).bind(response_format={"type": "json_object"})


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


PROMPT = """Dans le contexte de personnes agées ou handicapées qui souhaitent adapter leur habitat à leur situation

voici des liens qui détaillent le sujet :
Home safety environment
- https://www.cdc.gov/falls/index.html
- https://www.who.int/ageing/projects/age-friendly-environments/en/
- https://www.nia.nih.gov/health/home-safety-and-fall-prevention-older-adults
- [Personnes âgées et adaptation du logement](https://www.cairn.info/revue-gerontologie-et-societe1-2011-1-page-141.htm)

Ma prime Adapt’
- https://solidarites.gouv.fr/maprimeadapt-nouvelle-aide-pour-adapter-son-logement-la-perte-dautonomie

en utilisant ce contexte je vais te donner une image qui représente une pièce de maison avec divers équipements,
Répond moi seulement et uniquement avec un json au format suivant

{
  "room": "cuisine",
  "items": [
    {
      "item": "plan de travail",
      "recommendation": "Le plan de travail doit être à une hauteur accessible, éventuellement ajustable, pour les personnes en fauteuil roulant ou pour celles ayant des difficultés à se baisser.",
      "comment": "Il semble que le plan de travail soit à une hauteur standard. Il serait bénéfique de vérifier s'il est ajustable ou de le remplacer par un modèle réglable.",
      "status": "replace"
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
    image = encode_image("/Users/antoinebasset/Downloads/image3.jpeg")
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
