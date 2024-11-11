import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
import uuid
import os


class GameReviewSummarizer:
    def __init__(self, nltk_download=False, qdrant_url="http://localhost:6333", collection_name="game_reviews", qdrant_api_key=None, openai_api_key=None):
        self.initialize_nltk(nltk_download)
        self.summarizer = TextRankSummarizer()
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.qdrant_api_key = qdrant_api_key
        self.openai_api_key = openai_api_key
        self.tokenizer, self.model = self.load_embedding_model()

    def initialize_nltk(self, download):
        if download:
            nltk.download('punkt')

    def load_embedding_model(self):
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
        return tokenizer, model

    def scrape(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            p_tags = soup.find_all('p')
            return [re.sub(r'\s+', ' ', p.get_text().strip()) for p in p_tags if p.get_text().strip()]
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def filter_paragraphs(self, paragraphs, min_word_count=10):
        return list({p for p in paragraphs if len(p.split()) > min_word_count})

    def summarize_paragraphs(self, paragraphs):
        return [self.summarize_text(paragraph) for paragraph in paragraphs]

    def summarize_text(self, text, sentence_count=2):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer(parser.document, sentence_count)
        return " ".join(str(sentence) for sentence in summary)

    def vectorize_text(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                cls_embedding = self.model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
        return np.vstack(embeddings).tolist()

    def create_collection(self, vector_size=1024, distance="Cosine"):
        url = f"{self.qdrant_url}/collections/{self.collection_name}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.qdrant_api_key
        }
        payload = {"name": self.collection_name, "vectors": {"size": vector_size, "distance": distance}}
        response = requests.put(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"Collection '{self.collection_name}' created or already exists.")
        else:
            print(f"Failed to create collection: {response.text}")

    def upload_to_qdrant(self, original_texts, summaries, title, author):
        self.create_collection()
        vectors = self.vectorize_text(original_texts)

        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {
                    "title": title,
                    "author": author,
                    "original_text": original_texts[i],
                    "summary": summaries[i]
                }
            }
            for i, vector in enumerate(vectors)
        ]

        url = f"{self.qdrant_url}/collections/{self.collection_name}/points"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.qdrant_api_key
        }
        payload = {"points": points}

        response = requests.put(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            print(f"Uploaded {len(points)} points to Qdrant collection '{self.collection_name}'.")
        else:
            print(f"Failed to upload points: {response.status_code} - {response.text}")

    def fetch_point_details(self, point_id):
        url = f"{self.qdrant_url}/collections/{self.collection_name}/points/{point_id}"
        headers = {"Content-Type": "application/json", "api-key": self.qdrant_api_key}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("result", {})
        else:
            print(f"Failed to fetch details for point {point_id}: {response.status_code} - {response.text}")
            return None

    def chat_completion(self, model, user_message, temperature=0.7):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed: {response.status_code} - {response.text}")
            return None

    def run(self, url):
        paragraphs = self.scrape(url)
        if not paragraphs:
            print("Failed to retrieve content from URL.")
            return

        title, author, *content_paragraphs = paragraphs
        informative_paragraphs = self.filter_paragraphs(content_paragraphs)
        summaries = self.summarize_paragraphs(informative_paragraphs)

        if informative_paragraphs and summaries:
            self.upload_to_qdrant(informative_paragraphs, summaries, title, author)

    def search(self, query, top_k=5):
        vector_query = self.vectorize_text([query])[0]
        payload = {"vector": vector_query, "top": top_k}
        url = f"{self.qdrant_url}/collections/{self.collection_name}/points/search"
        headers = {"Content-Type": "application/json", "api-key": self.qdrant_api_key}

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Search request failed: {response.status_code} - {response.text}")
            return None

        results = response.json().get("result", [])
        return [self._format_match(result) for result in results if result]

    def _format_match(self, result):
        point_id = result["id"]
        score = result["score"]
        point_data = self.fetch_point_details(point_id)
        if not point_data:
            return None

        return {
            "title": point_data["payload"].get("title"),
            "author": point_data["payload"].get("author"),
            "original_text": point_data["payload"].get("original_text"),
            "summary": point_data["payload"].get("summary"),
            "score": score
        }

    def rephrase_summaries(self, query, results):
        if not results:
            print("No results found to rephrase.")
            return None, None

        top_result = results[0]
        context = [top_result] + [result for result in results[1:] if result['title'] == top_result['title']][:2] 

        summaries_to_rephrase = [entry['summary'] for entry in context]

        user_message = (
            f"Title: {top_result['title']}\nQuery: {query}\n"
            "Please rephrase and improve the grammar of the following summaries and expand prompt options to improve sentence flow and grammatical accuracy. Provide only the rephrased summaries as output, without any additional context or introductory phrases:\n"
            + "\n".join(summaries_to_rephrase)
        )

        rephrased_response = self.chat_completion(model="gpt-4o-mini", user_message=user_message, temperature=0.7)
        
        if rephrased_response:
            rephrased_text = rephrased_response['choices'][0]['message']['content']
            return rephrased_text, context
        else:
            print("Failed to generate rephrased summaries.")
            return None, context
        
    def save_output(self, query, context, rephrased_text, filename="output.json"):
        output_data = {
            "query": query,
            "context": context,
            "rephrased_text": rephrased_text
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"Output saved to {filename}")


if __name__ == "__main__":
    summarizer = GameReviewSummarizer(
        nltk_download=False,
        qdrant_url="https://03d1ca99-cb81-46e8-8b04-0e04895dc337.us-east4-0.gcp.cloud.qdrant.io",
        qdrant_api_key="QDRANT_API_KEY",
        openai_api_key="OPENAI_API_KEY"
    )

    # summarizer.run("https://www.gamespot.com/reviews/diablo-4-vessel-of-hatred-review-piercing-the-veil/1900-6418298/")
    # summarizer.run("https://www.gamespot.com/reviews/batman-arkham-shadow-review-i-am-batman/1900-6418308/")
    # summarizer.run("https://www.gamespot.com/reviews/redacted-review-prison-break/1900-6418307/")
    # summarizer.run("https://www.gamespot.com/reviews/call-of-duty-black-ops-6-campaign-review/1900-6418306/")
    # summarizer.run("https://www.gamespot.com/reviews/dragon-age-the-veilguard-review/1900-6418294/")
    # summarizer.run("https://www.gamespot.com/reviews/a-quiet-place-the-road-ahead-review-quite-a-pace/1900-6418305/")
    # summarizer.run("https://www.gamespot.com/reviews/fear-the-spotlight-review-blumhouses-first-video-game-is-best-enjoyed-as-an-intro-to-horror/1900-6418304/")
    # summarizer.run("https://www.gamespot.com/reviews/sonic-x-shadow-generations-review-reruns/1900-6418303/")
    # summarizer.run("https://www.gamespot.com/reviews/retrorealms-review-a-new-horror-multiverse-is-born/1900-6418302/")
    # summarizer.run("https://www.gamespot.com/reviews/mortal-kombat-1-khaos-reigns-review-organized-chaos/1900-6418301/")
    # summarizer.run("https://www.gamespot.com/reviews/super-mario-party-jamboree-review-this-party-is-too-crowded/1900-6418300/")
    # summarizer.run("https://www.gamespot.com/reviews/backyard-baseball-97-review-hit-parade/1900-6418299/")
    # summarizer.run("https://www.gamespot.com/reviews/dragon-ball-sparking-zero-review/1900-6418295/")
    # summarizer.run("https://www.gamespot.com/reviews/diablo-4-vessel-of-hatred-review-piercing-the-veil/1900-6418298/")
    # summarizer.run("https://www.gamespot.com/reviews/silent-hill-2-remake-review-born-from-a-wish/1900-6418296/")
    # summarizer.run("https://www.gamespot.com/reviews/metaphor-refantazio-review-everybody-wants-to-rule-the-world/1900-6418297/")
    # summarizer.run("https://www.gamespot.com/reviews/ea-sports-fc-25-review-lacking-title-winning-pedigree/1900-6418293/")
    # summarizer.run("https://www.gamespot.com/reviews/funko-fusion-review-pop-til-you-drop/1900-6418292/")
    # summarizer.run("https://www.gamespot.com/reviews/god-of-war-ragnarok-review-blood-sweat-and-tyrs/1900-6417993/")
    # summarizer.run("https://www.gamespot.com/reviews/the-legend-of-zelda-echoes-of-wisdom-review-a-link-between-eras/1900-6418291/")
    # summarizer.run("https://www.gamespot.com/reviews/ufo-50-review-space-shuttle-discovery/1900-6418288/")
    # summarizer.run("https://www.gamespot.com/reviews/dead-rising-deluxe-remaster-review-chopping-spree/1900-6418287/")
    # summarizer.run("https://www.gamespot.com/reviews/frostpunk-2-review-drawing-a-line-in-the-snow/1900-6418286/")
    # summarizer.run("https://www.gamespot.com/reviews/the-plucky-squire-review-every-trick-in-the-book/1900-6418285/")
    # summarizer.run("https://www.gamespot.com/reviews/wild-bastards-review-buck-around-and-find-out/1900-6418284/")
    # summarizer.run("https://www.gamespot.com/reviews/squirrel-with-a-gun-review-insert-acorn-y-joke/1900-6418283/")
    # summarizer.run("https://www.gamespot.com/reviews/marvel-vs-capcom-fighting-collection-review-new-age-of-heroes/1900-6418278/")
    # summarizer.run("https://www.gamespot.com/reviews/nba-2k25-review-luxury-taxed/1900-6418281/")
    # summarizer.run("https://www.gamespot.com/reviews/the-casting-of-frank-stone-review-habitual-ritual/1900-6418280/")
    # summarizer.run("https://www.gamespot.com/reviews/warhammer-40000-space-marine-2-review/1900-6418276/")
    # summarizer.run("https://www.gamespot.com/reviews/hollowbody-review-shattered-memories/1900-6418279/")
    # summarizer.run("https://www.gamespot.com/reviews/astro-bot-review-fly-me-to-the-moon/1900-6418277/")
    # summarizer.run("https://www.gamespot.com/reviews/world-of-warcraft-the-war-within-review-stay-awhile-and-listen/1900-6418275/")
    # summarizer.run("https://www.gamespot.com/reviews/star-wars-outlaws-review-missing-the-mark/1900-6418273/")
    # summarizer.run("https://www.gamespot.com/reviews/visions-of-mana-review-limited-tunnel-vision/1900-6418272/")
    # summarizer.run("https://www.gamespot.com/reviews/madden-nfl-25-review-gridiron-grates/1900-6418271/")
    # summarizer.run("https://www.gamespot.com/reviews/tactical-breach-wizards-review-breach-and-cast/1900-6418270/")
    # summarizer.run("https://www.gamespot.com/reviews/black-myth-wukong-review-monkey-business/1900-6418269/")
    # summarizer.run("https://www.gamespot.com/reviews/dustborn-review-words-hurt/1900-6418267/")
    # summarizer.run("https://www.gamespot.com/reviews/farewell-north-review-sit-stay/1900-6418266/")
    # summarizer.run("https://www.gamespot.com/reviews/steamworld-heist-2-review-like-clockwork/1900-6418264/")
    # summarizer.run("https://www.gamespot.com/reviews/creatures-of-ava-review-you-can-pet-the-planet/1900-6418263/")
    # summarizer.run("https://www.gamespot.com/reviews/thank-goodness-youre-here-review-propa-briish/1900-6418262/")
    # summarizer.run("https://www.gamespot.com/reviews/ea-sports-college-football-25-review-university-of-madden/1900-6418261/")
    # summarizer.run("https://www.gamespot.com/reviews/sylvio-black-waters-review-the-best-horror-series-youve-never-heard-of-does-it-again/1900-6418253/")
    # summarizer.run("https://www.gamespot.com/reviews/kunitsu-gami-path-of-the-goddess-review-danse-macabre/1900-6418258/")
    # summarizer.run("https://www.gamespot.com/reviews/the-first-descendant-review-grind-me-down/1900-6418259/")
    # summarizer.run("https://www.gamespot.com/reviews/bo-path-of-the-teal-lotus-review-a-hollow-night/1900-6418257/")
    # summarizer.run("https://www.gamespot.com/reviews/demon-slayer-sweep-the-board-review-sleep-once-bored/1900-6418255/")
    # summarizer.run("https://www.gamespot.com/reviews/final-fantasy-xiv-dawntrail-review-a-new-world/1900-6418254/")
    # summarizer.run("https://www.gamespot.com/reviews/final-fantasy-xiv-dawntrail-review-a-new-world/1900-6418254/")
    # summarizer.run("https://www.gamespot.com/reviews/gestalt-steam-and-cinder-review-steamed-maams/1900-6418252/")
    # summarizer.run("https://www.gamespot.com/reviews/nintendo-world-championships-nes-edition-review-go-go-mario/1900-6418249/")
    # summarizer.run("https://www.gamespot.com/reviews/teenage-mutant-ninja-turtles-splintered-fate-review-turtle-loop/1900-6418251/")
    # summarizer.run("https://www.gamespot.com/reviews/flintlock-the-siege-of-dawn-review-gunpowder-and-deicide/1900-6418250/")
    # summarizer.run("https://www.gamespot.com/reviews/flock-review-creature-comforts/1900-6418248/")
    # summarizer.run("https://www.gamespot.com/reviews/zenless-zone-zero-review-hackers-delight/1900-6418247/")
    # summarizer.run("https://www.gamespot.com/reviews/luigis-mansion-2-hd-review-weegee-board/1900-6418246/")
    # summarizer.run("https://www.gamespot.com/reviews/the-rogue-prince-of-persia-early-access-review-time-master/1900-6418245/")
    # summarizer.run("https://www.gamespot.com/reviews/still-wakes-the-deep-review-the-abyss-stares-back/1900-6418244/")
    # summarizer.run("https://www.gamespot.com/reviews/sand-land-review-tanks-a-lot/1900-6418216/")
    # summarizer.run("https://www.gamespot.com/reviews/stellar-blade-review-nier-as-it-can-get/1900-6418215/")
    # summarizer.run("https://www.gamespot.com/reviews/tales-of-kenzera-zau-review-bladedancing/1900-6418212/")
    # summarizer.run("https://www.gamespot.com/reviews/harold-halibut-review-lost-in-its-own-deep-sea/1900-6418211/")
    # summarizer.run("https://www.gamespot.com/reviews/children-of-the-sun-review-one-shot/1900-6418208/")
    # summarizer.run("https://www.gamespot.com/reviews/star-wars-battlefront-classic-collection-review-fire-away/1900-6418207/")
    # summarizer.run("https://www.gamespot.com/reviews/open-roads-review-quick-trip/1900-6418206/")
    # summarizer.run("https://www.gamespot.com/reviews/pepper-grinder-review-short-and-spicy/1900-6418205/")
    # summarizer.run("https://www.gamespot.com/reviews/mlb-the-show-24-review-base-hit/1900-6418203/")
    # summarizer.run("https://www.gamespot.com/reviews/princess-peach-showtime-review-drama-teacher/1900-6418198/")
    # summarizer.run("https://www.gamespot.com/reviews/rise-of-the-ronin-review-long-term-investment/1900-6418202/")
    # summarizer.run("https://www.gamespot.com/reviews/dragons-dogma-2-review-pawn-stars/1900-6418199/")
    # summarizer.run("https://www.gamespot.com/reviews/alone-in-the-dark-review-dimly-lit/1900-6418197/")
    # summarizer.run("https://www.gamespot.com/reviews/alone-in-the-dark-review-dimly-lit/1900-6418197/")
    # summarizer.run("https://www.gamespot.com/reviews/contra-operation-galuga-review-corps-run/1900-6418195/")
    # summarizer.run("https://www.gamespot.com/reviews/disney-dreamlight-valley-review-great-game-grueling-grind/1900-6418193/")
    # summarizer.run("https://www.gamespot.com/reviews/balatro-review-one-more-blind/1900-6418192/")
    # summarizer.run("https://www.gamespot.com/reviews/wwe-2k24-review-long-term-booking/1900-6418191/")
    # summarizer.run("https://www.gamespot.com/reviews/the-outlast-trials-review-immersion-therapy/1900-6418190/")
    # summarizer.run("https://www.gamespot.com/reviews/pennys-big-breakaway-review-if-it-aint-broke/1900-6418189/")
    # summarizer.run("https://www.gamespot.com/reviews/final-fantasy-7-rebirth-review-destinys-child/1900-6418187/")

    query = "I want to know about Super Mario Party"
    results = summarizer.search(query)

    rephrased_text, context = summarizer.rephrase_summaries(query, results)
    summarizer.save_output(query, context, rephrased_text, "data/output.json")
