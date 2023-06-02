import json
import time
from urllib.request import urlopen
import pandas as pd
from nltk import sent_tokenize


class SteamController:
    day_range = 9223372036854775807
    language = "english"

    def get_game_reviews(self, id):
        reviews_df = pd.DataFrame()
        cursor = "*"
        max_batches = 5
        current_batch = 0

        while True and current_batch < max_batches:
            url = f"https://store.steampowered.com/appreviews/{id}?json=1&num_per_page=100&day_range={self.day_range}&language={self.language}&cursor={cursor}"
            response = urlopen(url)
            data = json.loads(response.read())

            if data.get("cursor") is not None:
                new_cursor = data["cursor"]
                if new_cursor == cursor:
                    break
                cursor = new_cursor
            else:
                break

            new_reviews = pd.DataFrame(pd.json_normalize(data["reviews"]))
            reviews_df = pd.concat([reviews_df, new_reviews], ignore_index=True)

            current_batch += 1
            time.sleep(1)

        min_votes_up = 1
        min_playtime = min(reviews_df["author.playtime_at_review"].median(), 120)
        min_review_symbols_length = 20
        min_review_words_length = 3

        reviews_df["review_symbols_length"] = reviews_df.review.str.len()
        reviews_df["review_word_length"] = len(reviews_df.review.str.split())

        reviews_reduced_df = reviews_df.loc[
            (reviews_df["votes_up"] >= min_votes_up)
            & (reviews_df["received_for_free"] != "True")
            & (reviews_df["author.playtime_at_review"] >= min_playtime)
            & (reviews_df["review_symbols_length"] >= min_review_symbols_length)
            & (reviews_df["review_word_length"] >= min_review_words_length)
        ]

        reviews_reduced_df = reviews_reduced_df.reset_index()

        reviews_reduced_df["review"] = reviews_reduced_df["review"].apply(
            lambda x: sent_tokenize(x)
        )

        reviews_reduced_df = reviews_reduced_df.explode("review")

        reviews_reduced_df[
            "review_symbols_length"
        ] = reviews_reduced_df.review.str.len()
        reviews_reduced_df["review_word_length"] = len(
            reviews_reduced_df.review.str.split()
        )

        reviews_reduced_df = reviews_reduced_df.loc[
            (reviews_reduced_df["review_symbols_length"] >= min_review_symbols_length)
            & (reviews_reduced_df["review_word_length"] >= min_review_words_length)
        ]

        reviews_reduced_df = reviews_reduced_df[["review"]]

        return reviews_reduced_df