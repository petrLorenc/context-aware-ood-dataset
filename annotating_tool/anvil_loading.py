import pandas as pd
import time
import random
import anvil.tables as tables
from anvil.tables import app_tables

import anvil.server

anvil.server.connect("4OEXDBNLWWW2QO4B5PVRLBXW-6N6WDAJ2HJAJ6YNR")  # Make sure you replace this with your own Uplink key


def import_csv_data(file, question, topic):
    only_cnt = 50
    with open(file, "r") as f:
        df = pd.read_csv(f)
        cnt = len(app_tables.inputs_small.search())
        try:
            for d in random.sample(list(df.to_dict(orient="records")), k=only_cnt):
                # d is now a dict of {columnname -> value} for this row
                # We use Python's **kwargs syntax to pass the whole dict as
                # keyword arguments
                d["question"] = question
                d["topic"] = topic
                d["id"] = cnt
                cnt += 1
                try:
                    app_tables.inputs_small.add_row(**d)
                except:
                    pass

                time.sleep(random.random())
        except:
            print(file)



## 11 is duplicate
## 10 redundant

if __name__ == '__main__':
    import_csv_data("./data/raw/01.csv.new", question="What game do you have the best memories of?", topic="Games")
    import_csv_data("./data/raw/02.csv.new", question="When buying a game, do you prefer a physical, or a digital version?", topic="Games")
    import_csv_data("./data/raw/06.csv.new", question="Do you prefer playing against the computer, or against other humans?", topic="Games")
    import_csv_data("./data/raw/36.csv.new", question="What is your favorite game?", topic="Games")

    import_csv_data("./data/raw/03.csv.new", question="What did you do yesterday?", topic="Freetime")
    import_csv_data("./data/raw/04.csv.new", question="What are your plans for tomorrow?", topic="Freetime")
    import_csv_data("./data/raw/08.csv.new", question="Do you think mass tourism helps, or hurts cities in general?", topic="Freetime")
    import_csv_data("./data/raw/09.csv.new", question="Who is your favorite street artist?", topic="Freetime")
    import_csv_data("./data/raw/16.csv.new", question="What do you do to relax?", topic="Freetime")
    import_csv_data("./data/raw/17.csv.new", question="What are your plans for the coming weekend?", topic="Freetime")
    import_csv_data("./data/raw/24.csv.new", question="Do you miss any cultural events, now that many were cancelled?", topic="Freetime")
    import_csv_data("./data/raw/25.csv.new", question="What are your hobbies?", topic="Freetime")
    import_csv_data("./data/raw/29.csv.new", question="What's your favorite ice cream flavor?", topic="Freetime")
    import_csv_data("./data/raw/43.csv.new", question="What is your favorite day of the week?", topic="Freetime")

    import_csv_data("./data/raw/05.csv.new", question="What is your favorite one?", topic="Sports")
    import_csv_data("./data/raw/37.csv.new", question="If you could be the best in the world in any sport, which sport would you choose?",
                    topic="Sports")

    import_csv_data("./data/raw/07.csv.new", question="What food did you hate when you were little?", topic="Foods")
    import_csv_data("./data/raw/12.csv.new", question="What's your favorite drink?", topic="Foods")
    import_csv_data("./data/raw/14.csv.new", question="Do you usually eat out, or at home?", topic="Foods")
    import_csv_data("./data/raw/31.csv.new", question="What did you have for breakfast?", topic="Foods")
    import_csv_data("./data/raw/44.csv.new", question="What is your favorite?", topic="Foods")

    import_csv_data("./data/raw/15.csv.new", question="Who's your favorite writer?", topic="Books")
    import_csv_data("./data/raw/38.csv.new", question="Which one is your favorite?", topic="Books")
    import_csv_data("./data/raw/39.csv.new", question="Where do you prefer to read books?", topic="Books")

    import_csv_data("./data/raw/35.csv.new", question="What's your favorite field?", topic="Science")
    import_csv_data("./data/raw/23.csv.new", question="What would you like to invent?", topic="Science")
    import_csv_data("./data/raw/18.csv.new", question="What do you think is the most important invention ever created?", topic="Science")

    import_csv_data("./data/raw/19.csv.new", question="What's your favorite color?", topic="Fashion")
    import_csv_data("./data/raw/32.csv.new", question="Which two things would you never wear together?", topic="Fashion")
    import_csv_data("./data/raw/33.csv.new", question="What nice piece of clothing have you bought recently?", topic="Fashion")

    import_csv_data("./data/raw/20.csv.new", question="What's the last song that you've liked a lot?", topic="Music")
    import_csv_data("./data/raw/30.csv.new", question="What is your favorite one?", topic="Music")

    import_csv_data("./data/raw/21.csv.new", question="Do you consider yourself to be punctual?", topic="General")
    import_csv_data("./data/raw/22.csv.new", question="Do you come from a big family?", topic="General")

    import_csv_data("./data/raw/26.csv.new", question="Which one are you interested in seeing?", topic="Movies")
    import_csv_data("./data/raw/28.csv.new", question="Which character from the movie you'd wanna be friends with in real life?", topic="Movies")
    import_csv_data("./data/raw/34.csv.new", question="What is your favorite one?", topic="Movies")

    import_csv_data("./data/raw/41.csv.new", question="What is your favorite one? ", topic="Animals")
    import_csv_data("./data/raw/40.csv.new", question="If you could turn into an animal, which one would you choose?", topic="Animals")
    import_csv_data("./data/raw/42.csv.new", question="What animal do you think is the most dangerous?", topic="Animals")
