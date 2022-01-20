# Context-Aware OOD dataset
This repository will be used for code and data used in a paper *Context-Aware Out-of-Domain Detection in Conversational Domain* (all code and data will be uploaded after acceptance/rejection of paper). All data contain 49 dialogue points covering 12 topics.


Data format:
```
{
  "reference": {
    "dialogue_name": "movies/howchoose",
    "decision_node": "62"
  },
  "bot_response": [
    "How do you choose a movie to watch?",
    "What makes you choose a particular movie to watch?",
    "How do you decide which movie you'll watch?"
  ],
  "user_response": [
    {
      "train": [
        "my friends recommend me movies",
        "I use the word of mouth",
        "My friend recommends me movies",
        "Generally word of mouth",
        "I watch what my friends recommend me",
        "I use word of mouth",
        "word of mouth please",
        "word of mouth",
        "My friends recommend it to me",
        "I wait for the recommendation of my friends"
      ],
      "val": [
        "I ask my friends what to watch",
        "Friends recommend it",
        "it's based on recommendations"
      ],
      "test": [
        "My friends recommend it to me please",
        "I watch what my friends recommend me",
        "Just talking with my friends"
      ]
    },
    {
      "train": [
        "I found it on the imdb",
        "I check the internet and than i decide",
        "I google it",
        "imdb",
        "I check the imdb and than i decide",
        "internet",
        "I check the web and than i decide",
        "I found it on the internet",
        "I use the web",
        "I found it on the web",
        "I search imdb"
      ],
      "val": [
        "on the web",
        "I search the web",
        "I look it up online",
        "I look at imdb"
      ],
      "test": [
        "web",
        "the internet",
        "online",
        "I use the internet"
      ]
    },
    {
      "train": [
        "From internet reviews",
        "somehow I read reviews and then I decide",
        "I search internet for internet reviews",
        "I read movie reviews you see",
        "believe me find movies by reading reviews",
        "I read reviews and then I decide",
        "read reviews really and then I decide",
        "I find movies by reading reviews"
      ],
      "val": [
        "I read movie reviews",
        "based on movie reviews",
        "based on movie recommendations online"
      ],
      "test": [
        "I decide according to reviews",
        "I read reviews and then I decide you know",
        "I search in the internet for internet reviews",
        "according to film reviews"
      ]
    },
    {
      "train": [
        "I watch trailers",
        "From really trailers",
        "so trailers",
        "I watch teasers",
        "teasers",
        "From trailers",
        "previews you see",
        "previews",
        "I watch trailers yeah"
      ],
      "val": [
        "I check trailers",
        "I look up the trailers",
        "I decide based on the movie trailers",
        "I look at the teasers"
      ],
      "test": [
        "I watch trailers on youtube",
        "I check trailers please",
        "trailers",
        "previews",
        "teasers"
      ]
    },
    {
      "train": [
        "it's hard to say",
        "I do not really know",
        "it's hard to tell",
        "I can not remember",
        "do not know",
        "I do not remember",
        "it is hard to tell",
        "can't remember",
        "do not remember",
        "can not remember",
        "literally no idea",
        "I don't remember",
        "who knows",
        "don't remember",
        "I don't really know",
        "I can't decide",
        "it is difficult to tell",
        "I do not know",
        "it is hard to say",
        "no idea",
        "I can not decide",
        "it's difficult to say",
        "cannot remember",
        "don't know",
        "it is difficult to say",
        "I cannot decide"
      ],
      "val": [
        "I don't know",
        "do not really know",
        "it's difficult to tell",
        "I have no idea"
      ],
      "test": [
        "I have literally no idea really",
        "no idea really",
        "I cannot remember",
        "literally no idea really",
        "I have no idea really",
        "don't really know",
        "I can't remember",
        "I have literally no idea"
      ]
    },
    {
      "train": [
        "can you whatever is on",
        "what I find on Netflix",
        "somehow whatever's on TV",
        "whatever is on yeah",
        "whatever is on",
        "think whatever's on TV",
        "what Netflix suggests yeah"
      ],
      "val": [
        "what I find on Netflix you know",
        "just whatever",
        "I watch whatever",
        "I don't care what I watch"
      ],
      "test": [
        "whatever is on",
        "whatever's on TV",
        "what Netflix suggests",
        "what I find on Netflix",
        "I look at Netflix recommendations"
      ]
    },
    {
      "train": [
        "I always watch romantic movies",
        "I always watch action movies please",
        "I always watch comedies",
        "I always watch romantic movies",
        "I watch just my favorite genres",
        "I always watch action movies",
        "based on the genre"
      ],
      "val": [
        "I always watch crime movies",
        "I like detective stories",
        "Usually I watch romantic movies"
      ],
      "test": [
        "romantic movies",
        "basically watch just my favorite genres",
        "only my favourite genre",
        "thrillers",
        "comedies"
      ]
    }
  ],
  "out_of_domain": [
    "I like to play with animals",
    "What do you think about yellow pants",
    "Music is my life",
    "I speak fluently snake language as Harry Potter",
    "snake language",
    "i choose my clothes according to weather",
    "weather",
    "movies are something for simple people",
    "watching TV is not an exhausting acitivity",
    "I don't think there are good movies on the market",
    "I never go to the movie theatre",
    "Movie theatres are super expensive",
    "Cinemas are totally expensive",
    "do you smoke",
    "I don't eat apples",
    "this is not funny"
  ]
}
```
