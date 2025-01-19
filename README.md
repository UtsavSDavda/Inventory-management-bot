# Inventory-management-bot
An inventory management chatbot that perfroms statistical functions purely based on NLP techniques. No LLM usage required.
# What it is
This is a chatbot that applies intent recognition and entity recognition using Spacy and NLTK. It applies NER and pattern matching to figure out what a user might be asking for.
The chatbot can execute a few functions, such as sales prediction, brand comparison, inventory lookup, market basket analysis and many others.

Each of these functions require their own specific parameters to be passed into them. The role of the bot is to :
1. Extract entities from the user's text.
2. Extract intent of the user using pattern mathching.
3. Apply transformation to the entities to fit the format required for the functions as a parameter.
