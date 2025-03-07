# Inventory-management-bot
An inventory management chatbot that perfroms statistical functions purely based on NLP techniques. No LLM usage required.
# What it is
This is a chatbot that applies intent recognition and entity recognition using Spacy and NLTK. It applies NER and pattern matching to figure out what a user might be asking for.
The chatbot can execute a few functions, such as sales prediction, brand comparison, inventory lookup, market basket analysis and many others.

Each of these functions require their own specific parameters to be passed into them. The role of the bot is to :
1. Extract entities from the user's text.
2. Extract intent of the user using pattern mathching.
3. Apply transformation to the entities to fit the format required for the functions as a parameter.
# Functions performed by the bot
The bot generall performs these 3 types of functions:
1. Retrieval functions: Retrieve data as it is based on user queries.
2. Prediction functions: Predicts the sales of a product based on past data (exponential smoothing). Draws trends charts of the sales as well.
3. Analysis functions: The bot can do Market basket analysis to derive frequently bought together items, it can also perform comparisons of performance across different brands and different time periods.

The Bot can be trained to identify certain brands and keywords as well apart from all of this.
Currently it works with a few CSVs as it's source data, and can run completely offline. Let me know if you have any feedback, I'm open to ideas.
