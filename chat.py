import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lazy-load variables
model = None
intents = None
all_words = None
tags = None
initialized = False

def load_chatbot():
    """Load the chatbot model and data once (on first message)."""
    global model, intents, all_words, tags, initialized
    if initialized:
        return

    with open('intents.json', 'r', encoding='utf-8') as json_data:
        intents_data = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE, map_location=device)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words_local = data['all_words']
    tags_local = data['tags']
    model_state = data["model_state"]

    net = NeuralNet(input_size, hidden_size, output_size).to(device)
    net.load_state_dict(model_state)
    net.eval()

    # Assign globals
    model = net
    intents = intents_data
    all_words = all_words_local
    tags = tags_local
    initialized = True
    print("✅ Chatbot model loaded successfully!")

def get_response(msg):
    """Generate chatbot response."""
    load_chatbot()

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "No results found. Try another appropriate keyword."

if __name__ == "__main__":
    print("Chatbot ready! Type 'quit' to stop.")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        print("Bot:", get_response(msg))
