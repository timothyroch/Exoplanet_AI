import random

def random_greeting():
    greetings = ["Hello, world!", "Hi there!", "Greetings!", "Salutations!", "Hey!"]
    return random.choice(greetings)

if __name__ == "__main__":
    print(random_greeting())