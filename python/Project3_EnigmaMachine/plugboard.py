import string
from typing import Dict

# global constant string of all caps values
ALPHA = string.ascii_uppercase

class Plugboard:

    def __init__(self, connections = ""):
        # make a pair of letters connected via plugboard
        # dict to hold plugboard connections
        # type hinting https://youtu.be/fmCQ5XM4igA ensures that plugboard values entered by user are passed as strings and not index ints
        self.map: Dict[str, str] = {letter: letter for letter in ALPHA}

        if not connections.strip():
            return

        #print("Debug: connections")
        pairs = connections.upper().split() # split plugboard entry into tokenized 'pairs'
        #print("Debug: connections after pairs = connections.upper().split() : ")
        #print(pairs)

        if len(pairs) > 10:
            raise ValueError("Plugboard can only have maximum of 10 connections")

        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Plugboard connections must be made in pairs: {pair}")
            a, b = pair[0], pair[1]
            #print(f"debug: processing pair: pair a: {a} and pair b: {b}")

            if a not in ALPHA or b not in ALPHA:
                raise ValueError(f"Bad plugboard pair: {a}{b}")
            if self.map[a] != a or self.map[b] != b:
                raise ValueError(f"letters in plugboard can only be mapped once: {a}{b}")

            # reinforce a, b pairing (formerly x y)
            self.map[a] = b
            self.map[b] = a

    def encrypt(self, letter):
        return self.map.get(letter, letter)