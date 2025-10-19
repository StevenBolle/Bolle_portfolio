import string

# global constant string of all caps values
ALPHA = string.ascii_uppercase

class Reflector:
    # dictionary of available historical reflector choices
    # https://en.wikipedia.org/wiki/Enigma_rotor_details
    REFLECTOR_CHOICE = {
        'A':"EJMZALYXVBWFCRQUONTSPIKHGD",
        'B':"YRUHQSLDPXNGOKMIEBFZCWVJAT",
        'C':"FVPJIAOYEDRZXWGCTKUQSBNMHL"
    }

    def __init__(self, reflector):
        if reflector not in self.REFLECTOR_CHOICE:
            raise ValueError("Please choose a reflector from A, B or C")
        self.reflector = reflector
        self.choice = self.REFLECTOR_CHOICE[reflector]

    def reflect(self, letter):
        # reflect letter
        index = ALPHA.index(letter)
        return self.choice[index]