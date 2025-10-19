import string

# global string of all caps values
ALPHA = string.ascii_uppercase

class Rotor:

    # dictionaries to store historic rotor alpha sequences and starting notch positions
    ENIGMA_ROTORS = {
        1:"EKMFLGDQVZNTOWYHXUSPAIBRCJ",
        2:"AJDKSIRUXBLHWTMCQGZNPYFVOE",
        3:"BDFHJLCPRTXVZNYEIWGAKMUSQO",
        4:"ESOVPZJAYQUIRHKLNFTGKDCMWB",
        5:"VZBRGITYUPSDNHLXAWMJQOFECK"
    }

    N_POSITIONS = {
        1:"Q",
        2:"E",
        3:"V",
        4:"J",
        5:"Z"
    }

    def __init__(self, rotor_number, position=0):
        # self check
        if rotor_number not in self.ENIGMA_ROTORS:
            raise ValueError("Invalid Rotor number, select numbers 1 through 5 ONLY")

        self.rotor_num = rotor_number
        self.wiring = self.ENIGMA_ROTORS[rotor_number]
        self.notch = self.N_POSITIONS[rotor_number]
        if isinstance(position, str):
            position = string.ascii_uppercase.index(position.upper())
        self.position = position

    def step(self):
        self.position = (self.position + 1) % 26

    def encrypt_f(self, letter):
        # encrypt forward over rotors right to left
        index = (ALPHA.index(letter) + self.position) % 26
        swap = self.wiring[index]
        indexOut = (ALPHA.index(swap) - self.position) % 26

        return ALPHA[indexOut]

    def encrypt_b(self, letter):
        # encrypt over rotors in reverse order
        index = (ALPHA.index(letter) + self.position) % 26
        rotorWiring = self.wiring.index(ALPHA[index])
        indexOut = (rotorWiring - self.position) % 26

        return ALPHA[indexOut]

    def at_notch(self):
        return ALPHA[self.position] in self.notch