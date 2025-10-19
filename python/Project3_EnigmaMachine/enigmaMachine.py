class EnigmaMachine:

    def __init__(self, rotors, reflector, plugboard):
        # print('debug: enigmamachine class, constructor')
        if len(rotors) != 3:
            raise ValueError('3 rotors required')
        self.rotors = rotors
        self.reflector = reflector
        self.plugboard = plugboard
        self.initial_positions = [rotor.position for rotor in rotors]

    # name changed to describe action
    def step_rotors(self):
        # rotors move and use double stepping
        # https://en.wikipedia.org/wiki/Enigma_machine
        #print("\ndebug: step rotor engaged")
        # old rotor step logic scrapped
        first_rotor_at_notch = self.rotors[0].at_notch()
        second_rotor_at_notch = self.rotors[1].at_notch()

        if second_rotor_at_notch:
            self.rotors[1].step()
            self.rotors[2].step()
        elif first_rotor_at_notch:
            self.rotors[1].step()
        self.rotors[0].step()
        #print("debug: step rotor success")

    # name changed to describe function clearly
    # https://en.wikipedia.org/wiki/Enigma_machine each letter from message processed individually
    # based on historic movement of letter signal through enigma machine
    def process_letter(self, letter):
        #print("Debug: process letter called")
        if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return letter
        letter = self.plugboard.encrypt(letter)
        self.step_rotors()
        # letter processed from right to left
        for rotor in self.rotors:
            letter = rotor.encrypt_f(letter)
        # letter reflected
        letter = self.reflector.reflect(letter)
        # process letter left to right
        for rotor in reversed(self.rotors):
            letter = rotor.encrypt_b(letter)
        letter = self.plugboard.encrypt(letter)
        #print("debug: end of process letter function reached")
        return letter

    def encrypt_message(self, message):
        #print("debug: encrypt message called")
        # ensure message is in all caps
        message = message.upper()
        processed = ""
        for letter in message:
            if letter.isalpha():
                processed += self.process_letter(letter)
        #print("debug: end of encrypt message reached")
        return processed

    def reset_rotors(self):
        #print("Debug: reset rotors called")
        # use enumerate to reset moved positions to known original positions https://youtu.be/tys4c8Nf3xs
        for i, pos in enumerate(self.initial_positions):
            self.rotors[i].position = pos
        #print("Debug: rotors reset successfully")