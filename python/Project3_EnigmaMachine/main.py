import string
from enigmaMachine import EnigmaMachine
from rotor import Rotor
from plugboard import Plugboard
from reflector import Reflector


def main():
    rotors = [Rotor(rotor_number=1,position=0), Rotor(rotor_number=2,position=0),Rotor(rotor_number=3,position=0)]
    reflector = Reflector(reflector='B')
    plugboard = Plugboard("")
    enigma = EnigmaMachine(rotors,reflector,plugboard)

    flag = True
    original = ""
    last = ""
    original_positions = [0,0,0]

    print("\n--------- Enigma Machine ---------")
    print("Willkommen bei der Enigma-Maschine")

    print("\n\nUse menu to choose your own desired settings or\ngo directly to menu option 4 to get started!")

    while flag:
        print("\n****** MENU ******")
        print("1: Choose rotors and starting positions")
        print("2: Choose reflector")
        print("3: Set plugboard")
        print("4: Encrypt message")
        print("5: Decrypt message")
        print("6: Exit Enigma program")

        select = int(input("Enter selection, 1-6:  "))

        if select == 1:
            try:
                print("\nRotors available, numbered 1-8")
                roto_choice = input("Enter rotor number choices (no repeats), e.g. 123:  ").strip()
                roto_nums = [int(r) for r in roto_choice if r.isdigit()]

                if len(roto_nums) != 3:
                    print("Provide EXACTLY three rotor numbers!")
                    continue
                if not all(1 <= r <= 8 for r in roto_nums):
                    print("Error: Rotor selection not in range 1-5")
                    continue

                roto_pos = input("Enter rotor starting positions from A-Z, e.g. ABC :  ")
                if len(roto_pos) != 3:
                    print("Error: only choose 3 staring positions from A-Z")
                    continue
                rotor1 = Rotor(1, roto_pos[0])
                rotor2 = Rotor(2, roto_pos[1])
                rotor3 = Rotor(3, roto_pos[2])

                enigma = EnigmaMachine(rotors, reflector, plugboard)
                print("\nRotors and positions set successfully!")
            except ValueError as e:
                print(f"Error: {e}")

        elif select == 2:
            ref_input = input("\nPlease choose reflector from A, B or C:  ").strip().upper()
            if ref_input not in['A', 'B', 'C']:
                print("Please select a reflector from A, B or C")
                continue
            reflector = Reflector(reflector=ref_input)

            enigma = EnigmaMachine(rotors, reflector, plugboard)
            print("Reflector installed successfully!")

        elif select == 3:
            try:
                plug_choice = input("\nPlease enter plugboard connections, no repeated values,"
                                   "\n e.g. AB CD EF less than or equal to 10 pairs:")
                plugboard = Plugboard(plug_choice)

                enigma = EnigmaMachine(rotors, reflector, plugboard)
                print("Plugboard connections successful")
            except ValueError as e:
                print(f"Error: {e}")

        elif select == 4:
            print("\n!!!!! Enigmo-matic !!!!!")
            original = input("\nEnter message to encrypt: ")
            original = original.upper()
            original_positions = [rotor.position for rotor in rotors]

            print("\nEncryption in progress. . .")
            last = enigma.encrypt_message(original)
            print("Encrypted message: " + last)

        elif select == 5:
            print("\n!!!!! Decrypt-o-rator !!!!!")
            if not last:
                encrypt_new_prompt = input("\nNo encrypted message stored. Enter message to Enigma: ")
                encrypt_message = enigma.encrypt_message(encrypt_new_prompt)
                print("Encrypted message: " + encrypt_message)
            else:
                # reset rotors to process message during decryption
                enigma.reset_rotors()

                print("\nDecrypting message. . .")
                decrypt_message = enigma.encrypt_message(last).upper()
                print("Message decrypted: " + decrypt_message)

                if decrypt_message == original:
                    print("Decrypted message matches original: success!")
                else:
                    print("Decrypted message does not match original message."
                          "\n Original message: " + original)
            enigma.reset_rotors()

        elif select == 6:
            print("Exiting Enigmatomater...")
            flag = False
        else:
            print("\nInvalid choice, please select menu options 1-6")

if __name__ == "__main__":
    main()