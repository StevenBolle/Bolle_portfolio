from unittest import TestCase
import string
from enigmaMachine import EnigmaMachine
from rotor import Rotor
from plugboard import Plugboard
from reflector import Reflector

# global string of all caps values
ALPHA = string.ascii_uppercase

class RotorTest(TestCase):
    def test_initialize(self):
        # ARRANGE
        roto_num = 1
        position = 'A'

        # ACT
        rotor = Rotor(roto_num, position)

        # ASSERT
        self.assertEqual(rotor.rotor_num, 1)
        self.assertEqual(rotor.position, 0)
        self.assertEqual(rotor.wiring, Rotor.ENIGMA_ROTORS[1])
        self.assertEqual(rotor.notch, Rotor.N_POSITIONS[1])

    def test_step(self):
        # ARRANGE
        rotor = Rotor(1, 'A')

        # ACT
        rotor.step()

        # ASSERT
        self.assertEqual(rotor.position, 1)

        for _ in range(25):
            rotor.step()
        self.assertEqual(rotor.position,0)

    def test_encrypt_f(self):
        # ARRANGE
        rotor = Rotor(1,'A')

        # ACT

        # ASSERT
        self.assertEqual(rotor.encrypt_f('A'), 'E')
        self.assertEqual(rotor.encrypt_f('Z'), 'J')

    def test_encrypt_b(self):
        # ARRANGE
        rotor = Rotor(1, 'A')

        # ACT

        # ASSERT
        self.assertEqual(rotor.encrypt_b('E'), 'A')
        self.assertEqual(rotor.encrypt_b('J'), 'Z')

    def test_at_notch(self):
        # ARRANGE
        rotor = Rotor(1,'A')
        notch_position = Rotor.N_POSITIONS[1]

        # ACT
        while not rotor.at_notch():
            rotor.step()

        # ASSERT
        self.assertTrue(rotor.at_notch())
        rotor.step()
        self.assertFalse(rotor.at_notch())

class TestReflector(TestCase):

    def test_reflector(self):
        # ARRANGE
        reflector = Reflector('B')

        # ACT


        # ASSERT
        self.assertEqual(reflector.reflect('A'), 'Y')
        self.assertEqual(reflector.reflect('Y'), 'A')

        for letter in string.ascii_uppercase:
            mirrored = reflector.reflect(letter)
            self.assertEqual(reflector.reflect(mirrored), letter)

class TestPlugboard(TestCase):

    def test_swap(self):
        # Arrange
        plugboard = Plugboard("AG HL ES")
        # Act

        # Assert
        self.assertEqual(plugboard.encrypt('A'), 'G')
        self.assertEqual(plugboard.encrypt('G'), 'A')
        self.assertEqual(plugboard.encrypt('H'), 'L')
        self.assertEqual(plugboard.encrypt('L'), 'H')
        self.assertEqual(plugboard.encrypt('E'), 'S')
        self.assertEqual(plugboard.encrypt('S'), 'E')

    def test_unswapped(self):
        # Arrange
        plugboard = Plugboard()

        # Act

        # Assert
        self.assertEqual(plugboard.encrypt('A'), 'A')
        self.assertEqual(plugboard.encrypt('Z'), 'Z')
        for letter in string.ascii_uppercase:
            self.assertEqual(plugboard.encrypt(letter), letter)

class TestMachine(TestCase):

    def test_initialize_settings(self):
        # arrange
        rotor1 = Rotor(1, 'A')
        rotor2 = Rotor(2, 'B')
        rotor3= Rotor(3, 'C')
        reflector = Reflector('B')
        plugboard = Plugboard("HP LS OK")

        # act
        enigma = EnigmaMachine([rotor1,rotor2,rotor3], reflector, plugboard)

        # assert
        self.assertEqual(len(enigma.rotors), 3) # are 3 rotors init'd in machine?
        self.assertEqual(enigma.rotors[0].rotor_num, 1)
        self.assertEqual(enigma.rotors[1].rotor_num, 2)
        self.assertEqual(enigma.rotors[2].rotor_num, 3)
        self.assertEqual(enigma.initial_positions, [0,1,2])

    def test_rotor_stepping(self):
        # arrange
        rotor1 = Rotor(1, 'A')
        rotor2 = Rotor(2, 'B')
        rotor3 = Rotor(3, 'C')
        reflector = Reflector('B')
        plugboard = Plugboard("")
        enigma = EnigmaMachine([rotor1, rotor2, rotor3], reflector, plugboard)
        initial_pos = [rotor.position for rotor in enigma.rotors]
        # act
        enigma.step_rotors()
        # assert
        self.assertEqual(enigma.rotors[0].position,(initial_pos[0] + 1) % 26) # rotor 1 always steps

    def test_process_letter(self):
        # ARRANGE
        rotor1 = Rotor(1, 'A')
        rotor2 = Rotor(2, 'B')
        rotor3 = Rotor(3, 'C')
        reflector = Reflector('B')
        plugboard = Plugboard("AB")
        enigma = EnigmaMachine([rotor1, rotor2, rotor3], reflector, plugboard)

        # ACT
        end_result = enigma.process_letter('A')

        # ASSERT
        expected_letter = 'E'
        self.assertEqual(end_result, expected_letter)

    def test_encrypt(self):
        # ARRANGE
        rotor1 = Rotor(1, 'A')
        rotor2 = Rotor(2, 'B')
        rotor3 = Rotor(3, 'C')
        reflector = Reflector('B')
        plugboard = Plugboard("")
        enigma = EnigmaMachine([rotor1, rotor2, rotor3], reflector, plugboard)

        # ACT
        encrypt = enigma.encrypt_message("HELLO")

        # ASSERT
        encrypted_result = "UZCFS"
        self.assertEqual(encrypt, encrypted_result)

    def test_decrypt(self):
        # ARRANGE
        rotor1 = Rotor(1, 'A')
        rotor2 = Rotor(2, 'B')
        rotor3 = Rotor(3, 'C')
        reflector = Reflector('B')
        plugboard = Plugboard("")
        enigma = EnigmaMachine([rotor1, rotor2, rotor3], reflector, plugboard)
        orig_msg = "WAFLLESARETASTY"

        # ACT
        encrypted_msg = enigma.encrypt_message(orig_msg)
        enigma.reset_rotors()
        decrypted_msg = enigma.encrypt_message(encrypted_msg)

        # ASSERT
        self.assertEqual(decrypted_msg,orig_msg)

    def test_edge(self):
        # Arrange
        rotor1 = Rotor(1, 'A')
        rotor2 = Rotor(2, 'B')
        rotor3 = Rotor(3, 'C')
        reflector = Reflector('B')
        plugboard = Plugboard("")
        enigma = EnigmaMachine([rotor1, rotor2, rotor3], reflector, plugboard)

        # ACT
        encrypt_msg = enigma.encrypt_message("My NAME IS ST3V#N")

        # ASSERT
        self.assertFalse(' ' in encrypt_msg)
        self.assertFalse('y' in encrypt_msg)
        self.assertFalse('3' in encrypt_msg)
        self.assertFalse('#' in encrypt_msg)