package guessnum_2614;

// Author: Steven Bolle
// Date: 7/15/2023

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.security.SecureRandom;
import javax.swing.*;

public class GuessNum_2614 extends JFrame {
    
    private final JLabel k1;
    private final JLabel k2;
    private final JLabel k3;

    private final JLabel status;

    private final JTextField textField;

    private JButton playAgain;

    private Font font;
    
    private final SecureRandom random = new SecureRandom();
    private int randNum;
    
    public GuessNum_2614() {
        super("Guess the Number");
        setLayout(new FlowLayout());
        getContentPane().setBackground(Color.lightGray);

        numberGenerator();

        font = new Font("Serif", Font.BOLD, 18);

        k1 = new JLabel("I have a number between 1 and 1000. ");
        k1.setFont(font);
        add(k1);

        k2 = new JLabel("Can you guess my number?");
        k2.setFont(font);
        add(k2);

        k3 = new JLabel("Please enter your guess.");
        k3.setFont(font);
        add(k3);

        textField = new JTextField(3);
        textField.setToolTipText("Type your guess and press 'Enter'!");
        add(textField);

        font = new Font("Times", Font.BOLD, 15);
        status = new JLabel("Awaiting guess. . .");
        status.setFont(font);
        add(status);

        playAgain = new JButton("Play Again?");
        playAgain.setBackground(Color.CYAN);
        playAgain.setToolTipText("Click here to play again!");
        playAgain.addActionListener(new ActionListener() {
            
            @Override
            public void actionPerformed(ActionEvent e) {
                numberGenerator();
                textField.setEnabled(true);
                textField.setText("");
                status.setText("Status: waiting guess ..");
                getContentPane().setBackground(Color.WHITE);
                remove(playAgain);

                repaint();
                validate();
            }
        });
        
        Handler handler = new Handler();
	textField.addActionListener(handler);
    }
    
    private class Handler implements ActionListener {
        
        @Override
        public void actionPerformed(ActionEvent event) {
            String text = "";
            int guess = Integer.parseInt(event.getActionCommand());

            if (guess == randNum) {
                    text = "Correct!";
                    textField.setEnabled(false);
                    textField.setDisabledTextColor(Color.BLACK);
                    add(playAgain);
            } else if (guess < randNum)
                    text = "Too Low";
            else if (guess > randNum)
                    text = "Too High";


            if ((guess - randNum) > 100 || (guess - randNum) < -100)
                getContentPane().setBackground(Color.CYAN);
            else if (guess == randNum)
                getContentPane().setBackground(Color.YELLOW);
            else
                getContentPane().setBackground(Color.RED);

            status.setText("Status: " + text);
        }
    }

	private void numberGenerator() {
		randNum = 1 + random.nextInt(1000);
	}

    public static void main(String[] args) {
        GuessNum_2614 frame = new GuessNum_2614();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        frame.setResizable(false);
	frame.setSize(400, 275);
	frame.setLocationRelativeTo(null);
	frame.setVisible(true);
    }
    
}
