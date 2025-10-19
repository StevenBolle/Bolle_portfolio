package retargeted.assignment5_problem_4_avl_tree;

/**
 * Author: Steven Bolle
 * Professor Syed Rizvi
 * CIS-350 Data Structures and Algorithms
 * Assignment 5: Problem 4 AVL Tree
 */
public class Assignment5_problem_4_avl_tree {

    public static void main(String[] args) {
        System.out.println("-------------- AVL Tree --------------");
        System.out.println("--------------------------------------");
        
        // Create a new AVL tree
        AVLTreeMap avl = new AVLTreeMap();
        
        // Part 1: Insert key-value pairs
        System.out.println("Insert operations beginning: \n");
        avl.insert(5, 2);
        avl.insert(7, 3);
        avl.insert(15, 5);
        avl.insert(20, 7);
        avl.insert(25, 4);
        avl.insert(17, 7);
        avl.insert(32, 10);
        avl.insert(44, 10);
        avl.insert(48, 3);
        avl.insert(50, 5);
        avl.insert(62, 21);
        avl.insert(78, 29);
        avl.insert(88, 6);
        avl.insert(62, 40); // Note: This will update the value of key 62
        avl.insert(90, 7);
        
        System.out.println("\nFinal AVL Tree after insertions:");
        avl.printTree();
        
        // Part 2: Delete keys
        System.out.println("\n****************** Deletion of keys ***********************\n");
        System.out.println("************************************************************");
        avl.delete(32);
        avl.delete(88);
        avl.delete(90);
        
        System.out.println("\nFinal AVL Tree after deletions:");
        avl.printTree();
    } // end main
} // end package class
