package retargeted.assignment5_problem_4_avl_tree;

/**
 * Author: Steven Bolle
 * Professor Syed Rizvi
 * CIS-350 Data Structures and Algorithms
 * Assignment 5: Problem 4 AVL Tree
 */

import java.util.*;

public class AVLTreeMap {
    public class Node {
        public int key;
        public int value;
        public Node left;
        public Node right;
        public Node parent;
        public int height;
        
        public Node(int k, int v) {
            key = k;
            value = v;
            height = 1;
        }
    } // end Node class declaration
    
    // establish root and build tree from it
    public Node root;
    
    // node vitals and balance factor methods
    public void updateHeight(Node node) {
        if (node == null) return;
        node.height = 1 + Math.max(getHeight(node.left), getHeight(node.right));
    } // end updateHeight
    
    public int getHeight(Node node) {
        return (node == null) ? 0 : node.height;
    } // end getHeight
    
    public int getBalance(Node node) {
        if (node == null) return 0;
        return getHeight(node.left) - getHeight(node.right);
    } // end getBalance
    
    public void reportBal(Node node) {
        int balFactor = getBalance(node);
        if (balFactor > 1 || balFactor < -1) {
            System.out.println("Tree is out of balance at node key: " + node.key );
        }
    } // end reportBal function
    
    // node rotation for tree imbalances    
    public Node rotateRight(Node y) {
        System.out.println("\nNode key right rotation at: " + y.key);
        Node x = y.left;
        Node z = x.right;
        
        x.right = y;
        y.left = z;
        if (z != null) z.parent = y;
        
        x.parent = y.parent;
        y.parent = x;
        
        if (x.parent == null) {
            root = x;
        } // end if
        else {
            if (x.parent.left == y) {
            x.parent.left = x;
        } // end else
            else {
                x.parent.right = x;
            } // end nested else
        } // end else
        updateHeight(y);
        updateHeight(x);
        
        printTree();
        return x;
    } // end rotateright
    
    public Node rotateLeft(Node x) {
        System.out.println("Node key left rotation at: " + x.key);
        Node y = x.right;
        Node z = y.left;
        
        y.left = x;
        x.right = z;
        if (z != null) z.parent = x;
        
        y.parent = x.parent;
        x.parent = y;
        
        if (y.parent == null) {
            root = y;
        }
        else {
            if (y.parent.left == x) {
            y.parent.left = y;
        }
            else {
                y.parent.right = y;
            }
        }
        updateHeight(y);
        updateHeight(x);
        
        printTree();
        return y;
    } // end rotateleft
    
    public Node triNodeRestructure(Node node) {
        System.out.println("\nTriNode Restructure at key: " + node.key);
        int balFactor = getBalance(node);
        if (balFactor > 1) {
            if (getBalance(node.left) < 0) {
                node.left = rotateLeft(node.left);
                node.left.parent = node;
                return rotateRight(node);
            } // end if
            else {
                return rotateRight(node);
            } // end else
        } // end if
        else if (balFactor < -1) {
                if (getBalance(node.right) > 0) {
                    node.right = rotateRight(node.right);
                    node.right.parent = node;
                    return rotateLeft(node);
                }
                else {
                    return rotateLeft(node);
                } // end else
        } // end else if
        return node;
    } // end trinodeRestructure
    
    
    // insert and delete methods
    public void insert(int key, int value) {
        root = insertHelp(root , key, value, null);
        System.out.println("\nInserted key " + key + " into AVL Tree " + "with value: " + value);
        printTree();
        
    } // end insert function
    
    public Node insertHelp(Node node, int key, int value, Node parent) {
        if (node == null) {
            Node newNode = new Node(key, value);
            newNode.parent = parent;
            return newNode;
        } // end if
        if (key < node.key) {
            node.left = insertHelp(node.left, key, value, node);
        } // end in
        else if (key > node.key) {
            node.right = insertHelp(node.right, key, value, node);
        } // end else if
        else {
            node.value = value;
            return node;
        } // end else
        
        updateHeight(node);
        reportBal(node);
        
        int balFactor = getBalance(node);        
        if (balFactor > 1 || balFactor < -1) {
            node = triNodeRestructure(node);
        } // end if
        
//        Node current = node.parent;
//        while (current != null) {
//            updateHeight(current);
//            reportBal(current);
//            if (getBalance(current) > 1 || getBalance(current) < -1) {
//                current = triNodeRestructure(current);
//            }
//            current = current.parent;
//        }
        //printTree();
        updateHeight(root);
        return node;
    } // end insertHelp
    
    public void delete(int key) {
        System.out.println("\nAttempt to delete key: " + key);
        printTree();
        root = deleteHelp(root, key);
        System.out.println(" ******************** Tree after deletion ************************");
        System.out.println("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-");
        printTree();
    } // end delete function
    
    public Node deleteHelp(Node node, int key) {
        if (node == null) return null;
        if (key < node.key) {
            node.left = deleteHelp(node.left, key);
        } // end if
        else if (key > node.key) {
            node.right = deleteHelp(node.right, key);
        } // end else if
        else {
            if (node.left == null || node.right == null) {
                Node temporary = (node.left != null) ? node.left : node.right;
                if (temporary == null) {
                    node = null;
                } // end if
                else {
                    temporary.parent = node.parent;
                    node = temporary;
                } // end else
            } // end if
            else {
                Node nextKin = minNode(node.right);
                node.key = nextKin.key;
                node.value = nextKin.value;
                node.right = deleteHelp(node.right, nextKin.key);
            } // end else
        } // end else
        
        if (node == null) return null;
        updateHeight(node);
        reportBal(node);
        
        int balFactor = getBalance(node);
        if (balFactor > 1 || balFactor < -1) {
            node = triNodeRestructure(node);
        } // end if
        
//        Node current = node.parent;
//        while (current != null) {
//            updateHeight(current);
//            reportBal(current);
//            if (getBalance(current) > 1 || getBalance(current) < -1) {
//                current = triNodeRestructure(current);
//            }
//            current = current.parent;
//        }
        
        return node;
    } // end deleteHelp function
    
    public Node minNode(Node node) {
        Node curr = node;
        while (curr.left != null) {
            curr = curr.left;
        } // end while
        return curr;
    } // end minNode function
    
    // referenced https://github.com/harshshredding/A-beautiful-tree/blob/master/Tree.java
    public void printTree() {
        System.out.println("\nCurrent AVL Tree Structure:");
        if (root == null) {
            System.out.println("[empty]");
            return;
        } // end if

        int h = getHeight(root);
        System.out.println("Current height of tree: " + h +"\n");
        // put avl nodes in a linked list/queue
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        int level = 0;
        boolean moreLevels = true;

        while (level < h && moreLevels) {
            int levelCount = queue.size(); // nodes in this AVL level

            // offset for printing of first node on a level
            int firstIndent = (int) Math.pow(2, (h - level - 1)) - 1; // left margin
            int betweenSpacing = (int) Math.pow(2, (h - level)) - 1;  // spacing between nodes
        
            // adds edges between nodes
            StringBuilder nodeLine = new StringBuilder();
            StringBuilder edgeLine = new StringBuilder();

            // checks if next level is needed
            boolean foundNonNull = false;

            for (int i = 0; i < levelCount; i++) {
                Node node = queue.poll();

                // For the first node on this line
                if (i == 0) {
                    nodeLine.append(" ".repeat(firstIndent * 2)); // each 'unit' is about 2 spaces
                    edgeLine.append(" ".repeat(firstIndent * 2));
                } // end if 
                else {
                    // For subsequent nodes
                    nodeLine.append(" ".repeat(betweenSpacing * 2));
                    edgeLine.append(" ".repeat(betweenSpacing * 2));
                } // end else

                if (node == null) {
                    nodeLine.append("     "); // blank for null
                    edgeLine.append("      ");
                    queue.add(null); // add dummy children
                    queue.add(null);
                } // end if 
                else {
                    foundNonNull = true;
                    // logic for node info
                    String label = String.format("%d:%d(h=%d)", node.key, node.value, node.height);
                    nodeLine.append(String.format("%-11s", label));
                    // logic for edges
                    if (node.left != null || node.right != null) {
                        // adds edges from parent to child
                        edgeLine.append("  ");
                        edgeLine.append((node.left != null) ? "/" : "      ");
                        edgeLine.append("   ");
                        edgeLine.append((node.right != null) ? "\\" : "     ");
                    } // end if 
                    else {
                        edgeLine.append("     ");
                    } // end else
                    // put children in queue
                    queue.add(node.left);
                    queue.add(node.right);
                } // end else
            } // end for loop

            // print the node line
            System.out.println(nodeLine.toString());
            // if and node has a key/value, print an edge
            if (level < h - 1) { 
                System.out.println(edgeLine.toString());
            } // end if

            if (!foundNonNull) {
            moreLevels = false;
            } // end if
            level++;
        } // end while
    } // end printTree function
} // end avlTree class