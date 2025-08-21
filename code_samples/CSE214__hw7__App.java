 

public class App {
    public static void main(String[] args) {
        System.out.print("Enter an expression: ");
        java.util.Scanner sc = new java.util.Scanner(System.in);
        String expr = sc.nextLine();
        sc.close();
        System.out.println("eval: " + Parser.evalExpr(Parser.parseExpr(expr)));
        System.out.println("preorder: " + Parser.infixToPrefix(expr));
        System.out.println("postorder: " + Parser.infixToPostfix(expr));
        System.out.println("inorder: " + Parser.infixToInfix(expr));
    }
}
