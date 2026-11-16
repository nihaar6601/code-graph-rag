package examples;

class Outer {
    Outer() {}

    int calculate(int value) {
        return value * 2;
    }

    int calculate(int first, int second) {
        return first + second;
    }

    class Inner {
        String describe(String prefix) {
            return prefix.trim();
        }
    }
}
