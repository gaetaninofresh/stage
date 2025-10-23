
#include <iostream>

void foo() {
    std::cout << "foo()" << std::endl;
}

void bar() {
    std::cout << "bar()" << std::endl;
    foo();
}

void baz() {
    std::cout << "baz()" << std::endl;
    bar();
}

void rec(int i) {
    if (i > 0) {
        rec(--i);
    }
    else {
        return;
    }
}

int main() {
    std::cout << "main()" << std::endl;
    baz();
    foo();
    rec(200);
    return 0;
}
