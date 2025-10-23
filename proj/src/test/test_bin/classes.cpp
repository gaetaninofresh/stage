// rich_classes.cpp
// Compile: g++ -g -O0 -std=c++17 rich_classes.cpp -o rich_classes
// For tougher reverse engineering, compile with -s to strip symbols:
// g++ -O2 -std=c++17 rich_classes.cpp -o rich_classes_stripped && strip rich_classes_stripped

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <typeinfo>
#include <functional>
#include <map>
#include <cassert>

// ---------- Simple enum and POD ----------
enum class Color : int {
    Red = 1,
    Green = 2,
    Blue = 3
};

struct Point {
    int x;
    int y;
    Point(int _x = 0, int _y = 0) : x(_x), y(_y) {}
};

// ---------- Base class with virtual methods ----------
class Animal {
public:
    std::string name;
    int age;
    static int population;

    Animal(const std::string& n, int a) : name(n), age(a) {
        ++population;
        // purposely do something trivial in ctor
        // constructor reference is useful to find class usage
    }

    virtual ~Animal() {
        --population;
    }

    virtual std::string speak() const {
        return "???";
    }

    virtual std::string info() const {
        return "Animal: " + name;
    }

    void birthday() { ++age; }

    // non-virtual helper
    std::string basic_info() const {
        return name + ":" + std::to_string(age);
    }
};
int Animal::population = 0;


// ---------- Derived classes ----------
class Dog : public Animal {
public:
    std::string breed;
    Dog(const std::string& n, int a, const std::string& b) : Animal(n, a), breed(b) {}
    virtual ~Dog() {}

    virtual std::string speak() const override {
        return "Woof!";
    }

    virtual std::string info() const override {
        return "Dog: " + name + " [" + breed + "]";
    }

    void fetch(const std::string& what) {
        std::cout << name << " fetches " << what << std::endl;
    }
};

class Cat : public Animal {
public:
    bool indoor;
    Cat(const std::string& n, int a, bool in) : Animal(n, a), indoor(in) {}
    virtual ~Cat() {}

    virtual std::string speak() const override {
        return "Meow";
    }

    virtual std::string info() const override {
        return std::string("Cat: ") + name + (indoor ? " (indoor)" : " (outdoor)");
    }

    void scratch() {
        std::cout << name << " scratches the couch!" << std::endl;
    }
};

// ---------- Multiple inheritance ----------
class Walker {
public:
    virtual void walk() {
        std::cout << "Walking generically\n";
    }
    virtual ~Walker(){}
};

class Fish : public Animal, public Walker {
public:
    double depth;
    Fish(const std::string& n, int a, double d) : Animal(n,a), depth(d) {}
    virtual ~Fish(){}

    virtual std::string speak() const override {
        // fish don't speak, but return something anyway
        return "...glub...";
    }

    virtual std::string info() const override {
        return "Fish: " + name + " at depth " + std::to_string(depth);
    }

    virtual void walk() override { // nonsense but forces vtable entries for multiple inheritance
        std::cout << name << " flops awkwardly" << std::endl;
    }
};

// ---------- Template class that holds objects ----------
template<typename T>
class Holder {
public:
    T value;
    Holder(const T& v) : value(v) {}
    ~Holder() {}

    T get() const { return value; }
    void set(const T& v) { value = v; }
};

// ---------- Class with pointer fields, function pointers and friend ----------
class SecretKeeper {
private:
    std::string secret;

    // friend function will access private
    friend void reveal_secret(const SecretKeeper& s);

public:
    void (*notifier)(const std::string& msg); // function pointer field
    std::function<void(const std::string&)> lambda_cb; // std::function field

    SecretKeeper(const std::string& s) : secret(s), notifier(nullptr) {}

    void set_callback(std::function<void(const std::string&)> cb) {
        lambda_cb = cb;
    }

    void notify(const std::string& msg) const {
        if(notifier) notifier(msg.c_str());
        if(lambda_cb) lambda_cb(msg);
    }

    void store_secret(const std::string& s) { secret = s; }
};

// friend function definition
void reveal_secret(const SecretKeeper& s) {
    std::cout << "SECRET: " << s.secret << std::endl;
}

// ---------- Manager that references many classes ----------
class Zoo {
private:
    std::vector<std::unique_ptr<Animal>> animals;
    std::map<std::string, Animal*> lookup;

public:
    Zoo() {}
    ~Zoo() {}

    void add_animal(std::unique_ptr<Animal> a) {
        lookup[a->name] = a.get();
        animals.push_back(std::move(a));
    }

    Animal* find(const std::string& name) const {
        auto it = lookup.find(name);
        if (it != lookup.end()) return it->second;
        return nullptr;
    }

    void all_speak() const {
        for (const auto& ap : animals) {
            std::cout << ap->name << " says " << ap->speak() << std::endl;
        }
    }

    // purposely inline a function that references vtables via dynamic_cast / RTTI
    void print_types() const {
        for (const auto& ap : animals) {
            const Animal* a = ap.get();
            std::cout << a->name << " type: " << typeid(*a).name() << std::endl;
        }
    }
};

// ---------- Functions that create objects & reference methods ----------
void create_and_register(Zoo& z) {
    // constructs many objects and registers them
    z.add_animal(std::make_unique<Dog>("Rex", 5, "German Shepherd"));
    z.add_animal(std::make_unique<Cat>("Mittens", 3, true));
    z.add_animal(std::make_unique<Fish>("Nemo", 1, 12.5));
}

void use_objects(Zoo& z) {
    Animal* r = z.find("Rex");
    if (r) {
        std::cout << r->info() << std::endl;
        // downcast to Dog if possible
        Dog* d = dynamic_cast<Dog*>(r);
        if (d) {
            d->fetch("ball");
        }
    }

    Animal* m = z.find("Mittens");
    if (m) {
        std::cout << m->info() << std::endl;
        Cat* c = dynamic_cast<Cat*>(m);
        if (c) c->scratch();
    }
}

void dynamic_list_demo() {
    Zoo z;
    create_and_register(z);
    use_objects(z);
    z.all_speak();
    z.print_types();
}

// ---------- Some global objects and functions that reference them ----------
SecretKeeper GLOBAL_SECRET("s3cr3t!");
void global_notifier(const std::string& msg) {
    std::cout << "GLOBAL notifier: " << msg << std::endl;
}

void setup_global() {
    GLOBAL_SECRET.notifier = &global_notifier;
    GLOBAL_SECRET.set_callback([](const std::string& m) {
        std::cout << "lambda callback: " << m << std::endl;
    });
}

// ---------- Function that uses templates and PODs ----------
void geometry_demo() {
    Holder<Point> h(Point(10, 20));
    Point p = h.get();
    std::cout << "Point: " << p.x << "," << p.y << std::endl;

    Holder<std::string> hs(std::string("hello"));
    std::cout << "Holder contains: " << hs.get() << std::endl;
}

// ---------- Overloaded, inline, static methods ----------
class Utility {
public:
    static int add(int a, int b) { return a + b; }
    static double add(double a, double b) { return a + b; }

    inline static std::string tag = "UTIL";
    static void print_tag() {
        std::cout << "Tag=" << tag << std::endl;
    }
};

// ---------- Exception usage ----------
void throw_demo() {
    try {
        throw std::runtime_error("simulated error");
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << std::endl;
    }
}

// ---------- Main heavy function that touches everything ----------
int main(int argc, char** argv) {
    std::cout << "Program start\n";
    setup_global();
    GLOBAL_SECRET.notify("Hello world");
    reveal_secret(GLOBAL_SECRET);

    Zoo zoo;
    create_and_register(zoo);
    use_objects(zoo);

    geometry_demo();

    // template holder of animals (copying pointers / values)
    Holder<std::string> hs("sample");
    std::cout << "Holder: " << hs.get() << std::endl;

    // Demonstrate static and overloaded functions
    std::cout << "Sum int: " << Utility::add(2,3) << " Sum double: " << Utility::add(1.5, 2.5) << std::endl;
    Utility::print_tag();

    // dynamic container of heterogeneous objects
    dynamic_list_demo();

    // some calls to cause destructor sequences at program end
    std::cout << "Animal population before exit: " << Animal::population << std::endl;

    throw_demo();

    std::cout << "Program end\n";
    return 0;
}
