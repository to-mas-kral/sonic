#include "lexer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("lexer lookat", "[lexer lookat]") {
    auto input = std::string("LookAt 3 4 1.5  # eye\n"
                             ".5 .5 0  # look at point\n"
                             "0 0 1    # up vector\n"
                             "Camera \"perspective\" \"float fov\" 45");

    auto stream = StackFileStream(input);
    auto lexer = Lexer(&stream);

    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("LookAt")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("3")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("4")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("1.5")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string(".5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string(".5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("0")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("0")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("0")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("1")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("Camera")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("perspective")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("float")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("fov")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("45")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer comments newlines", "[lexer comments newlines]") {
    auto input = std::string("#\n"
                             "Camera\n"
                             "# dsds dsdsdsd s ds sdd s Sampler\n"
                             "#\n"
                             "WorldBegin\n");

    auto stream = StackFileStream(input);
    auto lexer = Lexer(&stream);

    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("Camera")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string("WorldBegin")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer floats", "[lexer floats]") {
    auto input = std::string(".05 -0.5 -0. 1. 5 1.91069e-15\n");

    auto stream = StackFileStream(input);
    auto lexer = Lexer(&stream);

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string(".05")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("-0.5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("-0.")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("1.")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("1.91069e-15")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer peek", "[lexer peek]") {
    auto input = std::string("1 2 3 4 5\n");

    auto stream = StackFileStream(input);
    auto lexer = Lexer(&stream);

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("1")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string("2")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string("2")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("2")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("3")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string("4")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("4")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string("5")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string("5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string("5")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Eof));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer brackets", "[lexer brackets]") {
    auto input = std::string("[.1 .1 .1]\n");

    auto stream = StackFileStream(input);
    auto lexer = Lexer(&stream);

    REQUIRE(lexer.next() == Lexeme(LexemeType::OpenBracket));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string(".1")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string(".1")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string(".1")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::CloseBracket));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}
