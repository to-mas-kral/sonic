#include "lexer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("lexer lookat", "[lexer lookat]") {
    constexpr auto input = std::string_view("LookAt 3 4 1.5  # eye\n"
                                            ".5 .5 0  # look at point\n"
                                            "0 0 1    # up vector\n"
                                            "Camera \"perspective\" \"float fov\" 45");

    auto lexer = Lexer(input);

    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("LookAt")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("3")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("4")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("1.5")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view(".5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view(".5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("0")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("0")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("0")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("1")));

    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("Camera")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("perspective")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("float")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("fov")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Quotes));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("45")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer comments newlines", "[lexer comments newlines]") {
    constexpr auto input = std::string_view("#\n"
                                            "Camera\n"
                                            "# dsds dsdsdsd s ds sdd s Sampler\n"
                                            "#\n"
                                            "WorldBegin\n");

    auto lexer = Lexer(input);

    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("Camera")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::String, std::string_view("WorldBegin")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer floats", "[lexer floats]") {
    constexpr auto input = std::string_view(".05 -0.5 -0. 1. 5 1.91069e-15\n");

    auto lexer = Lexer(input);

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view(".05")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("-0.5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("-0.")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("1.")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("1.91069e-15")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer peek", "[lexer peek]") {
    constexpr auto input = std::string_view("1 2 3 4 5\n");

    auto lexer = Lexer(input);

    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("1")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string_view("2")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string_view("2")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("2")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("3")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string_view("4")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("4")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string_view("5")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Num, std::string_view("5")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view("5")));
    REQUIRE(lexer.peek() == Lexeme(LexemeType::Eof));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}

TEST_CASE("lexer brackets", "[lexer brackets]") {
    constexpr auto input = std::string_view("[.1 .1 .1]\n");

    auto lexer = Lexer(input);

    REQUIRE(lexer.next() == Lexeme(LexemeType::OpenBracket));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view(".1")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view(".1")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Num, std::string_view(".1")));
    REQUIRE(lexer.next() == Lexeme(LexemeType::CloseBracket));
    REQUIRE(lexer.next() == Lexeme(LexemeType::Eof));
}
