#ifndef LEXER_H
#define LEXER_H

#include <optional>
#include <stdexcept>
#include <utility>

#include "../utils/basic_types.h"

#include <istream>

enum class LexemeType {
    String,
    Quotes,
    OpenBracket,
    CloseBracket,
    Num,
    Eof,
};

struct Lexeme {
    explicit
    Lexeme(const LexemeType type)
        : type(type) {}

    Lexeme(const LexemeType type, std::string src) : type(type), src(std::move(src)) {}

    bool
    operator==(const Lexeme &other) const {
        return type == other.type && src == other.src;
    }

    LexemeType type;
    std::string src{};
};

class Lexer {
public:
    explicit
    Lexer(std::istream *src)
        : src{src} {}

    Lexeme
    peek();

    Lexeme
    next();

private:
    template <typename F>
    std::string
    advance_while(F test) {
        // TODO: tweak the default size of this...
        std::string str{};

        while (true) {
            const auto next_ch = peek_char();
            if (!next_ch.has_value()) {
                break;
            }

            const auto ch = next_ch.value();

            if (!test(ch)) {
                break;
            }

            advance();
            str.push_back(ch);
        }

        if (str.empty()) {
            throw std::runtime_error("lexeme inner error: 0 matched chars");
        }

        return str;
    }

    Lexeme
    lex_string();

    Lexeme
    lex_num();

    void
    skip_whitespace_and_comments();

    std::optional<char>
    peek_char() const;

    void
    advance() const;

    // TODO: there might be some overhead to using istreams... could roll my own
    std::istream *src;
    std::optional<Lexeme> lexeme_buf{};
};

#endif // LEXER_H
