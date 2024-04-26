#ifndef LEXER_H
#define LEXER_H

#include <optional>
#include <stdexcept>
#include <string_view>

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

    Lexeme(const LexemeType type, const std::string_view &src) : type(type), src(src) {}

    bool
    operator==(const Lexeme &other) const {
        return type == other.type && src == other.src;
    }

    LexemeType type;
    std::string_view src{};
};

class Lexer {
public:
    explicit
    Lexer(const std::string_view &src)
        : src(src) {}

    Lexeme
    peek();

    Lexeme
    next();

private:
    template <typename F>
    std::string_view
    advance_while(F test) {
        auto chars_matched = 0;
        const auto size = src.size();

        while (chars_matched < size) {
            const auto ch = src.at(chars_matched);
            if (!test(ch)) {
                break;
            }

            chars_matched++;
        }

        if (chars_matched == 0) {
            throw std::runtime_error("lexeme inner error: 0 matched chars");
        }

        const auto substr = src.substr(0, chars_matched);
        src.remove_prefix(chars_matched);
        return substr;
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
    advance();

    std::string_view src{};
    std::optional<Lexeme> lexeme_buf{};
};

#endif // LEXER_H
