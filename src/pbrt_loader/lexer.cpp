#include "lexer.h"

#include <fmt/format.h>

Lexeme
Lexer::peek() {
    if (lexeme_buf.has_value()) {
        return lexeme_buf.value();
    } else {
        const auto lex = next();
        lexeme_buf = lex;
        return lexeme_buf.value();
    }
}

Lexeme
Lexer::next() {
    if (lexeme_buf.has_value()) {
        const auto val = lexeme_buf.value();
        lexeme_buf = {};
        return val;
    }

    const auto next_char_maybe_whitespace = peek_char();
    if (!next_char_maybe_whitespace.has_value()) {
        return Lexeme(LexemeType::Eof);
    }

    auto ch_maybe_whitespace = next_char_maybe_whitespace.value();
    if (ch_maybe_whitespace == '#' || std::isspace(ch_maybe_whitespace)) {
        skip_whitespace_and_comments();
    }

    const auto next_char = peek_char();
    if (!next_char.has_value()) {
        return Lexeme(LexemeType::Eof);
    }
    const auto ch = next_char.value();

    if (ch == '"') {
        advance();
        return Lexeme(LexemeType::Quotes);
    } else if (ch == '[') {
        advance();
        return Lexeme(LexemeType::OpenBracket);
    } else if (ch == ']') {
        advance();
        return Lexeme(LexemeType::CloseBracket);
    } else if (ch == '-' || ch == '.' || std::isdigit(ch)) {
        return lex_num();
    } else if (std::isalpha(ch)) {
        return lex_string();
    } else {
        throw std::runtime_error(fmt::format("invalid character: {}", ch));
    }
}

Lexeme
Lexer::lex_string() {
    const auto substr =
        advance_while([](const char ch) { return ch != '"' && !std::isspace(ch); });

    return Lexeme(LexemeType::String, substr);
}

Lexeme
Lexer::lex_num() {
    const auto substr = advance_while([](const char ch) {
        return ch == '-' || ch == '.' || ch == 'e' || std::isdigit(ch);
    });

    return Lexeme(LexemeType::Num, substr);
}

void
Lexer::skip_whitespace_and_comments() {
    while (peek_char().has_value()) {
        const auto next_ch = peek_char().value();
        if (next_ch == '#') {
            advance_while([](const char ch) { return ch != '\n'; });
            // skip the potential newline
            advance();
        } else if (std::isspace(next_ch)) {
            advance_while([](const char ch) { return std::isspace(ch); });
        } else {
            break;
        }
    }
}

std::optional<char>
Lexer::peek_char() const {
    if (!src.empty()) {
        return src.at(0);
    } else {
        return {};
    }
}

void
Lexer::advance() {
    if (!src.empty()) {
        src.remove_prefix(1);
    }
}
