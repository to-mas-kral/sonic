#ifndef STACK_FILE_STREAM_H
#define STACK_FILE_STREAM_H

#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

#include "../utils/basic_types.h"

struct SourceLocation {
    std::filesystem::path file_path;
    u32 line_counter{1};
};

struct CurrentFileStream {
    std::vector<char> buf = std::vector<char>(8192 * 1000);
    std::ifstream file_stream;
    SourceLocation src_location{};
};

class StackFileStream {
public:
    explicit
    StackFileStream(const std::filesystem::path &file_path) {
        file_streams.push_back(CurrentFileStream{});
        auto &stream = file_streams[file_streams.size() - 1];

        stream.file_stream.rdbuf()->pubsetbuf(stream.buf.data(), stream.buf.size());
        stream.file_stream.open(file_path);
    }

    explicit
    StackFileStream(const std::string &input)
        : is_string_input{true}, string_stream{input} {}

    void
    push_file(const std::filesystem::path &file_path) {
        file_streams.push_back(
            CurrentFileStream{.src_location = SourceLocation{.file_path = file_path}});
        auto &stream = file_streams[file_streams.size() - 1];

        stream.file_stream.rdbuf()->pubsetbuf(stream.buf.data(), stream.buf.size());
        stream.file_stream.open(file_path);
    }

    SourceLocation &
    src_location() {
        if (!is_string_input) {
            if (file_streams.empty()) {
                throw std::runtime_error("StackFileStream is empty");
            }

            return file_streams.back().src_location;
        } else {
            return string_source_location;
        }
    }

    void
    inc_line_counter() {
        if (!is_string_input) {
            if (file_streams.empty()) {
                throw std::runtime_error("StackFileStream is empty");
            }

            file_streams.back().src_location.line_counter++;
        } else {
            string_source_location.line_counter++;
        }
    }

    bool
    is_eof() {
        if (!is_string_input) {
            retire_eof_streams();

            if (file_streams.empty()) {
                return true;
            }

            return file_streams.back().file_stream.eof();
        } else {
            return string_stream.eof();
        }
    }
    
    std::optional<char>
    peek() {
        if (!is_string_input) {
            retire_eof_streams();

            if (file_streams.empty()) {
                return {};
            }

            return file_streams.back().file_stream.peek();
        } else {
            const auto ch = string_stream.peek();
            // istringstream seems to return -1 for the last character...
            if (ch == -1) {
                return {};
            } else {
                return ch;
            }
        }
    }

    char
    get() {
        if (!is_string_input) {
            retire_eof_streams();

            if (file_streams.empty()) {
                return EOF;
            }

            return file_streams.back().file_stream.get();
        } else {
            return string_stream.get();
        }
    }

    void
    retire_eof_streams() {
        while (!file_streams.empty()) {
            if (file_streams.back().file_stream.eof()) {
                file_streams.pop_back();
            } else {
                break;
            }
        }
    }

private:
    bool is_string_input = false;
    std::istringstream string_stream;
    SourceLocation string_source_location{};
    std::vector<CurrentFileStream> file_streams;
};

#endif // STACK_FILE_STREAM_H
