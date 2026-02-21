#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/emel.h"
#include "emel/jinja/ast.hpp"
#include "emel/jinja/lexer.hpp"

namespace emel::jinja::parser::detail {

template <class T>
static bool is_type(const emel::jinja::ast_ptr & ptr) {
  return dynamic_cast<const T *>(ptr.get()) != nullptr;
}

class recursive_descent_parser {
 public:
  explicit recursive_descent_parser(emel::jinja::program & out_program)
      : program_(out_program) {}

  bool parse(const emel::jinja::lexer_result & lexer_res) {
    tokens_ = &lexer_res.tokens;
    current_ = 0;
    error_ = EMEL_OK;
    error_pos_ = 0;
    program_.reset();

    while (current_ < tokens_->size() && ok()) {
      auto stmt = parse_any();
      if (!ok() || !stmt) {
        break;
      }
      program_.body.push_back(std::move(stmt));
    }

    if (!ok()) {
      program_.reset();
      program_.last_error = error_;
      program_.last_error_pos = error_pos_;
      return false;
    }

    program_.last_error = EMEL_OK;
    program_.last_error_pos = 0;
    return true;
  }

  int32_t error() const noexcept { return error_; }
  size_t error_pos() const noexcept { return error_pos_; }

 private:
  const emel::jinja::token & peek(size_t offset = 0) const {
    static const emel::jinja::token end_token{emel::jinja::token_type::Eof, "", 0};
    if (tokens_ == nullptr || current_ + offset >= tokens_->size()) {
      return end_token;
    }
    return (*tokens_)[current_ + offset];
  }

  bool ok() const noexcept { return error_ == EMEL_OK; }

  void set_error(int32_t code, size_t pos) {
    if (error_ != EMEL_OK) {
      return;
    }
    error_ = code;
    error_pos_ = pos;
  }

  bool expect(emel::jinja::token_type type) {
    if (!ok()) {
      return false;
    }
    const auto & t = peek();
    if (t.type != type) {
      set_error(EMEL_ERR_PARSE_FAILED, t.pos);
      return false;
    }
    ++current_;
    return true;
  }

  bool expect_identifier(std::string_view name) {
    if (!ok()) {
      return false;
    }
    const auto & t = peek();
    if (t.type != emel::jinja::token_type::Identifier || t.value != name) {
      set_error(EMEL_ERR_PARSE_FAILED, t.pos);
      return false;
    }
    ++current_;
    return true;
  }

  bool is(emel::jinja::token_type type) const {
    return peek().type == type;
  }

  bool is_identifier(std::string_view name) const {
    const auto & t = peek();
    return t.type == emel::jinja::token_type::Identifier && t.value == name;
  }

  bool is_statement(std::initializer_list<std::string_view> names) const {
    if (peek(0).type != emel::jinja::token_type::OpenStatement ||
        peek(1).type != emel::jinja::token_type::Identifier) {
      return false;
    }
    const std::string & val = peek(1).value;
    for (const auto & name : names) {
      if (val == name) {
        return true;
      }
    }
    return false;
  }

  template <class T, class... Args>
  emel::jinja::ast_ptr make_node(size_t start_pos, Args &&... args) {
    auto node = std::make_unique<T>(std::forward<Args>(args)...);
    if (tokens_ != nullptr && start_pos < tokens_->size()) {
      node->pos = (*tokens_)[start_pos].pos;
    } else {
      node->pos = 0;
    }
    return node;
  }

  emel::jinja::ast_ptr parse_any() {
    size_t start_pos = current_;
    const auto & t = peek();
    switch (t.type) {
      case emel::jinja::token_type::Comment:
        ++current_;
        return make_node<emel::jinja::comment_statement>(start_pos, t.value);
      case emel::jinja::token_type::Text:
        ++current_;
        return make_node<emel::jinja::string_literal>(start_pos, t.value);
      case emel::jinja::token_type::OpenStatement:
        return parse_jinja_statement();
      case emel::jinja::token_type::OpenExpression:
        return parse_jinja_expression();
      default:
        set_error(EMEL_ERR_PARSE_FAILED, t.pos);
        return nullptr;
    }
  }

  emel::jinja::ast_ptr parse_jinja_expression() {
    if (!expect(emel::jinja::token_type::OpenExpression)) {
      return nullptr;
    }
    auto result = parse_expression();
    if (!expect(emel::jinja::token_type::CloseExpression)) {
      return nullptr;
    }
    return result;
  }

  emel::jinja::ast_ptr parse_jinja_statement() {
    if (!expect(emel::jinja::token_type::OpenStatement)) {
      return nullptr;
    }

    const auto & name_token = peek();
    if (name_token.type != emel::jinja::token_type::Identifier) {
      set_error(EMEL_ERR_PARSE_FAILED, name_token.pos);
      return nullptr;
    }

    size_t start_pos = current_;
    std::string name = name_token.value;
    ++current_;

    emel::jinja::ast_ptr result;
    if (name == "set") {
      result = parse_set_statement(start_pos);
    } else if (name == "if") {
      result = parse_if_statement(start_pos);
      if (!expect(emel::jinja::token_type::OpenStatement)) {
        return nullptr;
      }
      if (!expect_identifier("endif")) {
        return nullptr;
      }
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
    } else if (name == "macro") {
      result = parse_macro_statement(start_pos);
      if (!expect(emel::jinja::token_type::OpenStatement)) {
        return nullptr;
      }
      if (!expect_identifier("endmacro")) {
        return nullptr;
      }
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
    } else if (name == "for") {
      result = parse_for_statement(start_pos);
      if (!expect(emel::jinja::token_type::OpenStatement)) {
        return nullptr;
      }
      if (!expect_identifier("endfor")) {
        return nullptr;
      }
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
    } else if (name == "break") {
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      result = make_node<emel::jinja::break_statement>(start_pos);
    } else if (name == "continue") {
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      result = make_node<emel::jinja::continue_statement>(start_pos);
    } else if (name == "call") {
      emel::jinja::ast_list caller_args;
      if (is(emel::jinja::token_type::OpenParen)) {
        caller_args = parse_args();
      }
      auto callee = parse_primary_expression();
      if (!ok() || !callee || !is_type<emel::jinja::identifier>(callee)) {
        set_error(EMEL_ERR_PARSE_FAILED, peek().pos);
        return nullptr;
      }
      auto call_args = parse_args();
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }

      emel::jinja::ast_list body;
      while (ok() && !is_statement({"endcall"})) {
        body.push_back(parse_any());
      }
      if (!expect(emel::jinja::token_type::OpenStatement)) {
        return nullptr;
      }
      if (!expect_identifier("endcall")) {
        return nullptr;
      }
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }

      auto call_expr = make_node<emel::jinja::call_expression>(
          start_pos, std::move(callee), std::move(call_args));
      result = make_node<emel::jinja::call_statement>(
          start_pos, std::move(call_expr), std::move(caller_args), std::move(body));
    } else if (name == "filter") {
      auto filter_node = parse_primary_expression();
      if (!ok()) {
        return nullptr;
      }
      if (filter_node && is_type<emel::jinja::identifier>(filter_node) &&
          is(emel::jinja::token_type::OpenParen)) {
        filter_node = parse_call_expression(std::move(filter_node));
      }
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }

      emel::jinja::ast_list body;
      while (ok() && !is_statement({"endfilter"})) {
        body.push_back(parse_any());
      }
      if (!expect(emel::jinja::token_type::OpenStatement)) {
        return nullptr;
      }
      if (!expect_identifier("endfilter")) {
        return nullptr;
      }
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      result = make_node<emel::jinja::filter_statement>(
          start_pos, std::move(filter_node), std::move(body));
    } else if (name == "generation" || name == "endgeneration") {
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      result = make_node<emel::jinja::noop_statement>(start_pos);
    } else {
      set_error(EMEL_ERR_PARSE_FAILED, name_token.pos);
      return nullptr;
    }
    return result;
  }

  emel::jinja::ast_ptr parse_set_statement(size_t start_pos) {
    auto left = parse_expression_sequence();
    emel::jinja::ast_ptr value = nullptr;
    emel::jinja::ast_list body;

    if (is(emel::jinja::token_type::Equals)) {
      ++current_;
      value = parse_expression_sequence();
    } else {
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      while (ok() && !is_statement({"endset"})) {
        body.push_back(parse_any());
      }
      if (!expect(emel::jinja::token_type::OpenStatement)) {
        return nullptr;
      }
      if (!expect_identifier("endset")) {
        return nullptr;
      }
    }
    if (!expect(emel::jinja::token_type::CloseStatement)) {
      return nullptr;
    }
    return make_node<emel::jinja::set_statement>(
        start_pos, std::move(left), std::move(value), std::move(body));
  }

  emel::jinja::ast_ptr parse_if_statement(size_t start_pos) {
    auto test = parse_expression();
    if (!expect(emel::jinja::token_type::CloseStatement)) {
      return nullptr;
    }

    emel::jinja::ast_list body;
    emel::jinja::ast_list alternate;

    while (ok() && !is_statement({"elif", "else", "endif"})) {
      body.push_back(parse_any());
    }

    if (ok() && is_statement({"elif"})) {
      size_t pos0 = current_;
      current_ += 2;
      alternate.push_back(parse_if_statement(pos0));
    } else if (ok() && is_statement({"else"})) {
      current_ += 2;
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      while (ok() && !is_statement({"endif"})) {
        alternate.push_back(parse_any());
      }
    }
    return make_node<emel::jinja::if_statement>(
        start_pos, std::move(test), std::move(body), std::move(alternate));
  }

  emel::jinja::ast_ptr parse_macro_statement(size_t start_pos) {
    auto name = parse_primary_expression();
    auto args = parse_args();
    if (!expect(emel::jinja::token_type::CloseStatement)) {
      return nullptr;
    }
    emel::jinja::ast_list body;
    while (ok() && !is_statement({"endmacro"})) {
      body.push_back(parse_any());
    }
    return make_node<emel::jinja::macro_statement>(
        start_pos, std::move(name), std::move(args), std::move(body));
  }

  emel::jinja::ast_ptr parse_expression_sequence(bool primary = false) {
    size_t start_pos = current_;
    emel::jinja::ast_list exprs;
    exprs.push_back(primary ? parse_primary_expression() : parse_expression());
    bool is_tuple = is(emel::jinja::token_type::Comma);
    while (ok() && is(emel::jinja::token_type::Comma)) {
      ++current_;
      exprs.push_back(primary ? parse_primary_expression() : parse_expression());
    }
    if (!ok()) {
      return nullptr;
    }
    return is_tuple ? make_node<emel::jinja::tuple_literal>(start_pos, std::move(exprs))
                    : std::move(exprs[0]);
  }

  emel::jinja::ast_ptr parse_for_statement(size_t start_pos) {
    auto loop_var = parse_expression_sequence(true);
    if (!is_identifier("in")) {
      set_error(EMEL_ERR_PARSE_FAILED, peek().pos);
      return nullptr;
    }
    ++current_;
    auto iterable = parse_expression();
    if (!expect(emel::jinja::token_type::CloseStatement)) {
      return nullptr;
    }

    emel::jinja::ast_list body;
    emel::jinja::ast_list alternate;

    while (ok() && !is_statement({"endfor", "else"})) {
      body.push_back(parse_any());
    }

    if (ok() && is_statement({"else"})) {
      current_ += 2;
      if (!expect(emel::jinja::token_type::CloseStatement)) {
        return nullptr;
      }
      while (ok() && !is_statement({"endfor"})) {
        alternate.push_back(parse_any());
      }
    }
    return make_node<emel::jinja::for_statement>(
        start_pos, std::move(loop_var), std::move(iterable),
        std::move(body), std::move(alternate));
  }

  emel::jinja::ast_ptr parse_expression() {
    return parse_if_expression();
  }

  emel::jinja::ast_ptr parse_if_expression() {
    auto a = parse_logical_or_expression();
    if (is_identifier("if")) {
      size_t start_pos = current_;
      ++current_;
      auto test = parse_logical_or_expression();
      if (is_identifier("else")) {
        size_t pos0 = current_;
        ++current_;
        auto false_expr = parse_if_expression();
        return make_node<emel::jinja::ternary_expression>(
            pos0, std::move(test), std::move(a), std::move(false_expr));
      }
      return make_node<emel::jinja::select_expression>(
          start_pos, std::move(a), std::move(test));
    }
    return a;
  }

  emel::jinja::ast_ptr parse_logical_or_expression() {
    auto left = parse_logical_and_expression();
    while (ok() && is_identifier("or")) {
      size_t start_pos = current_;
      auto op = (*tokens_)[current_++];
      left = make_node<emel::jinja::binary_expression>(
          start_pos, std::move(op), std::move(left), parse_logical_and_expression());
    }
    return left;
  }

  emel::jinja::ast_ptr parse_logical_and_expression() {
    auto left = parse_logical_negation_expression();
    while (ok() && is_identifier("and")) {
      size_t start_pos = current_;
      auto op = (*tokens_)[current_++];
      left = make_node<emel::jinja::binary_expression>(
          start_pos, std::move(op), std::move(left), parse_logical_negation_expression());
    }
    return left;
  }

  emel::jinja::ast_ptr parse_logical_negation_expression() {
    if (is_identifier("not")) {
      size_t start_pos = current_;
      auto op = (*tokens_)[current_++];
      return make_node<emel::jinja::unary_expression>(
          start_pos, std::move(op), parse_logical_negation_expression());
    }
    return parse_comparison_expression();
  }

  emel::jinja::ast_ptr parse_comparison_expression() {
    auto left = parse_additive_expression();
    while (ok()) {
      emel::jinja::token op;
      size_t start_pos = current_;
      if (is_identifier("not") && peek(1).type == emel::jinja::token_type::Identifier &&
          peek(1).value == "in") {
        op = {emel::jinja::token_type::Identifier, "not in", peek().pos};
        current_ += 2;
      } else if (is_identifier("in")) {
        op = (*tokens_)[current_++];
      } else if (is(emel::jinja::token_type::ComparisonBinaryOperator)) {
        op = (*tokens_)[current_++];
      } else {
        break;
      }
      left = make_node<emel::jinja::binary_expression>(
          start_pos, std::move(op), std::move(left), parse_additive_expression());
    }
    return left;
  }

  emel::jinja::ast_ptr parse_additive_expression() {
    auto left = parse_multiplicative_expression();
    while (ok() && is(emel::jinja::token_type::AdditiveBinaryOperator)) {
      size_t start_pos = current_;
      auto op = (*tokens_)[current_++];
      left = make_node<emel::jinja::binary_expression>(
          start_pos, std::move(op), std::move(left), parse_multiplicative_expression());
    }
    return left;
  }

  emel::jinja::ast_ptr parse_multiplicative_expression() {
    auto left = parse_test_expression();
    while (ok() && is(emel::jinja::token_type::MultiplicativeBinaryOperator)) {
      size_t start_pos = current_;
      auto op = (*tokens_)[current_++];
      left = make_node<emel::jinja::binary_expression>(
          start_pos, std::move(op), std::move(left), parse_test_expression());
    }
    return left;
  }

  emel::jinja::ast_ptr parse_test_expression() {
    auto operand = parse_filter_expression();
    while (ok() && is_identifier("is")) {
      size_t start_pos = current_;
      ++current_;
      bool negate = false;
      if (is_identifier("not")) {
        ++current_;
        negate = true;
      }
      auto test_id = parse_primary_expression();
      if (is(emel::jinja::token_type::OpenParen)) {
        test_id = parse_call_expression(std::move(test_id));
      }
      operand = make_node<emel::jinja::test_expression>(
          start_pos, std::move(operand), negate, std::move(test_id));
    }
    return operand;
  }

  emel::jinja::ast_ptr parse_filter_expression() {
    auto operand = parse_call_member_expression();
    while (ok() && is(emel::jinja::token_type::Pipe)) {
      size_t start_pos = current_;
      ++current_;
      auto filter = parse_primary_expression();
      if (is(emel::jinja::token_type::OpenParen)) {
        filter = parse_call_expression(std::move(filter));
      }
      operand = make_node<emel::jinja::filter_expression>(
          start_pos, std::move(operand), std::move(filter));
    }
    return operand;
  }

  emel::jinja::ast_ptr parse_call_member_expression() {
    auto member = parse_member_expression(parse_primary_expression());
    if (is(emel::jinja::token_type::OpenParen)) {
      return parse_call_expression(std::move(member));
    }
    return member;
  }

  emel::jinja::ast_ptr parse_call_expression(emel::jinja::ast_ptr callee) {
    size_t start_pos = current_;
    auto expr = make_node<emel::jinja::call_expression>(
        start_pos, std::move(callee), parse_args());
    auto member = parse_member_expression(std::move(expr));
    if (is(emel::jinja::token_type::OpenParen)) {
      return parse_call_expression(std::move(member));
    }
    return member;
  }

  emel::jinja::ast_list parse_args() {
    emel::jinja::ast_list args;
    if (!expect(emel::jinja::token_type::OpenParen)) {
      return args;
    }
    while (ok() && !is(emel::jinja::token_type::CloseParen)) {
      emel::jinja::ast_ptr arg;
      if (peek().type == emel::jinja::token_type::MultiplicativeBinaryOperator &&
          peek().value == "*") {
        size_t start_pos = current_;
        ++current_;
        arg = make_node<emel::jinja::spread_expression>(
            start_pos, parse_expression());
      } else {
        arg = parse_expression();
        if (is(emel::jinja::token_type::Equals)) {
          size_t start_pos = current_;
          ++current_;
          arg = make_node<emel::jinja::keyword_argument_expression>(
              start_pos, std::move(arg), parse_expression());
        }
      }
      args.push_back(std::move(arg));
      if (is(emel::jinja::token_type::Comma)) {
        ++current_;
      }
    }
    expect(emel::jinja::token_type::CloseParen);
    return args;
  }

  emel::jinja::ast_ptr parse_member_expression(emel::jinja::ast_ptr object) {
    size_t start_pos = current_;
    while (ok() && (is(emel::jinja::token_type::Dot) ||
                    is(emel::jinja::token_type::OpenSquareBracket))) {
      auto op = (*tokens_)[current_++];
      bool computed = op.type == emel::jinja::token_type::OpenSquareBracket;
      emel::jinja::ast_ptr prop;
      if (computed) {
        prop = parse_member_expression_arguments();
        if (!expect(emel::jinja::token_type::CloseSquareBracket)) {
          return nullptr;
        }
      } else {
        prop = parse_primary_expression();
      }
      object = make_node<emel::jinja::member_expression>(
          start_pos, std::move(object), std::move(prop), computed);
    }
    return object;
  }

  emel::jinja::ast_ptr parse_member_expression_arguments() {
    emel::jinja::ast_list slices;
    bool is_slice = false;
    size_t start_pos = current_;
    while (ok() && !is(emel::jinja::token_type::CloseSquareBracket)) {
      if (is(emel::jinja::token_type::Colon)) {
        slices.push_back(nullptr);
        ++current_;
        is_slice = true;
      } else {
        slices.push_back(parse_expression());
        if (is(emel::jinja::token_type::Colon)) {
          ++current_;
          is_slice = true;
        }
      }
    }
    if (!ok()) {
      return nullptr;
    }
    if (is_slice) {
      emel::jinja::ast_ptr start = slices.size() > 0 ? std::move(slices[0]) : nullptr;
      emel::jinja::ast_ptr stop = slices.size() > 1 ? std::move(slices[1]) : nullptr;
      emel::jinja::ast_ptr step = slices.size() > 2 ? std::move(slices[2]) : nullptr;
      return make_node<emel::jinja::slice_expression>(
          start_pos, std::move(start), std::move(stop), std::move(step));
    }
    return slices.empty() ? nullptr : std::move(slices[0]);
  }

  emel::jinja::ast_ptr parse_primary_expression() {
    size_t start_pos = current_;
    const auto & t = peek();
    ++current_;
    switch (t.type) {
      case emel::jinja::token_type::NumericLiteral: {
        const char * str = t.value.c_str();
        if (t.value.find('.') != std::string::npos) {
          char * end = nullptr;
          double v = std::strtod(str, &end);
          if (end == str) {
            set_error(EMEL_ERR_PARSE_FAILED, t.pos);
            return nullptr;
          }
          return make_node<emel::jinja::float_literal>(start_pos, v);
        }
        char * end = nullptr;
        long long v = std::strtoll(str, &end, 10);
        if (end == str) {
          set_error(EMEL_ERR_PARSE_FAILED, t.pos);
          return nullptr;
        }
        return make_node<emel::jinja::integer_literal>(start_pos, static_cast<int64_t>(v));
      }
      case emel::jinja::token_type::StringLiteral: {
        std::string val = t.value;
        while (is(emel::jinja::token_type::StringLiteral)) {
          val += peek().value;
          ++current_;
        }
        return make_node<emel::jinja::string_literal>(start_pos, std::move(val));
      }
      case emel::jinja::token_type::Identifier:
        return make_node<emel::jinja::identifier>(start_pos, t.value);
      case emel::jinja::token_type::OpenParen: {
        auto expr = parse_expression_sequence();
        if (!expect(emel::jinja::token_type::CloseParen)) {
          return nullptr;
        }
        return expr;
      }
      case emel::jinja::token_type::OpenSquareBracket: {
        emel::jinja::ast_list vals;
        while (ok() && !is(emel::jinja::token_type::CloseSquareBracket)) {
          vals.push_back(parse_expression());
          if (is(emel::jinja::token_type::Comma)) {
            ++current_;
          }
        }
        if (!expect(emel::jinja::token_type::CloseSquareBracket)) {
          return nullptr;
        }
        return make_node<emel::jinja::array_literal>(start_pos, std::move(vals));
      }
      case emel::jinja::token_type::OpenCurlyBracket: {
        emel::jinja::ast_pair_list pairs;
        while (ok() && !is(emel::jinja::token_type::CloseCurlyBracket)) {
          auto key = parse_expression();
          if (!expect(emel::jinja::token_type::Colon)) {
            return nullptr;
          }
          pairs.push_back({std::move(key), parse_expression()});
          if (is(emel::jinja::token_type::Comma)) {
            ++current_;
          }
        }
        if (!expect(emel::jinja::token_type::CloseCurlyBracket)) {
          return nullptr;
        }
        return make_node<emel::jinja::object_literal>(start_pos, std::move(pairs));
      }
      default:
        set_error(EMEL_ERR_PARSE_FAILED, t.pos);
        return nullptr;
    }
  }

  const std::vector<emel::jinja::token> * tokens_ = nullptr;
  size_t current_ = 0;
  int32_t error_ = EMEL_OK;
  size_t error_pos_ = 0;
  emel::jinja::program & program_;
};

} // namespace emel::jinja::parser::detail
