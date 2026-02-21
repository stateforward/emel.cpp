#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "emel/jinja/lexer.hpp"

namespace emel::jinja {

struct ast_node {
  size_t pos = 0;
  virtual ~ast_node() = default;
};

using ast_ptr = std::unique_ptr<ast_node>;
using ast_list = std::vector<ast_ptr>;
using ast_pair_list = std::vector<std::pair<ast_ptr, ast_ptr>>;

struct comment_statement : ast_node {
  std::string value;
  explicit comment_statement(std::string text) : value(std::move(text)) {}
};

struct string_literal : ast_node {
  std::string value;
  explicit string_literal(std::string text) : value(std::move(text)) {}
};

struct identifier : ast_node {
  std::string name;
  explicit identifier(std::string text) : name(std::move(text)) {}
};

struct integer_literal : ast_node {
  int64_t value = 0;
  explicit integer_literal(int64_t v) : value(v) {}
};

struct float_literal : ast_node {
  double value = 0.0;
  explicit float_literal(double v) : value(v) {}
};

struct tuple_literal : ast_node {
  ast_list values;
  explicit tuple_literal(ast_list vals) : values(std::move(vals)) {}
};

struct array_literal : ast_node {
  ast_list values;
  explicit array_literal(ast_list vals) : values(std::move(vals)) {}
};

struct object_literal : ast_node {
  ast_pair_list pairs;
  explicit object_literal(ast_pair_list vals) : pairs(std::move(vals)) {}
};

struct unary_expression : ast_node {
  token op;
  ast_ptr operand;
  unary_expression(token op_token, ast_ptr expr)
      : op(std::move(op_token)), operand(std::move(expr)) {}
};

struct binary_expression : ast_node {
  token op;
  ast_ptr left;
  ast_ptr right;
  binary_expression(token op_token, ast_ptr lhs, ast_ptr rhs)
      : op(std::move(op_token)),
        left(std::move(lhs)),
        right(std::move(rhs)) {}
};

struct ternary_expression : ast_node {
  ast_ptr test;
  ast_ptr true_expr;
  ast_ptr false_expr;
  ternary_expression(ast_ptr test_expr, ast_ptr true_value, ast_ptr false_value)
      : test(std::move(test_expr)),
        true_expr(std::move(true_value)),
        false_expr(std::move(false_value)) {}
};

struct select_expression : ast_node {
  ast_ptr value;
  ast_ptr test;
  select_expression(ast_ptr value_expr, ast_ptr test_expr)
      : value(std::move(value_expr)),
        test(std::move(test_expr)) {}
};

struct test_expression : ast_node {
  ast_ptr operand;
  bool negate = false;
  ast_ptr test;
  test_expression(ast_ptr operand_expr, bool neg, ast_ptr test_expr)
      : operand(std::move(operand_expr)),
        negate(neg),
        test(std::move(test_expr)) {}
};

struct filter_expression : ast_node {
  ast_ptr operand;
  ast_ptr filter;
  filter_expression(ast_ptr operand_expr, ast_ptr filter_expr)
      : operand(std::move(operand_expr)),
        filter(std::move(filter_expr)) {}
};

struct call_expression : ast_node {
  ast_ptr callee;
  ast_list args;
  call_expression(ast_ptr callee_expr, ast_list arguments)
      : callee(std::move(callee_expr)),
        args(std::move(arguments)) {}
};

struct call_statement : ast_node {
  ast_ptr call_expr;
  ast_list caller_args;
  ast_list body;
  call_statement(ast_ptr call_expr_in, ast_list caller_args_in, ast_list body_in)
      : call_expr(std::move(call_expr_in)),
        caller_args(std::move(caller_args_in)),
        body(std::move(body_in)) {}
};

struct filter_statement : ast_node {
  ast_ptr filter_node;
  ast_list body;
  filter_statement(ast_ptr filter_expr, ast_list body_in)
      : filter_node(std::move(filter_expr)),
        body(std::move(body_in)) {}
};

struct set_statement : ast_node {
  ast_ptr left;
  ast_ptr value;
  ast_list body;
  set_statement(ast_ptr left_expr, ast_ptr value_expr, ast_list body_in)
      : left(std::move(left_expr)),
        value(std::move(value_expr)),
        body(std::move(body_in)) {}
};

struct if_statement : ast_node {
  ast_ptr test;
  ast_list body;
  ast_list alternate;
  if_statement(ast_ptr test_expr, ast_list body_in, ast_list alternate_in)
      : test(std::move(test_expr)),
        body(std::move(body_in)),
        alternate(std::move(alternate_in)) {}
};

struct macro_statement : ast_node {
  ast_ptr name;
  ast_list args;
  ast_list body;
  macro_statement(ast_ptr name_expr, ast_list args_in, ast_list body_in)
      : name(std::move(name_expr)),
        args(std::move(args_in)),
        body(std::move(body_in)) {}
};

struct for_statement : ast_node {
  ast_ptr loop_var;
  ast_ptr iterable;
  ast_list body;
  ast_list alternate;
  for_statement(ast_ptr loop_var_in, ast_ptr iterable_in, ast_list body_in, ast_list alternate_in)
      : loop_var(std::move(loop_var_in)),
        iterable(std::move(iterable_in)),
        body(std::move(body_in)),
        alternate(std::move(alternate_in)) {}
};

struct break_statement : ast_node {
  break_statement() = default;
};

struct continue_statement : ast_node {
  continue_statement() = default;
};

struct noop_statement : ast_node {
  noop_statement() = default;
};

struct member_expression : ast_node {
  ast_ptr object;
  ast_ptr property;
  bool computed = false;
  member_expression(ast_ptr object_in, ast_ptr property_in, bool is_computed)
      : object(std::move(object_in)),
        property(std::move(property_in)),
        computed(is_computed) {}
};

struct slice_expression : ast_node {
  ast_ptr start;
  ast_ptr stop;
  ast_ptr step;
  slice_expression(ast_ptr start_expr, ast_ptr stop_expr, ast_ptr step_expr)
      : start(std::move(start_expr)),
        stop(std::move(stop_expr)),
        step(std::move(step_expr)) {}
};

struct keyword_argument_expression : ast_node {
  ast_ptr key;
  ast_ptr value;
  keyword_argument_expression(ast_ptr key_expr, ast_ptr value_expr)
      : key(std::move(key_expr)),
        value(std::move(value_expr)) {}
};

struct spread_expression : ast_node {
  ast_ptr operand;
  explicit spread_expression(ast_ptr expr) : operand(std::move(expr)) {}
};

struct program {
  ast_list body;
  int32_t last_error = EMEL_OK;
  size_t last_error_pos = 0;

  void reset() noexcept {
    body.clear();
    last_error = EMEL_OK;
    last_error_pos = 0;
  }
};

} // namespace emel::jinja
