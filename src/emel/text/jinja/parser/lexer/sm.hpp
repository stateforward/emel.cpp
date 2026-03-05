#pragma once

#include "emel/sm.hpp"
#include "emel/text/jinja/parser/lexer/actions.hpp"
#include "emel/text/jinja/parser/lexer/context.hpp"
#include "emel/text/jinja/parser/lexer/detail.hpp"
#include "emel/text/jinja/parser/lexer/guards.hpp"

namespace emel::text::jinja::parser::lexer {

struct initialized {};
struct scanning {};

struct text_boundary_candidate_decision {};
struct text_scan_exec {};
struct text_scan_result_decision {};
struct text_opening_block_decision {};
struct text_trim_opening_block_exec {};
struct text_trim_opening_block_result_decision {};
struct text_materialize_exec {};
struct text_finalize_exec {};
struct text_finalize_result_decision {};
struct text_finalize_token_exec {};
struct text_emit_result_decision {};
struct comment_candidate_decision {};
struct comment_scan_exec {};
struct comment_scan_result_decision {};
struct comment_finalize_exec {};
struct comment_finalize_result_decision {};
struct comment_unterminated_exec {};
struct comment_unterminated_result_decision {};
struct trim_prefix_candidate_decision {};
struct trim_prefix_scan_exec {};
struct trim_prefix_result_decision {};
struct trim_prefix_eof_exec {};
struct trim_prefix_eof_result_decision {};
struct space_scan_exec {};
struct space_scan_result_decision {};
struct space_eof_exec {};
struct space_eof_result_decision {};
struct unary_candidate_decision {};
struct unary_prefix_context_decision {};
struct unary_prefix_allowed_decision {};
struct unary_scan_exec {};
struct unary_scan_result_decision {};
struct mapping_candidate_decision {};
struct mapping_close_curly_exec {};
struct mapping_scan_exec {};
struct mapping_scan_result_decision {};
struct string_scan_decision {};
struct string_scan_exec {};
struct string_content_scan_exec {};
struct string_content_policy_decision {};
struct string_scan_result_decision {};
struct string_materialize_exec {};
struct string_status_decision {};
struct string_finalize_exec {};
struct string_finalize_result_decision {};
struct string_unterminated_exec {};
struct string_unterminated_result_decision {};
struct numeric_scan_exec {};
struct numeric_scan_result_decision {};
struct word_scan_exec {};
struct word_scan_result_decision {};
struct invalid_char_exec {};
struct invalid_char_result_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Intake.
        sml::state<initialized> <= *sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::invalid_next{} ]
          / action::reject_invalid_next

      , sml::state<initialized> <= sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::invalid_cursor_position{} ]
          / action::reject_invalid_cursor

      , sml::state<text_boundary_candidate_decision> <= sml::state<initialized>
          + sml::event<event::next_runtime>
          [ guard::valid_cursor_position{} ]
          / action::begin_scan

      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::invalid_next{} ]
          / action::reject_invalid_next

      , sml::state<scanning> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::invalid_cursor_position{} ]
          / action::reject_invalid_cursor

      , sml::state<text_boundary_candidate_decision> <= sml::state<scanning>
          + sml::event<event::next_runtime>
          [ guard::valid_cursor_position{} ]
          / action::begin_scan

      //------------------------------------------------------------------------------//
      // Text-boundary start decision.
      , sml::state<text_scan_exec> <= sml::state<text_boundary_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::at_text_boundary{} ]

      , sml::state<comment_candidate_decision> <= sml::state<text_boundary_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::not_at_text_boundary{} ]

      //------------------------------------------------------------------------------//
      // Text boundary phase.
      , sml::state<text_scan_result_decision> <= sml::state<text_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_text_boundary

      , sml::state<scanning> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<invalid_char_exec> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<text_opening_block_decision> <= sml::state<text_scan_result_decision>
          + sml::completion<event::next_runtime>

      , sml::state<text_trim_opening_block_exec> <= sml::state<text_opening_block_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_opening_block_ahead{} ]

      , sml::state<text_materialize_exec> <= sml::state<text_opening_block_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_opening_block_not_ahead{} ]

      , sml::state<invalid_char_exec> <= sml::state<text_opening_block_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<text_trim_opening_block_result_decision> <= sml::state<text_trim_opening_block_exec>
          + sml::completion<event::next_runtime>
          / action::probe_text_opening_trim

      , sml::state<text_materialize_exec> <= sml::state<text_trim_opening_block_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_opening_trim_stopped_on_newline{} ]
          / action::apply_text_opening_trim_to_newline

      , sml::state<text_materialize_exec> <= sml::state<text_trim_opening_block_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_opening_trim_to_zero{} ]
          / action::apply_text_opening_trim_to_zero

      , sml::state<text_materialize_exec> <= sml::state<text_trim_opening_block_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_opening_trim_keep_original{} ]

      , sml::state<invalid_char_exec> <= sml::state<text_trim_opening_block_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<text_finalize_exec> <= sml::state<text_materialize_exec>
          + sml::completion<event::next_runtime>
          / action::materialize_text_token

      , sml::state<text_finalize_result_decision> <= sml::state<text_finalize_exec>
          + sml::completion<event::next_runtime>
          [ guard::text_can_trim_leading_newline{} ]
          / action::trim_text_leading_newline

      , sml::state<text_finalize_result_decision> <= sml::state<text_finalize_exec>
          + sml::completion<event::next_runtime>
          [ guard::text_skip_trim_leading_newline{} ]

      , sml::state<invalid_char_exec> <= sml::state<text_finalize_exec>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<text_finalize_token_exec> <= sml::state<text_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_apply_lstrip_and_rstrip{} ]
          / action::lstrip_and_rstrip_text_token

      , sml::state<text_finalize_token_exec> <= sml::state<text_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_apply_lstrip_only{} ]
          / action::lstrip_text_token

      , sml::state<text_finalize_token_exec> <= sml::state<text_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_apply_rstrip_only{} ]
          / action::rstrip_text_token

      , sml::state<text_finalize_token_exec> <= sml::state<text_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_apply_no_strip{} ]

      , sml::state<invalid_char_exec> <= sml::state<text_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<text_emit_result_decision> <= sml::state<text_finalize_token_exec>
          + sml::completion<event::next_runtime>
          / action::finalize_text_boundary_token

      , sml::state<scanning> <= sml::state<text_emit_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_token_non_empty{} ]
          / action::emit_scanned_token

      , sml::state<comment_candidate_decision> <= sml::state<text_emit_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::text_token_empty{} ]

      , sml::state<invalid_char_exec> <= sml::state<text_emit_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      //------------------------------------------------------------------------------//
      // Comment-start decision.
      , sml::state<comment_scan_exec> <= sml::state<comment_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::starts_comment{} ]

      , sml::state<trim_prefix_candidate_decision> <= sml::state<comment_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::not_starts_comment{} ]

      //------------------------------------------------------------------------------//
      // Comment phase.
      , sml::state<comment_scan_result_decision> <= sml::state<comment_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_comment

      , sml::state<scanning> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<comment_finalize_exec> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::comment_terminated{} ]

      , sml::state<comment_unterminated_exec> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::comment_unterminated{} ]

      , sml::state<invalid_char_exec> <= sml::state<comment_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<comment_finalize_result_decision> <= sml::state<comment_finalize_exec>
          + sml::completion<event::next_runtime>
          / action::finalize_comment_token

      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<comment_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<comment_unterminated_result_decision> <= sml::state<comment_unterminated_exec>
          + sml::completion<event::next_runtime>
          / action::mark_comment_unterminated

      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<comment_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      //------------------------------------------------------------------------------//
      // Trim-prefix start decision.
      , sml::state<trim_prefix_scan_exec> <= sml::state<trim_prefix_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::starts_trim_prefix{} ]

      , sml::state<space_scan_exec> <= sml::state<trim_prefix_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::not_starts_trim_prefix{} ]

      //------------------------------------------------------------------------------//
      // Trim-prefix phase.
      , sml::state<trim_prefix_result_decision> <= sml::state<trim_prefix_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_trim_prefix

      , sml::state<trim_prefix_eof_exec> <= sml::state<trim_prefix_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::cursor_at_end{} ]

      , sml::state<space_scan_exec> <= sml::state<trim_prefix_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::cursor_not_at_end{} ]

      , sml::state<trim_prefix_eof_result_decision> <= sml::state<trim_prefix_eof_exec>
          + sml::completion<event::next_runtime>
          / action::mark_no_token_eof

      , sml::state<scanning> <= sml::state<trim_prefix_eof_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<trim_prefix_eof_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      //------------------------------------------------------------------------------//
      // Space-skip phase.
      , sml::state<space_scan_result_decision> <= sml::state<space_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_spaces

      , sml::state<space_eof_exec> <= sml::state<space_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::cursor_at_end{} ]

      , sml::state<unary_candidate_decision> <= sml::state<space_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::cursor_not_at_end{} ]

      , sml::state<space_eof_result_decision> <= sml::state<space_eof_exec>
          + sml::completion<event::next_runtime>
          / action::mark_no_token_eof

      , sml::state<scanning> <= sml::state<space_eof_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<space_eof_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<mapping_candidate_decision> <= sml::state<unary_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_not_candidate{} ]

      , sml::state<unary_prefix_context_decision> <= sml::state<unary_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_candidate{} ]

      , sml::state<invalid_char_exec> <= sml::state<unary_prefix_context_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_prefix_context_invalid{} ]

      , sml::state<unary_prefix_allowed_decision> <= sml::state<unary_prefix_context_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_prefix_context_valid{} ]

      , sml::state<mapping_candidate_decision> <= sml::state<unary_prefix_allowed_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_prefix_disallowed{} ]

      , sml::state<unary_scan_exec> <= sml::state<unary_prefix_allowed_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_prefix_allowed{} ]

      //------------------------------------------------------------------------------//
      // Unary phase.
      , sml::state<unary_scan_result_decision> <= sml::state<unary_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_unary

      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_numeric_suffix_present{} ]
          / action::emit_unary_numeric_token

      , sml::state<scanning> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::unary_numeric_suffix_absent{} ]
          / action::emit_unary_operator_token

      , sml::state<invalid_char_exec> <= sml::state<unary_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      //------------------------------------------------------------------------------//
      // Mapping-start decision.
      , sml::state<mapping_close_curly_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_expression_blocked_by_curly_depth{} ]

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_statement_trim{} ]
          / action::scan_mapping_open_statement_trim

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_statement_trim{} ]
          / action::scan_mapping_close_statement_trim

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_expression_trim{} ]
          / action::scan_mapping_open_expression_trim

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_expression_trim{} ]
          / action::scan_mapping_close_expression_trim

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_statement{} ]
          / action::scan_mapping_open_statement

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_statement{} ]
          / action::scan_mapping_close_statement

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_expression{} ]
          / action::scan_mapping_open_expression

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_expression_not_blocked{} ]
          / action::scan_mapping_close_expression

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_paren{} ]
          / action::scan_mapping_open_paren

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_paren{} ]
          / action::scan_mapping_close_paren

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_curly_bracket{} ]
          / action::scan_mapping_open_curly_bracket

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_curly_bracket{} ]
          / action::scan_mapping_close_curly_bracket

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_open_square_bracket{} ]
          / action::scan_mapping_open_square_bracket

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_close_square_bracket{} ]
          / action::scan_mapping_close_square_bracket

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_comma{} ]
          / action::scan_mapping_comma

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_dot{} ]
          / action::scan_mapping_dot

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_colon{} ]
          / action::scan_mapping_colon

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_pipe{} ]
          / action::scan_mapping_pipe

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_less_equal{} ]
          / action::scan_mapping_less_equal

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_greater_equal{} ]
          / action::scan_mapping_greater_equal

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_equal_equal{} ]
          / action::scan_mapping_equal_equal

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_bang_equal{} ]
          / action::scan_mapping_bang_equal

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_less{} ]
          / action::scan_mapping_less

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_greater{} ]
          / action::scan_mapping_greater

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_plus{} ]
          / action::scan_mapping_plus

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_minus{} ]
          / action::scan_mapping_minus

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_tilde{} ]
          / action::scan_mapping_tilde

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_star{} ]
          / action::scan_mapping_star

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_slash{} ]
          / action::scan_mapping_slash

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_percent{} ]
          / action::scan_mapping_percent

      , sml::state<mapping_scan_exec> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>
          [ guard::mapping_equals{} ]
          / action::scan_mapping_equals

      , sml::state<string_scan_decision> <= sml::state<mapping_candidate_decision>
          + sml::completion<event::next_runtime>

      //------------------------------------------------------------------------------//
      // Mapping phase.
      , sml::state<mapping_scan_result_decision> <= sml::state<mapping_close_curly_exec>
          + sml::completion<event::next_runtime>
          / action::scan_mapping_close_curly

      , sml::state<mapping_scan_result_decision> <= sml::state<mapping_scan_exec>
          + sml::completion<event::next_runtime>

      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<mapping_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      //------------------------------------------------------------------------------//
      // String / number / identifier decisions.
      , sml::state<string_scan_exec> <= sml::state<string_scan_decision>
          + sml::completion<event::next_runtime>
          [ guard::starts_string{} ]

      , sml::state<numeric_scan_exec> <= sml::state<string_scan_decision>
          + sml::completion<event::next_runtime>
          [ guard::starts_numeric{} ]

      , sml::state<word_scan_exec> <= sml::state<string_scan_decision>
          + sml::completion<event::next_runtime>
          [ guard::starts_word{} ]

      , sml::state<invalid_char_exec> <= sml::state<string_scan_decision>
          + sml::completion<event::next_runtime>

      , sml::state<string_content_scan_exec> <= sml::state<string_scan_exec>
          + sml::completion<event::next_runtime>
          / action::begin_string_scan

      , sml::state<string_content_policy_decision> <= sml::state<string_content_scan_exec>
          + sml::completion<event::next_runtime>

      , sml::state<string_scan_result_decision> <= sml::state<string_content_policy_decision>
          + sml::completion<event::next_runtime>
          [ guard::string_scan_immediate_termination_or_eof{} ]

      , sml::state<string_scan_result_decision> <= sml::state<string_content_policy_decision>
          + sml::completion<event::next_runtime>
          [ guard::string_scan_requires_content{} ]
          / action::scan_string_content

      , sml::state<invalid_char_exec> <= sml::state<string_content_policy_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<string_materialize_exec> <= sml::state<string_scan_result_decision>
          + sml::completion<event::next_runtime>

      , sml::state<string_status_decision> <= sml::state<string_materialize_exec>
          + sml::completion<event::next_runtime>
          / action::materialize_string_token

      , sml::state<scanning> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<string_unterminated_exec> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::string_not_terminated{} ]

      , sml::state<string_finalize_exec> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::string_terminated{} ]

      , sml::state<invalid_char_exec> <= sml::state<string_status_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<string_unterminated_result_decision> <= sml::state<string_unterminated_exec>
          + sml::completion<event::next_runtime>
          / action::mark_string_unterminated

      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<string_unterminated_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<string_finalize_result_decision> <= sml::state<string_finalize_exec>
          + sml::completion<event::next_runtime>
          / action::finalize_string_token

      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<string_finalize_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<numeric_scan_result_decision> <= sml::state<numeric_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_numeric

      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<numeric_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<word_scan_result_decision> <= sml::state<word_scan_exec>
          + sml::completion<event::next_runtime>
          / action::scan_word

      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_token_available{} ]
          / action::emit_scanned_token

      , sml::state<scanning> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_no_token_eof{} ]
          / action::emit_eof

      , sml::state<invalid_char_exec> <= sml::state<word_scan_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::scan_unhandled{} ]

      , sml::state<invalid_char_result_decision> <= sml::state<invalid_char_exec>
          + sml::completion<event::next_runtime>
          / action::mark_invalid_character

      , sml::state<scanning> <= sml::state<invalid_char_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_invalid_request{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<invalid_char_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_parse_failed{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<invalid_char_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_internal_error{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<invalid_char_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_untracked{} ]
          / action::emit_scan_error
      , sml::state<scanning> <= sml::state<invalid_char_result_decision>
          + sml::completion<event::next_runtime>
          [ guard::parse_error_unknown{} ]
          / action::emit_scan_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<scanning> <= sml::state<initialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected

      , sml::state<scanning> <= sml::state<scanning> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() : base_type() {}

  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const ::emel::text::jinja::lexer::event::next &ev) {
    event::next_ctx runtime_ctx{};
    event::next_runtime runtime_ev{ev, runtime_ctx};
    return base_type::process_event(runtime_ev);
  }
};

using Lexer = sm;

} // namespace emel::text::jinja::parser::lexer
