#pragma once

#include <boost/sml.hpp>

#include <cstddef>
#include <cctype>
#include <array>
#include <string>
#include <string_view>
#include <type_traits>

namespace emel::docs::detail {

inline std::string sanitize_mermaid(std::string_view name) {
  using mode_handler_t = void (*)(std::string &, char, std::size_t &) noexcept;
  static constexpr std::array<mode_handler_t, 3> MODE_HANDLERS = {
      +[](std::string & value, char, std::size_t &) noexcept {
        value.push_back('_');
      },
      +[](std::string & value, char ch, std::size_t &) noexcept {
        value.push_back(ch);
      },
      +[](std::string & value, char, std::size_t & i) noexcept {
        value.push_back('_');
        value.push_back('_');
        i += 1;
      },
  };

  std::string out;
  out.reserve(name.size());
  for (std::size_t i = 0; i < name.size(); ++i) {
    const char ch = name[i];
    const size_t has_next = static_cast<size_t>(i + 1 < name.size());
    const size_t is_scope =
        static_cast<size_t>(ch == ':' && has_next != 0 && name[i + 1] == ':');
    const unsigned char uch = static_cast<unsigned char>(ch);
    const size_t is_ident = static_cast<size_t>(std::isalnum(uch) != 0 || ch == '_');
    const std::array<size_t, 2> mode_candidates = {is_ident, 2u};
    const size_t mode = mode_candidates[static_cast<size_t>(is_scope != 0)];
    MODE_HANDLERS[mode](out, ch, i);
  }
  return out;
}

inline void append_non_empty_none(std::string &, const std::string &) {}

inline void append_non_empty_some(std::string & out_value, const std::string & suffix) {
  out_value += "_" + suffix;
}

inline void append_non_empty(std::string & out_value, const std::string & suffix) {
  using append_handler_t = void (*)(std::string &, const std::string &);
  static constexpr std::array<append_handler_t, 2> APPEND_HANDLERS = {
      append_non_empty_none,
      append_non_empty_some,
  };
  APPEND_HANDLERS[static_cast<size_t>(!suffix.empty())](out_value, suffix);
}

inline std::string shorten_type_name_no_lambda(std::string out,
                                               std::size_t,
                                               const std::string &) {
  return out;
}

inline std::string shorten_type_name_with_lambda(std::string out,
                                                 const std::size_t lambda_pos,
                                                 const std::string & marker) {
  std::string_view rest(out);
  rest.remove_prefix(lambda_pos + marker.size());
  const std::size_t end = rest.find('>');
  const size_t has_end = static_cast<size_t>(end != std::string::npos);
  std::string_view end_candidates[2] = {rest, rest.substr(0, end * has_end)};
  rest = end_candidates[has_end];

  const std::size_t slash = rest.find_last_of("/\\");
  const size_t has_slash = static_cast<size_t>(slash != std::string::npos);
  std::string_view slash_candidates[2] = {rest, rest.substr((slash + 1) * has_slash)};
  rest = slash_candidates[has_slash];

  std::string file;
  std::string line;
  std::string col;
  const std::size_t colon1 = rest.find(':');
  const size_t has_colon1 = static_cast<size_t>(colon1 != std::string::npos);
  const std::size_t colon2 = rest.find(':', colon1 + has_colon1);
  const size_t has_colon2 = has_colon1 & static_cast<size_t>(colon2 != std::string::npos);
  const size_t colon_mode = has_colon1 + has_colon2;

  using colon_handler_t = void (*)(std::string_view,
                                   std::size_t,
                                   std::size_t,
                                   std::string &,
                                   std::string &,
                                   std::string &) noexcept;
  static constexpr std::array<colon_handler_t, 3> COLON_HANDLERS = {
      +[](std::string_view value,
          std::size_t,
          std::size_t,
          std::string & file_out,
          std::string &,
          std::string &) noexcept {
        file_out.assign(value);
      },
      +[](std::string_view value,
          std::size_t colon1_value,
          std::size_t,
          std::string & file_out,
          std::string & line_out,
          std::string &) noexcept {
        file_out.assign(value.substr(0, colon1_value));
        line_out.assign(value.substr(colon1_value + 1));
      },
      +[](std::string_view value,
          std::size_t colon1_value,
          std::size_t colon2_value,
          std::string & file_out,
          std::string & line_out,
          std::string & col_out) noexcept {
        file_out.assign(value.substr(0, colon1_value));
        line_out.assign(value.substr(colon1_value + 1, colon2_value - colon1_value - 1));
        col_out.assign(value.substr(colon2_value + 1));
      },
  };

  COLON_HANDLERS[colon_mode](rest, colon1, colon2, file, line, col);

  auto trim_trailing_non_alnum = [](std::string & value) {
    while (!value.empty() &&
           std::isalnum(static_cast<unsigned char>(value.back())) == 0) {
      value.pop_back();
    }
  };
  trim_trailing_non_alnum(file);
  trim_trailing_non_alnum(line);
  trim_trailing_non_alnum(col);

  const std::size_t dot = file.rfind('.');
  const size_t has_dot = static_cast<size_t>(dot != std::string::npos);
  std::string dot_candidates[2] = {file, file.substr(0, dot * has_dot)};
  file = std::move(dot_candidates[has_dot]);

  std::string shortened = "lambda";
  append_non_empty(shortened, file);
  append_non_empty(shortened, line);
  append_non_empty(shortened, col);
  return shortened;
}

inline std::string shorten_type_name(std::string_view name) {
  std::string out(name);
  const std::size_t pos = out.rfind("::");
  const size_t has_namespace = static_cast<size_t>(pos != std::string::npos);
  std::string namespace_candidates[2] = {out, out.substr((pos + 2) * has_namespace)};
  out = std::move(namespace_candidates[has_namespace]);

  const std::string marker = "lambda at ";
  const std::size_t lambda_pos = out.find(marker);
  const size_t has_lambda = static_cast<size_t>(lambda_pos != std::string::npos);
  using lambda_handler_t = std::string (*)(std::string, std::size_t, const std::string &);
  static constexpr std::array<lambda_handler_t, 2> LAMBDA_HANDLERS = {
      shorten_type_name_no_lambda,
      shorten_type_name_with_lambda,
  };
  return LAMBDA_HANDLERS[has_lambda](std::move(out), lambda_pos, marker);
}

inline std::string mermaid_label(std::string_view name) {
  return sanitize_mermaid(shorten_type_name(name));
}

template <class T>
std::string raw_type_name() {
  return boost::sml::aux::string<T>::c_str();
}

template <class>
struct is_unexpected_event : std::false_type {};

template <class T, class event>
struct is_unexpected_event<boost::sml::back::unexpected_event<T, event>>
    : std::true_type {};

template <class>
struct is_completion_event : std::false_type {};

template <class event>
struct is_completion_event<boost::sml::back::completion<event>> : std::true_type {
  using payload_type = event;
};

template <class event>
std::string event_type_name() {
  if constexpr (is_unexpected_event<event>::value) {
    using mapped = boost::sml::back::get_event_t<event>;
    return raw_type_name<mapped>();
  }
  if constexpr (is_completion_event<event>::value) {
    using payload = typename is_completion_event<event>::payload_type;
    std::string completion_name = "completion<";
    completion_name += shorten_type_name(raw_type_name<payload>());
    completion_name += ">";
    return completion_name;
  }
  return raw_type_name<event>();
}

template <class event>
std::string mermaid_event_name() {
  if constexpr (std::is_same_v<event, boost::sml::back::anonymous>) {
    return {};
  }
  return mermaid_label(event_type_name<event>());
}

template <class event>
std::string table_event_name() {
  if constexpr (std::is_same_v<event, boost::sml::back::anonymous>) {
    return "-";
  }
  return shorten_type_name(event_type_name<event>());
}

}  // namespace emel::docs::detail
