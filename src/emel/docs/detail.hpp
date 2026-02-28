#pragma once

#include <boost/sml.hpp>

#include <cstddef>
#include <cctype>
#include <string>
#include <string_view>
#include <type_traits>

namespace emel::docs::detail {

inline std::string sanitize_mermaid(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  for (std::size_t i = 0; i < name.size(); ++i) {
    const char ch = name[i];
    if (ch == ':' && i + 1 < name.size() && name[i + 1] == ':') {
      out.push_back('_');
      out.push_back('_');
      ++i;
      continue;
    }
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) != 0 || ch == '_') {
      out.push_back(ch);
      continue;
    }
    out.push_back('_');
  }
  return out;
}

inline std::string shorten_type_name(std::string_view name) {
  std::string out(name);
  const std::size_t pos = out.rfind("::");
  if (pos != std::string::npos) {
    out = out.substr(pos + 2);
  }
  const std::string marker = "lambda at ";
  const std::size_t lambda_pos = out.find(marker);
  if (lambda_pos != std::string::npos) {
    std::string_view rest(out);
    rest.remove_prefix(lambda_pos + marker.size());
    const std::size_t end = rest.find('>');
    if (end != std::string::npos) {
      rest = rest.substr(0, end);
    }
    const std::size_t slash = rest.find_last_of("/\\");
    if (slash != std::string::npos) {
      rest = rest.substr(slash + 1);
    }
    std::string file;
    std::string line;
    std::string col;
    const std::size_t colon1 = rest.find(':');
    if (colon1 != std::string::npos) {
      file.assign(rest.substr(0, colon1));
      const std::size_t colon2 = rest.find(':', colon1 + 1);
      if (colon2 != std::string::npos) {
        line.assign(rest.substr(colon1 + 1, colon2 - colon1 - 1));
        col.assign(rest.substr(colon2 + 1));
      } else {
        line.assign(rest.substr(colon1 + 1));
      }
    } else {
      file.assign(rest);
    }
    auto trim_trailing_non_alnum = [](std::string & value) {
      while (!value.empty()) {
        const unsigned char ch = static_cast<unsigned char>(value.back());
        if (std::isalnum(ch) != 0) {
          break;
        }
        value.pop_back();
      }
    };
    trim_trailing_non_alnum(file);
    trim_trailing_non_alnum(line);
    trim_trailing_non_alnum(col);
    const std::size_t dot = file.rfind('.');
    if (dot != std::string::npos) {
      file = file.substr(0, dot);
    }
    std::string shortened = "lambda";
    if (!file.empty()) {
      shortened += "_";
      shortened += file;
    }
    if (!line.empty()) {
      shortened += "_";
      shortened += line;
    }
    if (!col.empty()) {
      shortened += "_";
      shortened += col;
    }
    return shortened;
  }
  return out;
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
