#ifndef _STL_EXT_IOSTREAM_HPP_
#define _STL_EXT_IOSTREAM_HPP_

#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdint>

#include "complex.hpp"
#include "type_traits.hpp"

namespace stl_ext
{

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

namespace detail
{

template <size_t I, typename... Args>
struct print_tuple_helper : print_tuple_helper<I-1, Args...>
{
    print_tuple_helper(std::ostream& os, const std::tuple<Args...>& t)
    : print_tuple_helper<I-1, Args...>(os, t)
    {
        os << std::get<I-1>(t);
        if (I < sizeof...(Args)) os << ", ";
    }
};

template <typename... Args>
struct print_tuple_helper<1, Args...>
{
    print_tuple_helper(std::ostream& os, const std::tuple<Args...>& t)
    {
        os << std::get<0>(t);
        if (1 < sizeof...(Args)) os << ", ";
    }
};

template <typename... Args>
struct print_tuple_helper<0, Args...>
{
    print_tuple_helper(std::ostream&, const std::tuple<Args...>&) {}
};

class ios_flag_saver
{
    public:
        ios_flag_saver(std::ostream& os)
        : os_(os), flags_(os.flags()) {}

        ~ios_flag_saver()
        {
            os_.flags(flags_);
        }

    protected:
        std::ostream& os_;
        std::ios::fmtflags flags_;
};

template <typename... Args>
struct printos_helper
{
    std::string fmt;
    std::tuple<Args&&...> args;

    printos_helper(const std::string& fmt_, Args&&... args_)
    : fmt(fmt_), args(std::forward<Args>(args_)...) {}

    printos_helper(const std::string& fmt_, std::tuple<Args&&...>&& args_)
    : fmt(fmt_), args(std::move(args_)) {}
};

template <typename T>
constexpr enable_if_arithmetic_t<T,bool> is_negative(const T& val) { return val < 0; }

template <typename T>
constexpr enable_if_not_arithmetic_t<T,bool> is_negative(const T&) { return false; }

template <typename T>
constexpr enable_if_arithmetic_t<T,bool> is_nonzero(const T& val) { return val != 0; }

template <typename T>
constexpr enable_if_not_arithmetic_t<T,bool> is_nonzero(const T&) { return false; }

template <typename T>
enable_if_t<is_integral<T>::value || is_pointer <T>::value>
print_integer(std::ostream& os, const T& val, bool is_signed)
{
    if (is_signed)
        os << static_cast<int64_t>(val);
    else
        os << static_cast<uint64_t>(val);
}

template <typename T>
enable_if_t<!is_integral<T>::value && !is_pointer <T>::value>
print_integer(std::ostream&, const T&, bool) {}

template <typename T>
enable_if_arithmetic_t<T>
print_float(std::ostream& os, const T& val)
{
    os << static_cast<double>(val);
}

template <typename T>
enable_if_not_arithmetic_t<T>
print_float(std::ostream&, const T&) {}

template <typename T>
enable_if_integral_t<T>
print_char(std::ostream& os, const T& val)
{
    os << static_cast<char>(val);
}

template <typename T>
enable_if_not_integral_t<T>
print_char(std::ostream&, const T&) {}

inline size_t string_size(const char* s)
{
    return strlen(s);
}

inline size_t string_size(const std::string& s)
{
    return s.size();
}

template <typename T>
enable_if_t<is_same<decay_t<T>,char*>::value ||
            is_same<decay_t<T>,const char*>::value ||
            is_same<decay_t<T>,std::string>::value>
print_string(std::ostream& os, const T& val, int width, int prec, bool left)
{
    int sz = string_size(val);
    if (prec != -1) sz = std::min(sz, prec);

    if (!left)
    {
        for (int i = (width < sz ? 0 : width-sz);i --> 0;)
        {
            os << ' ';
        }
    }

    for (int i = 0;i < sz;i++) os << val[i];

    if (left)
    {
        for (int i = (width < sz ? 0 : width-sz);i --> 0;)
        {
            os << ' ';
        }
    }
}

template <typename T>
enable_if_t<!is_same<decay_t<T>,char*>::value &&
            !is_same<decay_t<T>,const char*>::value &&
            !is_same<decay_t<T>,std::string>::value>
print_string(std::ostream&, const T&, int, int, bool) {}

class printf_conversion
{
    protected:
        bool _left;
        bool _zero;
        bool _sign;
        bool _alt;
        bool _space;
        int _width;
        int _prec;
        char _conv;

    public:
        printf_conversion(std::ostream& os, const std::string& fmt, std::string::const_iterator& i)
        : _left(false), _zero(false), _sign(false), _alt(false), _space(false),
          _width(0), _prec(-1), _conv(0)
        {
            std::string::const_iterator end = fmt.end();

            /*
             * Look for a conversion which is not trivial (%%).
             */
            while (true)
            {
                /*
                 * Copy non-converting chars to stream.
                 */
                while (i != end && *i != '%') os << *i++;

                /*
                 * No next conversion.
                 */
                if (i == end) return;

                /*
                 * Early termination is an error.
                 */
                if (++i == end) throw std::logic_error("Invalid conversion");

                /*
                 * Handle trivial %% conversions.
                 */
                if (*i == '%')
                {
                    os << '%';
                    ++i;
                    continue;
                }

                break;
            }


            /*
             * Check for a flag.
             */
            while (*i == ' ' || *i == '+' || *i == '-' || *i == '0' || *i == '#')
            {
                switch (*i)
                {
                    case ' ': _space = true; break;
                    case '+':  _sign = true; break;
                    case '-':  _left = true; break;
                    case '0':  _zero = true; break;
                    case '#':   _alt = true; break;
                }

                if (++i == end) throw std::logic_error("Invalid conversion");
            }

            /*
             * Check for a _width field.
             */
            if (*i >= '1' && *i <= '9')
            {
                do
                {
                    _width = _width*10 + (*i-'0');
                    if (++i == end) throw std::logic_error("Invalid conversion");
                }
                while (*i >= '0' && *i <= '9');
            }

            /*
             * Check for a precision field.
             */
            if (*i == '.')
            {
                ++i;
                _prec = 0;
                if (*i == '-')
                {
                    _prec = -1;
                    if (++i == end) throw std::logic_error("Invalid conversion");
                }
                while (*i >= '0' && *i <= '9')
                {
                    _prec = _prec*10 + (*i-'0');
                    if (++i == end) throw std::logic_error("Invalid conversion");
                }
                if (_prec < 0) _prec = 0;
            }

            /*
             * Finally, check the conversion.
             */
            if (*i == 'd' || *i == 'i' || *i == 'o' || *i == 'u' ||
                *i == 'x' || *i == 'X' || *i == 'e' || *i == 'E' ||
                *i == 'f' || *i == 'F' || *i == 'g' || *i == 'G' ||
                *i == 'a' || *i == 'A' || *i == 'c' || *i == 's' ||
                *i == 'p' || *i == 'j')
            {
                _conv = *i++;
            }
            else
            {
                throw std::logic_error("Invalid conversion.");
            }
        }

        template <typename T>
        void print(std::ostream& os, const T& arg)
        {
            ios_flag_saver flags(os);

            os << std::setfill(' ') << std::noshowpos << std::dec <<
                  std::right << std::nouppercase << std::noshowbase <<
                  std::noshowpoint;

            switch (_conv)
            {
                case 'd':
                case 'i':
                case 'o':
                case 'u':
                case 'x':
                case 'X':
                case 'p':
                {
                    if (_conv != 'p' && !is_integral<T>::value)
                        throw std::logic_error("Argument is not integral");

                    if (_conv == 'p' && !is_pointer<T>::value)
                        throw std::logic_error("Argument is not a pointer");

                    os << std::setw(_width) <<
                          std::setprecision(_prec == -1 ? 0 : _prec);

                    //TODO: _space

                    if (_conv == 'o')
                        os << std::oct;

                    if (_conv == 'x' || _conv == 'X' || _conv == 'p')
                        os << std::hex;

                    if (_conv == 'X')
                        os << std::uppercase;

                    if (_zero && !_left)
                        os << std::setfill('0');

                    if (_sign)
                        os << std::showpos;

                    if (_left) os << std::left;

                    if (_alt || _conv == 'p')
                        os << std::showbase;

                    if (_prec != 0 || is_nonzero(arg))
                    {
                        print_integer(os, arg, _conv == 'd' || _conv == 'i');
                    }
                }
                break;
                case 'e':
                case 'E':
                case 'f':
                case 'F':
                case 'g':
                case 'G':
                case 'a':
                case 'A':
                {
                    if (!is_arithmetic<T>::value)
                        throw std::logic_error("Argument is not arithmetic");

                    os << std::setw(_width) <<
                          std::setprecision(_prec == -1 ? 6 : _prec);
                    os.unsetf(std::ios::floatfield);

                    if (_conv == 'E' || _conv == 'F' ||
                        _conv == 'G' || _conv == 'A')
                        os << std::uppercase;

                    if (_conv == 'e' || _conv == 'E')
                        os << std::scientific;

                    if (_conv == 'f' || _conv == 'F')
                        os << std::fixed;

                    if (_conv == 'a' || _conv == 'A')
                        os << std::fixed; //FIXME: hexfloat

                    if (_zero && !_left)
                        os << std::setfill('0');

                    if (_sign)
                        os << std::showpos;

                    if (_left)
                        os << std::left;

                    if (_alt)
                        os << std::showpoint;

                    print_float(os, arg);
                }
                break;
                case 'c':
                {
                    if (!is_integral<T>::value)
                        throw std::logic_error("Argument is not integral");

                    while (_width --> 1) os << ' ';
                    print_char(os, arg);
                }
                break;
                case 's':
                {
                    if (!is_same<decay_t<T>,char*>::value &&
                        !is_same<decay_t<T>,const char*>::value &&
                        !is_same<decay_t<T>,std::string>::value)
                        throw std::logic_error("Argument is not a string");

                    print_string(os, arg, _width, _prec, _left);
                }
                break;
                case 'j':
                {
                    os << arg;
                }
                break;
            }
        }

        operator bool() const
        {
            return _conv != 0;
        }
};

template <int I, int N, typename... Args>
struct printos_printer
{
    printos_printer(std::ostream& os, const printos_helper<Args...>& h, std::string::const_iterator i)
    {
        printf_conversion conv(os, h.fmt, i);
        if (!conv) throw std::logic_error("More arguments than conversions");
        conv.print(os, std::get<I>(h.args));
        printos_printer<I+1, N, Args...>(os, h, i);
    }
};

template <int N, typename... Args>
struct printos_printer<N, N, Args...>
{
    printos_printer(std::ostream& os, const printos_helper<Args...>& h, std::string::const_iterator i)
    {
        printf_conversion conv(os, h.fmt, i);
        if (conv) throw std::logic_error("More conversions than arguments");
    }
};

template <typename T, typename=void>
struct is_container : std::false_type {};
template <typename T>
struct is_container<T, enable_if_exists_t<typename T::value_type,
                                          typename T::iterator>>
    : std::true_type {};

template <typename T, typename=void>
struct is_map : std::false_type {};
template <typename T>
struct is_map<T, enable_if_exists_t<typename T::key_type,
                                    typename T::mapped_type>>
    : std::true_type {};

template <typename T>
struct is_string : std::false_type {};
template <typename T>
struct is_string<std::basic_string<T>> : std::true_type {};

}
}

namespace std
{

template <typename... Args>
std::ostream& operator<<(std::ostream& os, const stl_ext::detail::printos_helper<Args...>& h)
{
    stl_ext::detail::printos_printer<0, sizeof...(Args), Args...>(os, h, h.fmt.begin());
    return os;
}

template <typename... Args>
std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& t)
{
    os << '(';
    stl_ext::detail::print_tuple_helper<sizeof...(Args), Args...>(os, t);
    os << ')';
    return os;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& t)
{
    os << '(' << t.first << ", " << t.second << ')';
    return os;
}

template <typename T>
stl_ext::enable_if_t<stl_ext::detail::is_map<T>::value, std::ostream&>
operator<<(std::ostream& os, const T& v)
{
    auto it0 = v.begin();
    auto it1 = v.end();
    os << "{";
    if (it0 != it1)
    {
        os << it0->first << ": " << it0->second;
        for (++it0;it0 != it1;++it0)
            os << ", " << it0->first << ": " << it0->second;
    }
    os << "}";
    return os;
}

template <typename T>
stl_ext::enable_if_t<stl_ext::detail::is_container<T>::value &&
                     !stl_ext::detail::is_map<T>::value &&
                     !stl_ext::detail::is_string<T>::value, std::ostream&>
operator<<(std::ostream& os, const T& v)
{
    auto it0 = v.begin();
    auto it1 = v.end();
    os << "[";
    if (it0 != it1)
    {
        os << *it0;
        for (++it0;it0 != it1;++it0) os << ", " << *it0;
    }
    os << "]";
    return os;
}

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const T v[N])
{
    os << "[";
    if (N)
    {
        os << v[0];
        for (int i = 1;i < N;i++) os << ", " << v[i];
    }
    os << "]";
    return os;
}

}

#if 0

namespace stl_ext
{

template <typename... Args>
detail::printos_helper<Args...> printos(const std::string& fmt, Args&&... args)
{
    return detail::printos_helper<Args...>(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
void print(const std::string& fmt, Args&&... args)
{
    cout << printos(fmt, std::forward<Args>(args)...);
}

template <typename... Args>
detail::printos_helper<Args...> operator%(const std::string& fmt, std::tuple<Args&&...>&& args)
{
    return detail::printos_helper<Args...>(fmt, std::move(args));
}

template <typename... Args>
std::tuple<Args&&...> fmt(Args&&... args)
{
    return std::forward_as_tuple(std::forward<Args>(args)...);
}

namespace detail
{
    template <typename T>
    struct sigfig_printer
    {
        T val;
        int sigfigs;
    };

    template <typename T>
    enable_if_not_complex_t<T,std::ostream&>
    operator<<(std::ostream& os, const sigfig_printer<T>& p)
    {
        ios_flag_saver flags(os);

        double l = log10(std::abs(p.val));
        auto d = lrint(l < 0 ? l-1 : l);
        if (std::abs(d) > 2)
        {
            os << std::scientific << std::setprecision(p.sigfigs) << p.val;
        }
        else
        {
            os << std::fixed << std::setprecision(p.sigfigs-d) << p.val;
        }

        return os;
    }

    template <typename T>
    enable_if_complex_t<T,std::ostream&>
    operator<<(std::ostream& os, const sigfig_printer<T>& p)
    {
        typedef real_type_t<T> R;
        return os << '(' << sigfig_printer<R>{p.val.real(), p.sigfigs} << ", " <<
                            sigfig_printer<R>{p.val.imag(), p.sigfigs} << ')';
    }
}

template <typename T>
detail::sigfig_printer<T> printToAccuracy(const T& value, double accuracy)
{
    int sigfigs = static_cast<int>(lrint(ceil(-log10(accuracy))));
    return detail::sigfig_printer<T>{value, sigfigs};
}

}

#endif

#endif
