#ifndef MP_COMPLEX_HPP
#define MP_COMPLEX_HPP

#include <iostream>
#include <sstream>
#include <cmath>
#include <complex>
#include <string>
#include <limits>
#include <iomanip>


/*
 * A lightweight templated implementation of complex numbers of arbitrary
 * scalar types.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     2/24/2019
 *    Modifyed by Rosa Martinez Corral to remove dependence on boost. 04/11/2019
 */
// ----------------------------------------------------------- //
//          CLASS TEMPLATE DEFINITIONS FOR mp_complex          //
// ----------------------------------------------------------- //
template <typename T>
class mp_complex
{
    private:
        // Real and imaginary parts
        T real_;
        T imag_;

    public:
        // ----------------------------------------------------------- //
        //                CONSTRUCTORS AND DESTRUCTORS                 //
        // ----------------------------------------------------------- //
        mp_complex()
        {
            /* 
             * Default constructor; set both real and imaginary parts
             * to zero. 
             */
            this->real_ = 0.0;
            this->imag_ = 0.0;
        }

        mp_complex(T real)
        {
            /*
             * Constructor with real scalar value.
             */
            this->real_ = real;
            this->imag_ = 0.0;
        }

        template <typename U>
        mp_complex(U real)
        {
            /*
             * Constructor with real scalar value of arbitrary type.
             */
            T real_in(real);
            this->real_ = real;
            this->imag_ = 0.0;
        }

        mp_complex(T real, T imag)
        {
            /*
             * Constructor with scalar values for the real and imaginary
             * parts.
             */
            this->real_ = real;
            this->imag_ = imag;
        }

        template <typename U, typename V>
        mp_complex(U real, V imag)
        {
            /*
             * Constructor with scalar values of arbitrary types.
             */
            T real_in(real);
            T imag_in(imag);
            this->real_ = real_in;
            this->imag_ = imag_in;
        }

        mp_complex(std::string real, std::string imag)
        {
            /*
             * Constructor with values specified via string inputs. 
             */
            T real_in(real);
            T imag_in(imag);
            this->real_ = real_in;
            this->imag_ = imag_in;
        }

        ~mp_complex()
        {
            /*
             * (Empty) destructor.
             */
        }

        // ----------------------------------------------------------- //
        //                      OUTPUT TO STREAM                       //
        // ----------------------------------------------------------- //
        friend std::ostream& operator<<(std::ostream& stream, const mp_complex<T>& z)
        {
            /*
             * Output a string representation of the ``mp_complex`` object
             * into a stream.
             */
            // Configure precision of the output stream
            stream << std::setprecision(std::numeric_limits<T>::max_digits10);

            // Output "a + b*i" if a is nonzero and b is positive
            if (z.real_ != 0 && z.imag_ > 0)
                stream << z.real_ << " + " << z.imag_ << "*i";
            // Output "a - b*i" if a is nonzero and b is negative
            else if (z.real_ != 0 && z.imag_ < 0) 
                stream << z.real_ << " - " << -z.imag_ << "*i";
            // Output "a" if b is zero
            else if (z.real_ != 0 && z.imag_ == 0)
                stream << z.real_;
            // Output "b" if a is zero
            else if (z.real_ == 0 && z.imag_ != 0)
                stream << z.imag_ << "*i";
            // Output "0" if a and b are zero
            else
                stream << z.real_;
            
            return stream;
        }

        // ----------------------------------------------------------- //
        //                 REAL/IMAGINARY PART GETTERS                 //
        // ----------------------------------------------------------- //
        T real() const
        {
            /*
             * Return real part.
             */
            return this->real_;
        }

        T imag() const
        {
            /*
             * Return imaginary part.
             */
            return this->imag_;
        }

        // ----------------------------------------------------------- //
        //                ASSIGNMENT OPERATOR OVERLOADS                // 
        // ----------------------------------------------------------- //
        mp_complex& operator=(const mp_complex& other)
        {
            /*
             * Trivial assignment.
             */
            if (this == &other)    // Guard against self-assignment
                return (*this);

            this->real_ = other.real();
            this->imag_ = other.imag();
            return (*this);
        }

        mp_complex& operator=(const T& other)
        {
            /*
             * Assignment to a real scalar value.
             */
            this->real_ = other;
            return (*this);
        }

        template <typename U>
        mp_complex& operator=(const U& other)
        {
            /*
             * Assignment to a real scalar value of arbitrary type.
             */
            T real_in(other);
            this->real_ = real_in;
            return (*this);
        }

        // ----------------------------------------------------------- //
        //            COMPLEX ARITHMETIC OPERATOR OVERLOADS            //
        // ----------------------------------------------------------- //
        mp_complex operator+(const mp_complex& other) const
        {
            /*
             * Addition by a second complex number.
             */
            return mp_complex(this->real_ + other.real(), this->imag_ + other.imag());
        }

        mp_complex operator+(const T& other) const
        {
            /*
             * Right-addition by a real scalar.
             */
            return mp_complex(this->real_ + other, this->imag_);
        }

        friend mp_complex operator+(const T& other, const mp_complex<T>& num)
        {
            /*
             * Left-addition by a real scalar.
             */
            return (num + other);
        }

        mp_complex& operator+=(const mp_complex& other)
        {
            /*
             * In-place addition by a second complex number.
             */
            this->real_ += other.real();
            this->imag_ += other.imag();
            return (*this);
        }

        mp_complex& operator+=(const T& other)
        {
            /*
             * In-place addition by a real scalar.
             */
            this->real_ += other;
            return (*this);
        }

        mp_complex operator-(const mp_complex& other) const
        {
            /*
             * Subtraction by a second complex number.
             */
            return mp_complex(this->real_ - other.real(), this->imag_ - other.imag());
        }

        mp_complex operator-(const T& other) const
        {
            /*
             * Subtraction by a real scalar.
             */
            return mp_complex(this->real_ - other, this->imag_);
        }

        friend mp_complex operator-(const T& other, const mp_complex<T>& num)
        {
            /*
             * Subtraction of a complex number from a real scalar.
             */
            return mp_complex(other - num.real(), -num.imag());
        }

        mp_complex& operator-=(const mp_complex& other)
        {
            /*
             * In-place subtraction by a second complex number.
             */
            this->real_ -= other.real();
            this->imag_ -= other.imag();
            return (*this);
        }

        mp_complex& operator-=(const T& other)
        {
            /*
             * In-place subtraction by a real scalar.
             */
            this->real_ -= other;
            return (*this);
        }

        mp_complex operator*(const mp_complex& other) const
        {
            /*
             * Multiplication by a second complex number.
             */
            T prod_real = this->real_ * other.real() - this->imag_ * other.imag();
            T prod_imag = this->real_ * other.imag() + this->imag_ * other.real();
            return mp_complex(prod_real, prod_imag);
        }

        mp_complex operator*(const T& other) const
        {
            /*
             * Right-multiplication by a real scalar.
             */
            return mp_complex(this->real_ * other, this->imag_);
        }

        friend mp_complex operator*(const T& other, const mp_complex<T>& num)
        {
            /*
             * Left-multiplication by a real scalar.
             */
            return (num * other);
        }

        mp_complex& operator*=(const mp_complex& other)
        {
            /*
             * In-place multiplication by a second complex number.
             */
            T prod_real = this->real_ * other.real() - this->imag_ * other.imag();
            T prod_imag = this->real_ * other.imag() + this->imag_ * other.real();
            this->real_ = prod_real;
            this->imag_ = prod_imag;
            return (*this);
        }

        mp_complex& operator*=(const T& other)
        {
            /*
             * In-place multiplication by a real scalar.
             */
            this->real_ *= other;
            this->imag_ *= other;
            return (*this);
        }

        mp_complex operator/(const mp_complex& other) const
        {
            /*
             * Division by a second complex number.
             */
            T a = this->real_, b = this->imag_;
            T c = other.real(), d = other.imag();
            T denom = c * c + d * d;
            T quot_real = (a * c + b * d) / denom;
            T quot_imag = (b * c - a * d) / denom;
            return mp_complex(quot_real, quot_imag);
        }

        mp_complex operator/(const T& other) const
        {
            /* 
             * Division by a real scalar.
             */
            return mp_complex(this->real_ / other, this->imag_ / other);
        }

        friend mp_complex operator/(const T& other, const mp_complex<T>& num)
        {
            /*
             * Division of a real scalar by a complex number.
             */
            T a = other, b = num.real(), c = num.imag();
            T denom = b * b + c * c;
            T quot_real = a * b / denom;
            T quot_imag = -a * c / denom;
            return mp_complex(quot_real, quot_imag);
        }

        mp_complex& operator/=(const mp_complex& other)
        {
            /*
             * In-place division by a second complex number.
             */
            T a = this->real_, b = this->imag_;
            T c = other.real(), d = other.imag();
            T denom = c * c + d * d;
            T quot_real = (a * c + b * d) / denom;
            T quot_imag = (b * c - a * d) / denom;
            this->real_ = quot_real;
            this->imag_ = quot_imag;
            return (*this);
        }

        mp_complex& operator/=(const T& other)
        {
            /*
             * In-place division by a real scalar.
             */
            this->real_ /= other;
            this->imag_ /= other;
            return (*this);
        }

        // ----------------------------------------------------------- //
        //          EQUALITY AND INEQUALITY OPERATOR OVERLOADS         //
        // ----------------------------------------------------------- //
        bool operator==(const mp_complex& other) const
        {
            /*
             * Equality. 
             */
            return (this->real_ == other.real_ && this->imag_ == other.imag_);
        }

        bool operator!=(const mp_complex& other) const
        {
            /*
             * Inequality.
             */
            return !(*this == other);
        }

        // ----------------------------------------------------------- //
        //                       UNARY FUNCTIONS                       //
        // ----------------------------------------------------------- //
        mp_complex conj() const
        {
            /*
             * Complex conjugate.
             */
            return mp_complex(this->real_, -this->imag_);
        }

        T abs() const
        {
            /*
             * Absolute value, i.e., modulus.
             */
            using std::sqrt;
            //using boost::multiprecision::sqrt;
            return sqrt(this->real_ * this->real_ + this->imag_ * this->imag_);
        }

    
        // ----------------------------------------------------------- //
        //                      BINARY FUNCTIONS                       //
        // ----------------------------------------------------------- //
        bool isclose(const mp_complex& other, T tol = 1e-12) const
        {
            /*
             * Return true if the mp_complex object is within the 
             * given tolerance of the given mp_complex object.
             */
            return ((*this - other).abs() < tol);
        }
};

#endif
