#ifndef UNIPOLY_HPP
#define UNIPOLY_HPP

#include <iostream>
#include <utility>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "complexnoboost.hpp"

/*
 * An templated implementation of the UniPoly class with arbitrary
 * numerical real-valued coefficients.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     2/26/2019
 * Modified by Rosa Martinez Corral to remove boost dependence. 04/11/2019
 */

namespace utils {

template <typename T>
class UniPoly;

template <typename T>
std::ostream& operator<<(std::ostream& stream, const UniPoly<T>& poly);

// ------------------------------------------------------ //
//            STRUCT DEFINITIONS FOR SolverStats          //
// ------------------------------------------------------ //
template <typename T>
struct SolverStats
{
    /*
     * A lightweight struct for storing statistics from the polynomial
     * solvers implemented in the UniPoly class. 
     */
    std::string method;     // Method ("weierstrass" or "aberth")
    unsigned niter;         // Number of iterations performed
    T delta;                // Maximum absolute change between 
                            // consecutive iterates of a root
};

// -------------------------------------------------------- //
//          TEMPLATE CLASS DEFINITIONS FOR UniPoly          //
// -------------------------------------------------------- //
template <typename T>
class UniPoly {
    /*
     * Bare-bones implementation of a single-variable polynomial class.
     */
    private:
        // var: A string containing the variable name
        std::string var;
        
        // coefs: A vector of real-valued coefficients
        std::vector<T> coefs;

        // -------------------------------------------------------- //
        //                  PRIVATE HELPER METHODS                  //
        // -------------------------------------------------------- //
        void eraseLeadZeros()
        {
            /*
             * Erase leading zeros from this->coefs.
             */
            while (this->coefs.front() == 0.0)
            this->coefs.erase(this->coefs.begin());
        }

        // -------------------------------------------------------- //
        //                  PRIVATE SOLVER METHODS                  //
        // -------------------------------------------------------- //
        std::vector<mp_complex<T> > iterWeierstrass(std::vector<mp_complex<T> > curr_roots)
        {
            /*
             * Computes one iteration of the Weierstrass method.
             */
            std::vector<mp_complex<T> > new_roots;

            // For each root z_i, correct it as:
            // z_i <- z_i - f(z_i) / (a_n \prod_{j!=i}{(z_i - z_j)})
            for (unsigned i = 0; i < curr_roots.size(); i++)
            {
                mp_complex<T> root = curr_roots[i];
                mp_complex<T> correction_denom = this->coefs[0];
                for (unsigned j = 0; j < curr_roots.size(); j++)
                {
                    if (j != i)
                        correction_denom *= (root - curr_roots[j]);
                }
                mp_complex<T> correction = this->eval(root) / correction_denom;
                new_roots.push_back(root - correction); 
            }

            return new_roots;
        }

        std::vector<mp_complex<T> > iterAberth(std::vector<mp_complex<T> > curr_roots)
        {
            /*
             * Computes one iteration of the Aberth-Ehrlich method. 
             */
            std::vector<mp_complex<T> > new_roots;

            // For each root z_i, correct it as:
            // z_i -= N(z_i) / (1 - N(z_i) * \sum_{j!=i}{1/(z_i - z_j)}i)
            // where N(z_i) = f(z_i) / f'(z_i)
            UniPoly<T> deriv = this->diff();
            for (unsigned i = 0; i < curr_roots.size(); i++)
            {
                mp_complex<T> root = curr_roots[i];
                mp_complex<T> newton = this->eval(root) / deriv.eval(root);
                mp_complex<T> correction_denom = 0.0;
                for (unsigned j = 0; j < curr_roots.size(); j++)
                {
                    if (j != i)
                        correction_denom += (1.0 / (root - curr_roots[j]));
                }
                mp_complex<T> correction = newton / (1.0 - newton * correction_denom);
                new_roots.push_back(root - correction); 
            }

            return new_roots;
        }

        std::pair<std::vector<mp_complex<T> >, SolverStats<T> > rootsWeierstrass(unsigned max_iter = 5000,
                                                                                 T tol = 1e-20)
        {
            /*
             * Computes all complex roots via the Weierstrass method.
             *
             * TODO Change initialization
             */
            // Initialize roots to powers of (0.4 + 0.9i)
            std::vector<mp_complex<T> > roots;
            mp_complex<T> z(0.4, 0.9);
            mp_complex<T> root_init;
            unsigned degree = this->coefs.size() - 1;
            for (unsigned i = 0; i < degree; i++)
            {
                root_init = (i == 0) ? z : (roots.back() * z);
                roots.push_back(root_init);
            }

            // Run desired number of Weierstrass iterations
            unsigned niter = 0;
            T delta = 2.0 * tol;
            std::vector<mp_complex<T> > new_roots; 
            while (niter < max_iter && delta > tol)
            {
                new_roots = this->iterWeierstrass(roots);
                niter += 1;
                // Compute the maximum update magnitude over all roots
                delta = 0;
                for (unsigned i = 0; i < degree; i++)
                {
                    T update = (new_roots[i] - roots[i]).abs();
                    if (update > delta)
                        delta = (new_roots[i] - roots[i]).abs();
                }
                // Update roots after computing update magnitudes
                roots = new_roots;
            }

            // Store solution statistics in a SolverStats object
            SolverStats<T> stats = {.method = "weierstrass", .niter = niter, .delta = delta};

            return std::make_pair(roots, stats);
        }

        std::pair<std::vector<mp_complex<T> >, SolverStats<T> > rootsAberth(unsigned max_iter = 5000,
                                                                            T tol = 1e-20)
        {
            /*
             * Computes all complex roots via the Aberth-Ehrlich method.
             *
             * TODO Change initialization
             */
            // Initialize roots to powers of (0.4 + 0.9i)
            std::vector<mp_complex<T> > roots;
            mp_complex<T> z(0.4, 0.9);
            mp_complex<T> root_init;
            unsigned degree = this->coefs.size() - 1;
            for (unsigned i = 0; i < degree; i++)
            {
                root_init = (i == 0) ? z : (roots.back() * z);
                roots.push_back(root_init);
            }

            // Run desired number of Aberth-Ehrlich iterations
            unsigned niter = 0;
            T delta = 2.0 * tol;
            std::vector<mp_complex<T> > new_roots; 
            while (niter < max_iter && delta > tol)
            {
                new_roots = this->iterAberth(roots);
                niter += 1;
                // Compute the maximum update magnitude over all roots
                delta = 0;
                for (unsigned i = 0; i < degree; i++)
                {
                    T update = (new_roots[i] - roots[i]).abs();
                    if (update > delta) delta = update;
                }
                // Update roots after computing update magnitudes
                roots = new_roots;
            }

            // Store solution statistics in a SolverStats object
            SolverStats<T> stats = {.method = "aberth", .niter = niter, .delta = delta};

            return std::make_pair(roots, stats);
        }

    public:
        // -------------------------------------------------------- //
        //                       CONSTRUCTORS                       //
        // -------------------------------------------------------- //
        UniPoly()
        {
            /* 
             * Empty constructor; initialize to the zero polynomial. 
             */
            this->var = "x";
            this->coefs.push_back(0.0);
        }

        UniPoly(std::string var)
        {
            /*
             * Constructor with input variable; initialize to the zero
             * polynomial. 
             */
            this->var = var;
            this->coefs.push_back(0.0);
        }

        UniPoly(std::string var, std::vector<std::string> coefs)
        {
            /*
             * Constructor with input variable and string-valued coefficients.
             */
            this->var = var;
            for (auto&& coef_str : coefs)
            {
                T coef(coef_str);
                this->coefs.push_back(coef);
            }
            this->eraseLeadZeros();    // Remove leading zeros
        }

        UniPoly(std::string var, std::vector<T> coefs)
        {
            /*
             * Constructor with input variable and coefficients.
             */
            this->var = var;
            this->coefs = coefs;
            this->eraseLeadZeros();    // Remove leading zeros
        }

        ~UniPoly()
        {
            /* 
             * (Empty) destructor.
             */
        }

        // -------------------------------------------------------- //
        //                   GETTERS AND SETTERS                    //
        // -------------------------------------------------------- //
        std::string getVar() const
        {
            /*
             * Return this->var.
             */
            return this->var;
        }

        std::vector<T> getCoefs() const
        {
            /*
             * Return this->coefs.
             */
            return this->coefs;
        }

        // -------------------------------------------------------- //
        //                     OUTPUT TO STREAM                     //
        // -------------------------------------------------------- //
        friend std::ostream& operator<<(std::ostream& stream, const UniPoly<T>& poly)
        {
            /*
             * Output a string representation of the polynomial to stream.
             */
            // Configure precision of the output stream
            stream << std::setprecision(std::numeric_limits<T>::max_digits10);

            unsigned degree, power;
            T coef;

            // Run through the terms in descending power ...
            degree = poly.coefs.size() - 1;
            for (unsigned idx = 0; idx < poly.coefs.size(); ++idx)
            {
                // Get the power of the current term
                power = degree - idx;

                // Grab its coefficient
                coef = poly.coefs[idx];

                // Skip over all zero terms
                if (coef == 0.0) continue;
                // If the term is leading, simply add the term (no signs)
                else if (power == degree) 
                {
                    switch (power)
                    {
                        // If the term is a constant, no need to indicate variable
                        case 0:
                            stream << coef;
                            break;
                        // If the term is linear, no need to indicate power
                        case 1:
                            if (coef == 1.0)
                                stream << poly.var;
                            else
                                stream << coef << "*" << poly.var;
                            break;
                        // Otherwise, format term as <coef>*<var>^<power>
                        default:
                            if (coef == 1.0)
                                stream << poly.var << "^" << power;
                            else
                                stream << coef << "*" << poly.var << "^" << power;
                            break;
                    }
                }
                // If the term is positive, add a plus sign followed by the term
                else if (coef > 0)
                {
                    switch (power)
                    {
                        // If the term is a constant, no need to indicate variable
                        case 0:
                            stream << "+ " << coef;
                            break;
                        // If the term is linear, no need to indicate power
                        case 1:
                            if (coef == 1.0)
                                stream << "+ " << poly.var;
                            else
                                stream << "+ " << coef << "*" << poly.var;
                            break;
                        // Otherwise, format term as <coef>*<var>^<power>
                        default:
                            if (coef == 1.0)
                                stream << poly.var << "^" << power;
                            else
                                stream << "+ " << coef << "*" << poly.var << "^"
                                       << power;
                            break;
                    }
                }
                // If the term is negative, add a minus sign followed by the term
                // (with the minus sign stripped from the coefficient)
                else
                {
                    switch (power)
                    {
                        // If the term is a constant, no need to indicate variable
                        case 0:
                            stream << "- " << -coef;
                            break;
                        // If the term is linear, no need to indicate power
                        case 1:
                            if (-coef == 1.0)
                                stream << "- " << poly.var;
                            else 
                                stream << "- " << -coef << "*" << poly.var;
                            break;
                        // Otherwise, format term as <coef>*<var>^<power>
                        default:
                            if (-coef == 1.0)
                                stream << "- " << poly.var << "^" << power;
                            else
                                stream << "- " << -coef << "*" << poly.var << "^"
                                       << power;
                            break;
                    }
                }
                // Demarcate terms with spaces
                if (idx < poly.coefs.size() - 1) stream << " ";
            }
            return stream;
        }

        // -------------------------------------------------------- //
        //                     ADDITION OPERATORS                   //
        // -------------------------------------------------------- //
        UniPoly operator+(const UniPoly& other) const
        {
            /*
             * Addition with a second polynomial.  
             */
            // Check that the two polynomials have the same variable
            if ((this->var).compare(other.var))
                throw std::invalid_argument("Variables of input polynomials do not match");

            // Copy coefficient vectors from the two UniPoly objects
            // and match their lengths
            std::vector<T> this_coefs(this->coefs);
            std::vector<T> other_coefs(other.coefs);
            int diff = this_coefs.size() - other_coefs.size();
            if (diff > 0)           // Prepend zeros onto other_coefs
            {
                for (int i = 0; i < diff; i++)
                    other_coefs.insert(other_coefs.begin(), 0.0);
            }
            else if (diff < 0)      // Prepend zeros onto this_coefs
            {
                for (int i = 0; i < -diff; i++)
                    this_coefs.insert(this_coefs.begin(), 0.0);
            }

            // Add each pair of matching coefficients
            std::vector<T> sum_coefs;
            for (int i = 0; i < this_coefs.size(); i++)
                sum_coefs.push_back(this_coefs[i] + other_coefs[i]);

            return UniPoly(this->var, sum_coefs);
        }

        UniPoly operator+(const T& other) const
        {
            /*
             * Addition with a scalar value on the right.
             */
            std::vector<T> sum_coefs(this->coefs);
            sum_coefs.back() += other;
            return UniPoly(this->var, sum_coefs);
        }

        friend UniPoly operator+(const T& other, const UniPoly<T>& poly)
        {
            /*
             * Addition with a scalar value on the left.
             */
            return (poly + other);
        }

        UniPoly& operator+=(const UniPoly& other)
        {
            /*
             * In-place addition with a second polynomial. 
             */
            // Check that the two polynomials have the same variable
            if ((this->var).compare(other.var))
                throw std::invalid_argument("Variables of input polynomials do not match");

            // Copy coefficient vector from the second UniPoly object
            // and match this->coefs to its length
            std::vector<T> other_coefs(other.coefs);
            int diff = this->coefs.size() - other_coefs.size();
            if (diff > 0)           // Prepend zeros onto other_coefs
            {
                for (unsigned i = 0; i < diff; i++)
                    other_coefs.insert(other_coefs.begin(), 0.0);
            }
            else if (diff < 0)      // Prepend zeros onto this_coefs
            {
                for (unsigned i = 0; i < -diff; i++)
                    this->coefs.insert(this->coefs.begin(), 0.0);
            }

            // Add each pair of matching coefficients
            for (unsigned i = 0; i < this->coefs.size(); i++)
                this->coefs[i] += other_coefs[i];

            return (*this);
        }

        UniPoly& operator+=(const T& other)
        {
            /*
             * In-place addition with a scalar value.
             */
            this->coefs.back() += other;
            return (*this);
        }

        // -------------------------------------------------------- //
        //                     NEGATION OPERATOR                    //
        // -------------------------------------------------------- //
        UniPoly operator-() const
        {
            /*
             * Negation.
             */
            std::vector<T> neg_coefs;
            for (auto&& coef : this->coefs) neg_coefs.push_back(-coef);
            return UniPoly(this->var, neg_coefs);
        }

        // -------------------------------------------------------- //
        //                   SUBTRACTION OPERATORS                  //
        // -------------------------------------------------------- //
        UniPoly operator-(const UniPoly& other) const
        {
            /*
             * Subtraction by a second polynomial. 
             */
            return (*this + (-other));
        }

        UniPoly operator-(const T& other) const
        {
            /*
             * Subtraction by a scalar value.
             */
            return (*this + (-other));
        }

        friend UniPoly operator-(const T& other, const UniPoly<T>& poly)
        {
            /*
             * Subtraction of a UniPoly object from a scalar value.
             */
            return (-poly + other);
        }

        UniPoly& operator-=(const UniPoly& other)
        {
            /*
             * In-place subtraction by a second polynomial.
             */
            this->operator+=(-other);
            return (*this);
        }

        UniPoly& operator-=(const T& other)
        {
            /*
             * In-place subtraction by a scalar value.
             */
            this->operator+=(-other);
            return (*this);
        }

        // -------------------------------------------------------- //
        //                  MULTIPLICATION OPERATORS                //
        // -------------------------------------------------------- //
        UniPoly operator*(const UniPoly& other)
        {
            /*
             * Multiplication by a second polynomial. 
             */
            unsigned i, j;

            // Check that the two polynomials have the same variable
            if ((this->var).compare(other.var))
                throw std::invalid_argument("Variables of input polynomials do not match");

            // Compute the degrees of the two polynomials and their product
            unsigned this_degree = (this->coefs).size() - 1;
            unsigned other_degree = (other.coefs).size() - 1;
            unsigned prod_degree = this_degree + other_degree;

            // Initialize a vector of coefficients for the product
            std::vector<T> prod_coefs;
            for (i = 0; i < prod_degree + 1; i++) prod_coefs.push_back(0.0);

            // For each pair of terms in this and other, update the
            // corresponding coefficient in the product
            unsigned deg_this_i, deg_other_j, deg_prod_ij, idx_prod_ij;
            for (i = 0; i < this_degree + 1; i++)
            {
                for (j = 0; j < other_degree + 1; j++)
                {
                    // Get the degrees of the ith/jth terms in this and other
                    deg_this_i = this_degree - i;
                    deg_other_j = other_degree - j;

                    // Degree of the corresponding term in the product is
                    // the sum of the two degrees 
                    deg_prod_ij = deg_this_i + deg_other_j;
                    idx_prod_ij = prod_degree - deg_prod_ij;

                    // Update the corresponding term in the product
                    prod_coefs[idx_prod_ij] = (
                        prod_coefs[idx_prod_ij] + (this->coefs[i] * other.coefs[j])
                    );
                }
            }
            return UniPoly(this->var, prod_coefs);
        }

        UniPoly operator*(const T& other) const
        {
            /*
             * Multiplication by a scalar on the right.
             */
            std::vector<T> mult_coefs;
            for (auto&& coef : this->coefs) mult_coefs.push_back(other * coef);
            return UniPoly(this->var, mult_coefs);
        }

        friend UniPoly operator*(const T& other, const UniPoly<T>& poly)
        {
            /*
             * Multiplication by a scalar on the left.
             */
            return (poly * other);
        }

        UniPoly& operator*=(const UniPoly& other)
        {
            /*
             * In-place multiplication by a second polynomial.
             */
            unsigned i, j;

            // Check that the two polynomials have the same variable
            if ((this->var).compare(other.var))
                throw std::invalid_argument("Variables of input polynomials do not match");

            // Compute the degrees of the two polynomials and their product
            unsigned this_degree = (this->coefs).size() - 1;
            unsigned other_degree = (other.coefs).size() - 1;
            unsigned prod_degree = this_degree + other_degree;

            // Initialize a vector of coefficients for the product
            std::vector<T> prod_coefs;
            for (i = 0; i < prod_degree + 1; i++) prod_coefs.push_back(0.0);

            // For each pair of terms in this and other, update the
            // corresponding coefficient in the product
            unsigned deg_this_i, deg_other_j, deg_prod_ij, idx_prod_ij;
            for (i = 0; i < this_degree + 1; i++)
            {
                for (j = 0; j < other_degree + 1; j++)
                {
                    // Get the degrees of the ith/jth terms in this and other
                    deg_this_i = this_degree - i;
                    deg_other_j = other_degree - j;

                    // Degree of the corresponding term in the product is
                    // the sum of the two degrees 
                    deg_prod_ij = deg_this_i + deg_other_j;
                    idx_prod_ij = prod_degree - deg_prod_ij;

                    // Update the corresponding term in the product
                    prod_coefs[idx_prod_ij] = (
                        prod_coefs[idx_prod_ij] + (this->coefs[i] * other.coefs[j])
                    );
                }
            }

            this->coefs = prod_coefs;
            return (*this);
        }

        UniPoly& operator*=(const T& other)
        {
            /*
             * In-place multiplication by a scalar value.
             */
            for (auto&& coef : this->coefs) coef *= other;
            return (*this);
        }

        // -------------------------------------------------------- //
        //                 SCALAR DIVISION OPERATORS                //
        // -------------------------------------------------------- //
        UniPoly operator/(const T& other)
        {
            /*
             * Division by a scalar value.
             */
            std::vector<T> div_coefs;
            for (auto&& coef : this->coefs) div_coefs.push_back(coef / other);
            return UniPoly(this->var, div_coefs);
        }

        UniPoly& operator/=(const T& other)
        {
            /*
             * In-place division by a scalar value.
             */
            for (auto&& coef : this->coefs) coef /= other;
            return (*this);
        }

        // -------------------------------------------------------- //
        //             EQUALITY AND INEQUALITY OPERATORS            //
        // -------------------------------------------------------- //
        bool operator==(const UniPoly& other)
        {
            /*
             * Equality.
             */
            // First check that the degrees are the same
            if (this->coefs.size() != other.coefs.size()) return false;
            else
            {
                // Then check that each pair of coefficients are equal
                for (unsigned i = 0; i < this->coefs.size(); i++)
                {
                    if (this->coefs[i] != other.coefs[i])
                        return false;
                }
            }
            // Finally check that the variables are equal
            return (!(this->var).compare(other.var));
        }

        bool operator!=(const UniPoly& other)
        {
            /*
             * Inequality.
             */
            return !((*this) == other);
        }

        // -------------------------------------------------------- //
        //                      OTHER OPERATORS                     //
        // -------------------------------------------------------- //
        bool isPositive()
        {
            /*
             * Return true if all coefficients are positive.
             */
            for (auto&& coef : this->coefs)
            {
                if (coef < 0.0) return false;
            }
            return true;
        }

        T leadCoef()
        {
            /*
             * Return the leading coefficient.
             */
            return this->coefs[0];
        }

        UniPoly monic()
        {
            /*
             * Return the polynomial divided by its leading coefficient.
             */
            T lead_coef = this->coefs[0];

            // Make a copy of the coefficients vector ...
            std::vector<T> monic_coefs(this->coefs);

            // ... and divide each by the leading coefficient
            for (auto&& coef : monic_coefs) coef /= lead_coef;

            // Instantiate the monic polynomial
            return UniPoly(this->var, monic_coefs);
        }

        UniPoly reduce()
        {
            /*
             * Return the polynomial divided by the highest power of 
             * this->var such that the result remains polynomial,
             * that is, the lowest common power of this->var among
             * all terms.
             */
            // Make a copy of the coefficients vector ...
            std::vector<T> reduced_coefs(this->coefs);

            // ... and pop all trailing zeros
            while (reduced_coefs.back() == 0.0) reduced_coefs.pop_back();

            // Instantiate the reduced polynomial
            return UniPoly(this->var, reduced_coefs);
        }

        T eval(T x)
        {
            /*
             * Evaluation at given real scalar value.
             */
            // Evaluate the polynomial via Horner's method:
            // a_0 + a_1 x + ... + a_n x^n
            //     = a_0 + (a_1 + x (a_2 + ... + x (a_{n-1} + x a_n) ... ))
            T value = this->coefs[0];
            unsigned degree = this->coefs.size() - 1;
            for (unsigned i = 1; i < this->coefs.size(); i++)
            {
                value *= x;
                value += this->coefs[i];
            }
            return value;
        }

        mp_complex<T> eval(mp_complex<T> x)
        {
            /*
             * Evaluation at given complex scalar value. 
             */
            // Evaluate the polynomial via Horner's method:
            // a_0 + a_1 x + ... + a_n x^n
            //     = a_0 + (a_1 + x (a_2 + ... + x (a_{n-1} + x a_n) ... ))
            mp_complex<T> value = this->coefs[0];
            unsigned degree = (this->coefs).size() - 1;
            for (unsigned i = 1; i < this->coefs.size(); i++)
            {
                value *= x;
                value += this->coefs[i];
            }
            return value;
        }

        UniPoly diff()
        {
            /*
             * Differentiation by this->var.
             */
            // If the polynomial is a constant, return zero
            int degree = this->coefs.size() - 1;
            if (degree == 0) return UniPoly(this->var);

            // Otherwise, iterate over all terms but the constant term
            std::vector<T> deriv_coefs;
            for (int i = 0; i < this->coefs.size() - 1; i++)
            {
                T coef = this->coefs[i];
                T power(degree - i);
                deriv_coefs.push_back(coef * power);
            }

            // Instantiate the derivative polynomial
            return UniPoly(this->var, deriv_coefs);
        }

        std::pair<std::vector<mp_complex<T> >, SolverStats<T> > roots(std::string method = "aberth",
                                                                      T tol = 1e-20, 
                                                                      unsigned max_iter = 5000)
        {
            /*
             * Computes all complex roots via the Weierstrass method or
             * the Aberth-Ehrlich method.
             */
            if (!method.compare("weierstrass"))
                return this->rootsWeierstrass(max_iter, tol);
            else
                return this->rootsAberth(max_iter, tol);
        }

};

}   // namespace utils

#endif
