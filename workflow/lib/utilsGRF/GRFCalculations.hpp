#include <pybind11/pybind11.h>
 #include <pybind11/numpy.h>
#include <stdlib.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>
#include <polynomial.hpp>
#include <posstpfunc_cpp_boost_multiprT.hpp>
#include <type_traits>
 
using namespace std;



namespace py=pybind11;

//convert long double or boost multiprecision type to double
//create a template that defines the general behaviour, and specialise for the case where it is long double. 

template <typename T>
double convert_to_double(T a){
	return a.template convert_to<double>();
}

template <>
double convert_to_double(long double a){
	return (double) a;
}


template <typename T, typename Tpolyc, typename polytype, typename thresholdtype>
class GRFCalculations{

	/*T: type for numerator and denominator of GRF and their operations
    Tpolyc: typedef number<mpc_complex_backend<precision_poly> > mympc_poly;
    polytype: typedef Polynomial<precision_poly> polytype;
    (Tpolyc and polytype should be of same precision, used to compute the zeros of a polynomial)
    threshold: threshold to decide if imaginary part of root is zero. With its own type.
	*/

	protected:
	    vector<T> num;
		vector<T> den;
		vector<T> rhos;

    public:

		unsigned int precision;
		//unsigned int precision_poly;
		

		//GRFCalculations(unsigned int &precision, unsigned int &precision_poly) :  precision(precision), precision_poly(precision_poly) {};
	    //GRFCalculations() :  precision(precision) {};

        void set_precision(unsigned int prec){
	        precision=prec;
	    }

	    unsigned int get_precision(){
	        return precision;
	    }
	    
	    void fill_num_den(py::array_t<double> parsar, py::array_t<double>othervars);
        
        void fill_rhos(py::array_t<double> parsar, py::array_t<double>othervars, double xval);

		void print_num_den(){
	    	cout.precision(precision);
	    	cout << num[0] <<"," << num[1] << "\n";
	    	cout << den[0] <<"," << den[1] << "\n";
	    	cout.flush();
	    }

	    double interfaceGRF(double varGRFval ){

	    	T result;
	    	T varGRFval_T=varGRFval;
            result=GRFatxonly<T>(num,den,varGRFval_T);
            return convert_to_double<T>(result);

        }

        py::array_t<double> getrhos(int rhsz){
        	//"""rhsz is the number of rhos. If we just calculate it based on the size of rhos vector, it is not working"""
        	py::array_t<double> resultpy = py::array_t<double>(rhsz);
		    py::buffer_info bufresultpy = resultpy.request();
		    double *ptrresultpy=(double *) bufresultpy.ptr;
		    
        	for (int i=0; i<rhsz;i++){
        		ptrresultpy[i]=convert_to_double<T>(rhos[i]);
        	}
        	return resultpy;

        }

        py::array_t<double> interface_return_num_den(int ncoeffs){

        	py::array_t<double> resultpy = py::array_t<double>(2*ncoeffs);
		    py::buffer_info bufresultpy = resultpy.request();
		    double *ptrresultpy=(double *) bufresultpy.ptr;
		    

		    for (int i=0;i<num.size();i++){
		    	ptrresultpy[i]=convert_to_double<T>(num[i]);
		    }
		    int j=num.size();
		    for (int i=0;i<den.size();i++){
		    	ptrresultpy[i+j]=convert_to_double<T>(den[i]);
		    }		    
		    resultpy.resize({2,ncoeffs});

		    return resultpy;

        }

        py::array_t<double> interfaceps(bool verbose=false, double thresholdimag_=1e-15, bool minx0=false, bool maxx1=false, bool writex05coeffs=false, bool absder=false, bool normalisefirst=true, string fnamecoeffs="filename.txt") {

        
            vector<T>result;
            vector<T> min_max(2);
            if (minx0 and maxx1){
            	min_max={0,1};
            }else{
            compute_min_maxGRF<T>(num,den,min_max);
            if (minx0){
            	min_max[0]=0;
            }
            if (maxx1){
            	min_max[1]=1;
            }
            }

            T Gmin=min_max[0];
            T Gmax=min_max[1];
            T midpoint=Gmin+0.5*(Gmax-Gmin);

            if (verbose){
            	cout << "Gmin: " << Gmin << "\n";
            	cout << "Gmax: " << Gmax << "\n";
            	cout << "midpoint: " << midpoint << "\n";
            }

            
            if  (Gmax<0){
                result={-1.0,-1.0,-1.0};
            }else{
            	thresholdtype thresholdimag=thresholdimag_;
                result=compute_pos_stp<T,Tpolyc,polytype,thresholdtype>(num,den,"aberth",verbose,midpoint,thresholdimag,writex05coeffs, absder, normalisefirst,fnamecoeffs);

            }

            py::array_t<double> resultpy = py::array_t<double>(3);
            py::buffer_info bufresultpy = resultpy.request();
            double *ptrresultpy=(double *) bufresultpy.ptr;
            ptrresultpy[0]=convert_to_double<T>(result[0]);
            ptrresultpy[1]=convert_to_double<T>(result[1]);
            ptrresultpy[2]=convert_to_double<T>(result[2]);

            return  resultpy;
        }

        py::array_t<double> interfacemonotonic(double thresholdimag_=1e-15) {
    
            vector<T> result;
            thresholdtype thresholdimag=thresholdimag_;
            result=compute_monotonic<T,Tpolyc,polytype,thresholdtype>(num,den,thresholdimag); //return {-1} if derivative is 0, {-2} if no roots for the derivative of the GRF, -3 for each root out of the 10^-15,10^15 range, and the roots otherwise
            int n=result.size();
            py::array_t<double> resultpy = py::array_t<double>(n);
		    py::buffer_info bufresultpy = resultpy.request();
		    double *ptrresultpy=(double *) bufresultpy.ptr;
		    for (int i=0;i<n;i++){
		        //py::print("Result is",result[i]);
		        if (result[i]<-0.5){
		        ptrresultpy[i]=convert_to_double<T>(result[i]);
		        }else{
		        if ((result[i]<pow(10.0,15))&&(result[i]>pow(10.0,-15))){
		            ptrresultpy[i]=convert_to_double<T>(result[i]);
		        }else{
		            ptrresultpy[i]=-3;
		        }
		        }
		    }
            return resultpy;
        }
        

};
