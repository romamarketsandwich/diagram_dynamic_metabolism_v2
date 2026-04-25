

#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <polynomial.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/mpc.hpp>


using namespace std;
using namespace Eigen;
using boost::lexical_cast;
using std::string;

using boost::multiprecision::number; 
using boost::multiprecision::mpfr_float_backend; 
using boost::multiprecision::mpc_complex_backend; 

typedef number<mpfr_float_backend<40> >  mpfr_40;
typedef number<mpc_complex_backend<40> > mpc_40;
//namespace py=pybind11;

/* 
    Code to compute position and steepness for a GRF, and function to assess if it is monotonic or not.
    Copyright (C) <2021>  <Rosa Martinez Corral>
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""
*/

vector <long double> trim_zerocoeff(vector<long double> coeffs){
	vector<long double>::const_iterator first = coeffs.begin() + 1;
    vector<long double>::const_iterator last = coeffs.end();
    vector<long double> newVec(first, last);
    return newVec;
}



void get_positive_real_roots_aberth(vector<long double> coeffsx05, vector<long double> &pos_real){

    //cout << "solving with boost\n";
    long double root;

    unsigned max_iter = 10000; 
    double tol = std::numeric_limits<double>::epsilon(); 
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_real_distribution<> dist(0, 1);
    number<mpfr_float_backend<40> > threshold = std::numeric_limits<number<mpfr_float_backend<40> > >::epsilon(); 


    //If the constant coefficient of the polynomial is 0, 
    //e.g. 0+bx+ax^2=0
    //then it is equivalent to x(b+ax)=0
    //so it is equivalent to finding the roots of a polynomial of one degree less
    //the initialization method only works when constant coefficient is not zero, so update
    //cout << "trimming\n";
    //cout.flush();
    while ((abs(coeffsx05[0])<threshold)&&(coeffsx05.size()>1)){
        coeffsx05=trim_zerocoeff(coeffsx05);
    }
    //cout << "trimmed\n";
    //cout.flush();
    if (coeffsx05.size()>1){
        
        Polynomial<40> p(coeffsx05);
        //cout << "going to compute roots\n";
        //cout.flush();
        std::pair<std::vector<mpc_40>, bool> roots_computed = p.roots(max_iter,tol,tol,rng,dist);
        if (roots_computed.second){ //converged
        //cout << "roots computed ";
        //cout.flush();
        for (unsigned i=0;i<roots_computed.first.size();i++){
            /*cout << roots_computed.first[i];
            cout <<"\n";
            cout.flush();*/
            if (abs(roots_computed.first[i].imag()) < threshold){
                //py::print("root",i,roots[i]);
                
                if (roots_computed.first[i].real()>threshold){
                    pos_real.push_back(static_cast<long double>(roots_computed.first[i].real()));

                }
            }
        }
        }
    }
}


void get_positive_real_roots(vector<long double> &coeffs, vector<long double> &pos_real){
    //Returns roots of polynomial given by coeffs that are positive and real. Coeffs should be in increasing order. 
    //Computes eigenvalues of companion matrix. It is not always accurate.

    //vector<double>coeffs;

    std::vector<long double>::size_type Nc=coeffs.size();
    std::vector<long double>::size_type i,j, Nrows, lastidx;
    
    Nrows=Nc-1;
    lastidx=Nrows-1;

    Matrix< long double, Dynamic, Dynamic> A(Nrows,Nrows);
    A.setZero(Nrows,Nrows);
    //MatrixXd A = MatrixXd::Zero(Nrows,Nrows); //This would be the same. Matrix with dynamic allocation of rows and columns, filled with 0
    //But do not use the code below. It produces inaccurate results!!
    //for (i=1;i<Nrows;i++){
    //    for (j=0;j<Nrows;j++){
    //    A(i,j)=0; // 
    //    }
    //}
    
    for (i=1;i<Nrows;i++){
        A(i,i-1)=1; // fill diagonal (-1 diagonal)
    }
    for (i=0;i<Nrows;i++){
        A(i,lastidx)=-1 * coeffs[i]/coeffs[Nrows]; //fill coefficients
    }

    //VectorXcd rootsout;
    Matrix<complex<long double>,Dynamic,1> rootsout;
    EigenSolver<Matrix<long double, Dynamic, Dynamic>> es(A, false); //no need to compute eigenvectors
    rootsout=es.eigenvalues();

    //vector<double> pos_real;
    //py::print("The roots out are");
    //for (i=0;i<rootsout.size();i++){
    //    py::print(rootsout[i]);
    //}

    if (rootsout.size()>0){
    for (i=0;i<rootsout.size();i++){
        //py::print("Checking root",rootsout[i]);
        if (rootsout[i].imag() ==0){
            if (rootsout[i].real()>0){
                pos_real.push_back(rootsout[i].real());
            }     
        }
        }
    }
    
    //py::print("The roots found are");
    //for (i=0;i<pos_real.size();i++){
    //    py::print(pos_real[i]);
    //}

}


void product_coeffs_vectors(vector<long double> &c1, vector<long double>  &c2, vector<long double> &result){
    //vector<double> result(degree+degree);
    std::vector<long double>::size_type idx, i, j, n1, n2;
    n1=c1.size();
    n2=c2.size();
    for ( i=0;i<n1;i++){
        for ( j=0;j<n2;j++){
            idx=i+j;      
            result[idx]+=c1[i]*c2[j];
        }
    }
} 


void remove_zeros_endvector(vector<long double> &v){ //, vector<double> &vout){
    //py::print("removing zeros", v.size());
    //cout << "removing zeros";

    while ((abs(v.back())<1e-20)&(v.size()>0)){
        //py::print(v.back());
        v.pop_back();
    }
    
    /*std::vector<double>::size_type i;
    std::vector<double>::size_type j;
    std::vector<double>::size_type Nc;
    //int i,j,Nc;
    Nc=vin.size();
    //py::print(Nc-1,vin[Nc-1]);

    //std::vector<double>::reverse_iterator rit = vin.rbegin();
    if (abs(vin[Nc-1])<1e-15){
        for (i = Nc-1;i>=0;i--){
            if (abs(vin[i])>1e-15){
                for (j=0;j<=i;j++){
                    vout.push_back(vin[j]);
                }
                //py::print("new Nc",Nc);
                break;
            }
        }
    }else{
        for (j=0;j<Nc;j++){
                    vout.push_back(vin[j]);
                }
        //vout=vin;
    }
    //py::print("length after removing",vout.size());
    */
   

}

void get_fraction_derivative_coeffs(vector<long double> &c1, vector<long double> &c2, vector<long double> &derivativenum, vector<long double> &derivativeden){
    //Computes derivative of the fraction given by c1/c2 in terms of the coefficients. Returns the coefficients of the numerator and the denominator of the derivative.
    //c1 contains coefficients of numerator, c2 contains the coefficients of the denominator. Both c1 and c2 should have 0 in the case of 0 coefficients before the coefficient with last degree. E.g. 0,0,x^2,0,x^3 but not 0,0,x^2,0,x^3,0."""
    
    std::vector<long double>::size_type nnum, nden, degreenum, degreeden, degreesum, i, j;
    nnum=c1.size();
    nden=c2.size();
    degreenum=nnum-1;
    degreeden=nden-1;
    degreesum=degreenum+degreeden;

    //printf("size nnum is %lu, nden is %lu\n",nnum,nden);
    //printf("size der vector is %lu, dder vector is %lu\n",derivativenum.size(),derivativeden.size());
    

    //compute coefficients of derivative of the numerator and denominator polynomials
    vector<long double> dc1(degreenum);
    vector<long double> dc2(degreeden);
    
    //start at 1 so that derivative of term of degree 0 disappears
    for (i=1;i<nnum;i++){
        dc1[i-1]=i*c1[i];
        //printf("dc1 %u, %Le , %Le",i, c1[i], dc1[i-1]);     
    }
    for (i=1;i<nden;i++){
        dc2[i-1]=i*c2[i];
    }

    
    vector<long double> dc1_dot_c2(degreesum); //derivative of numerator is degreenum-1, and denominator is degreeden. Need to add one for the degree 0. 
    product_coeffs_vectors(dc1,c2,dc1_dot_c2); //derivative of numerator * denominator
    vector<long double> dc2_dot_c1(degreesum);
    product_coeffs_vectors(dc2,c1,dc2_dot_c1); //derivative of denominator * numerator
    //vector<long double> densq(degreesum+1,0); //denominator^2;
    product_coeffs_vectors(c2,c2,derivativeden);

    /*
    py::print("dc1dot");

    for(i=0;i<dc1_dot_c2.size();i++){
        py::print(dc1_dot_c2[i]);
        }
    for(i=0;i<dc2_dot_c1.size();i++){
        py::print(dc2_dot_c1[i]);
        }
    */
    //vector<double> dernum;
    for (i=0;i<degreesum;i++){     
        derivativenum[i]=(dc1_dot_c2[i]-dc2_dot_c1[i]);
    }

}

vector<double> compute_pos_stp(vector<long double> &num, vector<long double> &den, string rootmethod, bool verbose=false, double halfmax=0.5, bool writex05coeffs= false, bool absderivative=false, string fnamecoeffs="filename.txt"){
    if (verbose){
    cout << "Function executed with:" << "rootmethod: " << rootmethod << "\n, verbose:" << verbose << "\n halfmax:" << halfmax << "\n writex05coeffs: " << writex05coeffs << "\n absderivative: " << absderivative;
    cout << "Printing halfmax\n";
    cout<< halfmax;
    cout << "\n";
    cout.flush();
    }
    std::vector<long double>::size_type i, j, nnum, nden;
    long double i1;
    long double i2;
    vector<long double> x05v;
    long double x05;

    

    nnum=num.size();
    nden=den.size(); 
    
   
    vector<long double> coeffsx05(max(nnum,nden)); 
    for (i=0;i<max(nnum,nden);i++){
        if (i<nnum){
            i1=num[i];
        }else{
            i1=0;
        }
        if (i<nden){
            i2=halfmax*den[i];
        }else{
            i2=0;
        }

        coeffsx05[i]=i1-i2;
    }
    if (verbose){
    cout << "Printing GRF num\n";
    for (i=0;i<max(nnum,nden);i++){
        //cout << num[i];
        cout << lexical_cast<string>(num[i]);
        cout << "\n";
        cout.flush();
    }
    }

    



     //Find roots of coeffsx05, and keep the ones that are positive and real. Throw an error if this happens more than once. 
    
    
    
    if (rootmethod=="simple"){
        get_positive_real_roots(coeffsx05, x05v);
    }else if (rootmethod=="aberth"){
        get_positive_real_roots_aberth(coeffsx05, x05v);
    }
  
    //py::print("roots x05");
    vector<double> result = {-1.0, -1.0, -1.0}; //adding x05 as well
    //double maxder=-1;
    //double xmaxder=-1;


    if (x05v.size()==0){ //if there are two points of crossing the half max due to nonmonotonicity, take the smallest
        if (verbose){
        cout<<"x05 could not be computed.\n";
        }
        if (writex05coeffs){
            ofstream MyFile(fnamecoeffs);
            for (i=0;i<coeffsx05.size();i++){
            //cout << num[i];
            MyFile << lexical_cast<string>(coeffsx05[i]) << ",";
            
            }
            MyFile << "-1" << "\n";
            MyFile.close();
        }
  
        return result; //unsuccessful result

    }else if(x05v.size()==1){
        x05=x05v[0];
    }else{
       
        //i=distance(x05v.begin(),min_element(x05v.begin(),x05v.end()));
        //x05=x05v[i];
        x05=*std::min_element(x05v.begin(),x05v.end());
        
    }
    result[2]=x05;

    if (verbose){
        printf("x05: %Le\n",x05);
        }
    if (writex05coeffs){
        ofstream MyFile(fnamecoeffs);
        for (i=0;i<coeffsx05.size();i++){
        //cout << num[i];
        MyFile << lexical_cast<string>(coeffsx05[i]) << ",";
        
        }
        MyFile << x05 << "\n";
        MyFile.close();
    }

    //py::print("x05 is ", x05);
    //printf("x05: %Le\n",x05);
    

    double x05double = (double) x05;
    
    
    //if x05:

        //Normalise coefficients of num and den so that hopefully numbers become more reasonable
        
        
    for (i=0;i<nnum;i++){
        num[i]=num[i]*pow(x05,(int) i);
        //py::print("num",num[i]);

        //printf("term %d, power x05 %Le, num[i] %Le\n",i, pow(x05,i), num[i]);
    }
    

    for (i=0;i<nden;i++){
        den[i]=den[i]*pow(x05,(int) i);
        //py::print("den",den[i]);
    }
    
    
    
    long double xp, num_sum, den_sum;
   
    std::vector<long double>::size_type Niter;

    //derivative of GRF. It is a fraction: derivativenum/derivativeden
    //GRF numerator has degree nnum-1 and denominator has degree nden-1. So den*d(num)/dx -num*d(den)/dx is of degree nden-1+nnum-1-1. 
    // So the array has to be of size nnum+nden-3+1 to account for 0.
    //GRF denominator has degree nden-1. So squared of this is degree nden-1+nden-1. 

    vector<long double> derivativenum(nnum+nden-2); //(nsize)
    vector<long double> derivativeden(nden+nden-1);

    get_fraction_derivative_coeffs(num,den,derivativenum,derivativeden);
    //py::print("derivative coeffs");
    remove_zeros_endvector(derivativenum); //,derivativenum);
    remove_zeros_endvector(derivativeden); //_,derivativeden);
    //py::print("removed zeros");
    
    nnum=derivativenum.size();
    nden=derivativeden.size();

    if ((nnum==0)||(nden==0)){
        return result; //unsuccessful
    }



    vector<long double> derivative2num(nnum+nden-2);
    vector<long double> derivative2den(nden+nden-1);
    //cout << "going for second";

    

    //second derivative
    //py::print("going for second");
    get_fraction_derivative_coeffs(derivativenum,derivativeden,derivative2num,derivative2den);
    
    //cout << "done second\n";
    remove_zeros_endvector(derivative2num);
    remove_zeros_endvector(derivative2den);

   
    
    if ((derivative2num.size()<2)||(derivative2den.size()==0)){
        if (verbose){
        cout << "unsuccessful derivative2, sizes are: ";
        printf("num %d, den %d", derivative2num.size(),derivative2den.size());
        }
        return result; //unsuccessful
    }

     if (verbose){
    cout << "Printing d2 num\n";
    for (i=0;i<derivative2num.size();i++){
        cout << i << "," << derivative2num[i];
        cout << "\n";
        cout.flush();
    }

    cout << "Printing d2 den\n";
    for (i=0;i<derivative2den.size();i++){
        cout << i << "," << derivative2den[i];
        cout << "\n";
        cout.flush();
    }
    }
    
    //After discussing with Felix on April 2020, we decided that if there was a max on the negative side it wouldn't matter, so discarding this.
    /*long double secondderx0;
    if (derivative2den[0]>1e-15){
            secondderx0=derivative2num[0]/derivative2den[0];
        if ((secondderx0)<0){
            if (verbose){
              cout << "max derivative at 0. Clarifying: \n ";
              cout << derivative2num[0] << "," << derivative2den[0] << "\n" << endl;
              cout<<secondderx0;
              cout<<"\n";
              cout<<derivative2num[0];
              cout<<"\n";
              cout<<derivative2den[0];
              cout<<"\n";
            }
            return result; //,ax derivative is in fact at 0
        }
    }else{
        return result; // if I cannot check the second derivative at 0, then do not return any result
    }*/



    //py::print("second");
    
    
    vector<long double> critpoints;
    if (rootmethod=="simple"){
        get_positive_real_roots(derivative2num,critpoints);
    }else if (rootmethod=="aberth"){
        get_positive_real_roots_aberth(derivative2num,critpoints);
    }

        //py::print("criticalpoints");
    if (verbose){
    cout << "critpoints for derivative2num\n ";
    if (critpoints.size()==0){
        cout << "No criticalpoints found. The coefficients were \n";
        for (i=0;i<derivative2num.size();i++){
            cout << derivative2num[i];
            cout << "\n";
        }
        cout << "end coeffs derivative2num\n";
    }else{
        for(i=0;i<critpoints.size();i++){
            printf("%Le,", critpoints[i]);
        }
        cout << "\n";
        }
    }


    //if (critpoints.size()==0){
    //    return result;
    //}else{
         //py::print("no critical points found");
    //    cout << "no critical points found\n";

    if (critpoints.size()>0){
        long double mincritpoint;
        long double thirdderx, thirdderx0;
        nnum=derivative2num.size();
        nden=derivative2den.size();
        vector <long double> maxderv;
        vector <long double> xmaxderv;
        vector<long double> derivative3num(nnum+nden-2);
        vector<long double> derivative3den(nden+nden-1);

        if (absderivative){
            nnum=derivativenum.size();
            nden=derivativeden.size();
            for (j=0;j<critpoints.size();j++){
                num_sum=0;
                den_sum=0;
                
                for (i=0;i<nnum;i++){
                    xp=pow(critpoints[j],(int)i);
                    num_sum+=derivativenum[i]*xp;
                }
                for (i=0;i<nden;i++){
                    xp=pow(critpoints[j],(int)i);
                    den_sum+=derivativeden[i]*xp;
                    //py::print("Adding to num",derivative3num[i],xp,num_sum);
                    //py::print("Adding to den",derivative3den[i],xp,den_sum);
                }
                if  (verbose){
                    cout << "derivative value at a critical point " << critpoints[j] <<":"<< num_sum/den_sum <<","<<abs(num_sum/den_sum);
                }


                maxderv.push_back(abs(num_sum/den_sum));
                xmaxderv.push_back(critpoints[j]);
            }


        }else{


        
        //py::print("original size", nnum+nden-2, derivative3num.size());



        get_fraction_derivative_coeffs(derivative2num,derivative2den,derivative3num,derivative3den);

        remove_zeros_endvector(derivative3num);
        remove_zeros_endvector(derivative3den);
        //py::print(derivative2num.size(), derivative2den.size());
        //py::print("derivative3", derivative3num.size());
        
        if ((derivative3num.size()>0)&&(derivative3den.size()>0)){
            if (verbose){
                cout << "Looking at derivative 3\n ";
            }

                       
            for (j=0;j<critpoints.size();j++){
                num_sum=0.0;
                den_sum=0.0;
                nnum=derivative3num.size();
                nden=derivative3den.size();
                for (i=0;i<nnum;i++){
                    xp=pow(critpoints[j],i);
                    num_sum+=derivative3num[i]*xp;
                }
                for (i=0;i<nden;i++){
                    xp=pow(critpoints[j],i);
                    den_sum+=derivative3den[i]*xp;
                    //py::print("Adding to num",derivative3num[i],xp,num_sum);
                    //py::print("Adding to den",derivative3den[i],xp,den_sum);
                }
                thirdderx=num_sum/den_sum;
                if (verbose){
                    cout << "For critpoint" << critpoints[j] << "value is" << thirdderx;
                }
                //py::print("For point",critpoints[j],num_sum,den_sum,thirdderx);
                if (thirdderx<0){ //local maximum

                    num_sum=0;
                    den_sum=0;
                    nnum=derivativenum.size();
                    nden=derivativeden.size();
                    for (i=0;i<nnum;i++){
                        xp=pow(critpoints[j],(int)i);
                        num_sum+=derivativenum[i]*xp;
                    }
                    for (i=0;i<nden;i++){
                        xp=pow(critpoints[j],(int)i);
                        den_sum+=derivativeden[i]*xp;
                        //py::print("Adding to num",derivative3num[i],xp,num_sum);
                        //py::print("Adding to den",derivative3den[i],xp,den_sum);
                    }


                    maxderv.push_back(num_sum/den_sum);
                    xmaxderv.push_back(critpoints[j]);
                }
            }
            }//derivative3numsize
        }//absderivative
            


        Niter=maxderv.size();
        if (Niter>0){
            i=distance(maxderv.begin(),max_element(maxderv.begin(),maxderv.end()));
            
            //need to check that max derivative is greater than the derivative at 0
            double derivative_at_0=derivativenum[0]/derivativeden[0];
            if (absderivative){
                derivative_at_0=abs(derivative_at_0);
            }
            if (derivative_at_0<maxderv[i]){
                result[1]=maxderv[i]; //*x05;
                result[0]=xmaxderv[i]; ///x05;
            }else{
                if (verbose){
                cout<<"Max derivative occurs at 0!" << derivative_at_0 << "\n";
                }
            }
        }

                
                //py::print("maxder",maxder);
                //py::print("xmaxder",xmaxder);
                //result[0]=xmaxder;
                //result[1]=maxder;


            //else{
                //py::print("Could not find max derivative despite finding critical points.");
                //cout << "Could not find max derivative despite finding critical points.";
            //}
        
        
    }//critpoints

    //x05

    
    //printf("at the end: %g, %g\n", xmaxder, maxder);
    //vector<double> result = {xmaxder, maxder};
    if (verbose){
        cout << "returning\n";
        cout.flush();
    }
    return result;
}

vector <double> compute_monotonic(vector<long double> &num, vector<long double> &den){


    std::vector<long double>::size_type i, j, nnum, nden;
    long double i1;
    long double i2;

    nnum=num.size();
    nden=den.size(); 
    vector<double> result;


    //derivative of GRF. It is a fraction: derivativenum/derivativeden
    //GRF numerator has degree nnum-1 and denominator has degree nden-1. So den*d(num)/dx -num*d(den)/dx is of degree nden-1+nnum-1-1. 
    // So the array has to be of size nnum+nden-3+1 to account for 0.
    //GRF denominator has degree nden-1. So squared of this is degree nden-1+nden-1. 

    vector<long double> derivativenum(nnum+nden-2); //(nsize)
    vector<long double> derivativeden(nden+nden-1);

    get_fraction_derivative_coeffs(num,den,derivativenum,derivativeden);
    remove_zeros_endvector(derivativenum); //,derivativenum);
    remove_zeros_endvector(derivativeden); //_,derivativeden);

    if ((derivativenum.size()<2)||(derivativeden.size()==0)){
        result = {-1.0};
        return result;
    }

    //critical points of derivative correspond to roots of derivativenum
    vector<long double> critpoints;
    //get_positive_real_roots(derivative2num,critpoints); //critical points are derivative2=0 so numerator of derivative2=0;
    get_positive_real_roots_aberth(derivativenum,critpoints); //critical points are derivative2=0 so numerator of derivative2=0;

    if (critpoints.size()>0){
        for (i=0;i<critpoints.size();i++){
            result.push_back(critpoints[i]);
        }
    }else{
        result = {-2.0};
        
    }
    return result;
}
