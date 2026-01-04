
#include <pybind11/pybind11.h>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <boost/random.hpp>



using namespace std;
using boost::lexical_cast;
using std::string;


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

// With respect to the original version, this accepts any type, including boost::multiprecision::number types with float/complex backends with user-defined precision.  
//T: type of GRF coefficients and operations with them
//polytype: type of polynomial class. 
//Tpolyc: type of complex number ot collect polynomial roots.
//Tpolyc and polytype should be of same precision.

template <typename T>
T GRFatxonly(vector<T>&num, vector<T>&den, T varGRFval){ 

    
    T numsum=0;
    T densum=0;
    typename std::vector<T>::size_type i;
    for (i=0;i<num.size();i++){
        numsum+=num[i]*pow(varGRFval,(int)i);
    }
    for (i=0;i<den.size();i++){
        densum+=den[i]*pow(varGRFval,(int)i);
    }
    T result=numsum/densum;
    return result;
    
}

template <typename T>
void compute_min_maxGRF(vector<T> num, vector<T> den, vector<T> &min_max){

    
    T Gmax=-1;
    T Gmin=1e20;

    
    int Nmax=6000;
    double xvalmin=-60;
    double xvalmax=60;
    double step=(xvalmax-xvalmin)/Nmax;
    T xval;
    T xval10;
    T GRFval;
    //double ymin,ymax;
    
        
        
    //compute min and max numerically
    //I need to identify the min before the max. Otherwise it can be that the min is after the max and I think this is not what we want
        
    xval=xvalmin;
    vector <T> GRFvalsvec(Nmax);
    
    for (int i=0;i<Nmax;i++){
        xval10=pow(10,xval);
        //xvecmax[i]=xval10;
        GRFval=GRFatxonly(num,den,xval10);
        GRFvalsvec[i]=GRFval;
        xval+=step;
        if (GRFval>Gmax){
            Gmax=GRFval;
        }
        if (GRFval<Gmin){
            Gmin=GRFval;
        }
    }

    /*//now go over the function values, and pick the minimum as long as Gmax is not reached
    //find where the max occurs
    int idx=distance(GRFvalsvec.begin(),max_element(GRFvalsvec.begin(),GRFvalsvec.end()));
    for (int i=0;i<idx;i++){
        GRFval=GRFvalsvec[i];
        if (GRFval<Gmin){
           Gmin=GRFval;
        }
    }*/

    min_max[0]=Gmin;
    min_max[1]=Gmax;
}

//template with vectors. See https://stackoverflow.com/questions/19094340/stdvector-as-a-template-function-argument
template <typename T>
vector <T> trim_zerocoeff(vector<T> coeffs){
	typename vector<T>::const_iterator first = coeffs.begin() + 1;
    typename vector<T>::const_iterator last = coeffs.end();
    vector<T> newVec(first, last);
    return newVec;
}


template <typename T, typename Tpolyc, typename polytype, typename thresholdtype>
void get_positive_real_roots_aberth(vector<T> coeffsx05, vector<T> &pos_real, thresholdtype threshold){
    /*coefficients are of type T 
    Tpolyc: typedef number<mpc_complex_backend<precision_poly> > mympc_poly;
    polytype: typedef Polynomial<precision_poly> polytype;
    threshold: threshold to decide if imaginary part of root is zero. With its own type.
    */
    //cout << "solving with boost\n";
    unsigned max_iter = 10000; 
    double tol = std::numeric_limits<double>::epsilon(); 
    boost::random::mt19937 rng(1234567890);
    boost::random::uniform_real_distribution<> dist(0, 1);
    T threshold2 = std::numeric_limits<T>::epsilon(); 
    //number<mpfr_float_backend<20> > threshold = std::numeric_limits<number<mpfr_float_backend<20> >>::epsilon(); 
    //Letting this be another type and a user-defined parameter as in some cases it affects the results
    

    //If the constant coefficient of the polynomial is 0, 
    //e.g. 0+bx+ax^2=0
    //then it is equivalent to x(b+ax)=0
    //so it is equivalent to finding the roots of a polynomial of one degree less
    //the initialization method only works when constant coefficient is not zero, so update
    //cout << "trimming\n";
    //cout.flush();
    while ((abs(coeffsx05[0])< threshold2)&&(coeffsx05.size()>1)){
        coeffsx05=trim_zerocoeff<T>(coeffsx05);
    }
    //cout << "trimmed\n";
    //cout.flush();
    if (coeffsx05.size()>1){
        
        polytype p(coeffsx05);
        //cout << "going to compute roots\n";
        //cout.flush();
        std::pair<std::vector<Tpolyc>, bool> roots_computed = p.roots(max_iter,tol,tol,rng,dist);
        if (roots_computed.second){ //converged
        //cout << "roots computed ";
        //cout.flush();
        for (unsigned i=0;i<roots_computed.first.size();i++){
            //cout << roots_computed.first[i];
            //cout <<"\n";
            //cout.flush();
            if (abs(roots_computed.first[i].imag()) < threshold){ //this threshold shouldn't be too small otherwise we miss relevant roots
                //cout << "imaginary part of root under threshold" << roots_computed.first[i];
                //py::print("root",i,roots[i]);
                
                if (roots_computed.first[i].real()>0){
                    pos_real.push_back(roots_computed.first[i].real().template convert_to<T>());

                }
            }//else{
             //   cout << "imaginary part of root is not under threshold" << roots_computed.first[i];
            //}
        }
        }
    }
}


template <typename T>
void product_coeffs_vectors(vector<T> &c1, vector<T>  &c2, vector<T> &result){
    //vector<double> result(degree+degree);
    typename vector<T>::size_type idx, i, j, n1, n2;
    n1=c1.size();
    n2=c2.size();
    for ( i=0;i<n1;i++){
        for ( j=0;j<n2;j++){
            idx=i+j;      
            result[idx]+=c1[i]*c2[j];
        }
    }
} 

template <typename T>
void remove_zeros_endvector(vector<T> &v){ //, vector<double> &vout){
    //py::print("removing zeros", v.size());
    //cout << "removing zeros";

    T threshold = std::numeric_limits<T>::epsilon(); 

    while ((abs(v.back())<threshold)&(v.size()>0)){
        //py::print(v.back());
        v.pop_back();
    }

}

template <typename T>
void get_fraction_derivative_coeffs(vector<T> &c1, vector<T> &c2, vector<T> &derivativenum, vector<T> &derivativeden){
    //Computes derivative of the fraction given by c1/c2 in terms of the coefficients. Returns the coefficients of the numerator and the denominator of the derivative.
    //c1 contains coefficients of numerator, c2 contains the coefficients of the denominator. Both c1 and c2 should have 0 in the case of 0 coefficients before the coefficient with last degree. E.g. 0,0,x^2,0,x^3 but not 0,0,x^2,0,x^3,0."""
    
    typename vector<T>::size_type nnum, nden, degreenum, degreeden, degreesum, i, j;
    nnum=c1.size();
    nden=c2.size();
    degreenum=nnum-1;
    degreeden=nden-1;
    degreesum=degreenum+degreeden;

    //printf("size nnum is %lu, nden is %lu\n",nnum,nden);
    //printf("size der vector is %lu, dder vector is %lu\n",derivativenum.size(),derivativeden.size());
    

    //compute coefficients of derivative of the numerator and denominator polynomials
    vector<T> dc1(degreenum);
    vector<T> dc2(degreeden);
    
    //start at 1 so that derivative of term of degree 0 disappears
    for (i=1;i<nnum;i++){
        dc1[i-1]=i*c1[i];
        //printf("dc1 %u, %Le , %Le",i, c1[i], dc1[i-1]);     
    }
    for (i=1;i<nden;i++){
        dc2[i-1]=i*c2[i];
    }

    
    vector<T> dc1_dot_c2(degreesum); //derivative of numerator is degreenum-1, and denominator is degreeden. Need to add one for the degree 0. 
    product_coeffs_vectors<T>(dc1,c2,dc1_dot_c2); //derivative of numerator * denominator
    vector<T> dc2_dot_c1(degreesum);
    product_coeffs_vectors<T>(dc2,c1,dc2_dot_c1); //derivative of denominator * numerator
    //vector<long double> densq(degreesum+1,0); //denominator^2;
    product_coeffs_vectors<T>(c2,c2,derivativeden);

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

template<typename T, typename Tpolyc, typename polytype, typename thresholdtype>
vector<T> compute_pos_stp(vector<T> &num, vector<T> &den, string rootmethod, bool verbose, T halfmax, thresholdtype thresholdimag, bool writex05coeffs, bool absderivative, bool normalisefirst, string fnamecoeffs){
    //thresholdimag: threshold under which imaginary part of root is considered to be 0
    if (rootmethod != "aberth"){
        cout << "rootmethod should be aberth. Exiting.";
        return {-1., -1., -1.};
    }
    if (verbose){
    cout << "Function executed with:" << "rootmethod: " << rootmethod << "\n, verbose:" << verbose << "\n halfmax:" << halfmax << "\n thresholdimag: " << thresholdimag << "\n writex05coeffs: " << writex05coeffs << "\n absderivative: " << absderivative << "\n normalisefirst: " <<  normalisefirst << "\n fnamecoeffs: "<< fnamecoeffs <<"\n";
    cout.flush();
    }

    typename vector<T>::size_type i, j, nnum, nden;
    T i1;
    T i2;
    vector<T> x05v;
    T x05;

    nnum=num.size();
    nden=den.size(); 
       
    vector<T> coeffsx05(max(nnum,nden)); 
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
    cout.precision(50);
    for (i=0;i<max(nnum,nden);i++){
        //cout << num[i];
        cout << num[i];
        cout << "\n";
        cout.flush();
    }
    }


     //Find roots of coeffsx05, and keep the ones that are positive and real. Throw an error if this happens more than once. 
    
    if (rootmethod=="simple"){
        ; //not implemented in this case. Before this led to execute get_positive_real_roots(coeffsx05, x05v);
    }else if (rootmethod=="aberth"){
        get_positive_real_roots_aberth<T,Tpolyc,polytype,thresholdtype>(coeffsx05, x05v,thresholdimag);
    }
  
    //py::print("roots x05");
    vector<T> result = {-1.0, -1.0, -1.0}; //adding x05 as well
    //double maxder=-1;
    //double xmaxder=-1;


    if (x05v.size()==0){ 
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
        //if there are two points of crossing the half max due to nonmonotonicity, take the smallest
       
        //i=distance(x05v.begin(),min_element(x05v.begin(),x05v.end()));
        //x05=x05v[i];
        x05=*std::min_element(x05v.begin(),x05v.end());
        
    }
    
    result[2]=x05;

    if (verbose){
        cout << "x05: "<< x05 << "\n";
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
        

    //Normalise coefficients of num and den so that hopefully numbers become more reasonable
    // as of october 20201, I am removing this. It seems this causes problems when the coefficients get too big. Then pos and stp need to be normalised at the end.

    if (normalisefirst){   
        for (i=0;i<nnum;i++){
            num[i]=num[i]*pow(x05,(int) i);
            //py::print("num",num[i]);
            //printf("term %d, power x05 %Le, num[i] %Le\n",i, pow(x05,i), num[i]);
        }
        
        for (i=0;i<nden;i++){
            den[i]=den[i]*pow(x05,(int) i);
            //py::print("den",den[i]);
        }
    }
    
    T xp, num_sum, den_sum;
   
    typename vector<T>::size_type Niter;

    //derivative of GRF. It is a fraction: derivativenum/derivativeden
    //GRF numerator has degree nnum-1 and denominator has degree nden-1. So den*d(num)/dx -num*d(den)/dx is of degree nden-1+nnum-1-1. 
    //So the array has to be of size nnum+nden-3+1 to account for 0.
    //GRF denominator has degree nden-1. So squared of this is degree nden-1+nden-1. 

    vector<T> derivativenum(nnum+nden-2); //(nsize)
    vector<T> derivativeden(nden+nden-1);

    get_fraction_derivative_coeffs<T>(num,den,derivativenum,derivativeden);
    //py::print("derivative coeffs");
    remove_zeros_endvector<T>(derivativenum); //,derivativenum);
    remove_zeros_endvector<T>(derivativeden); //_,derivativeden);
    //py::print("removed zeros");
    
    nnum=derivativenum.size();
    nden=derivativeden.size();

    if ((nnum==0)||(nden==0)){
        return result; //unsuccessful
    }


    vector<T> derivative2num(nnum+nden-2);
    vector<T> derivative2den(nden+nden-1);
    //cout << "going for second";
    //second derivative
    //py::print("going for second");
    get_fraction_derivative_coeffs<T>(derivativenum,derivativeden,derivative2num,derivative2den);
    //cout << "done second\n";
    remove_zeros_endvector<T>(derivative2num);
    remove_zeros_endvector<T>(derivative2den);
    
    if ((derivative2num.size()<2)||(derivative2den.size()==0)){
        if (verbose){
            cout << "unsuccessful derivative2, sizes are: ";
            cout << derivative2num.size() << "\n";
            cout << derivative2den.size() << "\n";
            //fix this printf("num %d, den %d", derivative2num.size(),derivative2den.size());
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
    
    
    vector<T> critpoints;
    if (rootmethod=="simple"){
        ; //get_positive_real_roots(derivative2num,critpoints);
    }else if (rootmethod=="aberth"){
        get_positive_real_roots_aberth<T,Tpolyc,polytype,thresholdtype>(derivative2num,critpoints,thresholdimag);
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
                cout << critpoints[i] << ",";
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
        T mincritpoint;
        T thirdderx, thirdderx0;
        nnum=derivative2num.size();
        nden=derivative2den.size();
        vector <T> maxderv;
        vector <T> xmaxderv;
        vector<T> derivative3num(nnum+nden-2);
        vector<T> derivative3den(nden+nden-1);

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
        get_fraction_derivative_coeffs<T>(derivative2num,derivative2den,derivative3num,derivative3den);

        remove_zeros_endvector<T>(derivative3num);
        remove_zeros_endvector<T>(derivative3den);
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
            T derivative_at_0=derivativenum[0]/derivativeden[0];
            if (absderivative){
                derivative_at_0=abs(derivative_at_0);
            }
            if (derivative_at_0<maxderv[i]){
                if (normalisefirst){
                        result[1]=maxderv[i];
                        result[0]=xmaxderv[i];
                }else{
                        result[1]=(maxderv[i]*x05);
                        result[0]=(xmaxderv[i]/x05);
                }
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

template <typename T, typename Tpolyc, typename polytype, typename thresholdtype>
vector <T> compute_monotonic(vector<T> &num, vector<T> &den, thresholdtype thresholdimag){

    
    typename vector<T>::size_type i, j, nnum, nden;
    long double i1;
    long double i2;

    nnum=num.size();
    nden=den.size(); 
    vector<T> result;


    //derivative of GRF. It is a fraction: derivativenum/derivativeden
    //GRF numerator has degree nnum-1 and denominator has degree nden-1. So den*d(num)/dx -num*d(den)/dx is of degree nden-1+nnum-1-1. 
    // So the array has to be of size nnum+nden-3+1 to account for 0.
    //GRF denominator has degree nden-1. So squared of this is degree nden-1+nden-1. 

    vector<T> derivativenum(nnum+nden-2); //(nsize)
    vector<T> derivativeden(nden+nden-1);

    get_fraction_derivative_coeffs<T>(num,den,derivativenum,derivativeden);
    remove_zeros_endvector<T>(derivativenum); //,derivativenum);
    remove_zeros_endvector<T>(derivativeden); //_,derivativeden);

    if ((derivativenum.size()<2)||(derivativeden.size()==0)){
        result = {-1.0};
        return result;
    }

    //critical points of derivative correspond to roots of derivativenum
    vector<T> critpoints;
    //get_positive_real_roots(derivative2num,critpoints); //critical points are derivative2=0 so numerator of derivative2=0;
    get_positive_real_roots_aberth<T,Tpolyc,polytype,thresholdtype>(derivativenum,critpoints,thresholdimag); //critical points are derivative2=0 so numerator of derivative2=0;

    if (critpoints.size()>0){
        for (i=0;i<critpoints.size();i++){
            result.push_back(critpoints[i]);
        }
    }else{
        result = {-2.0};
    }
    return result;
}
