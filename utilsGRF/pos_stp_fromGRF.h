#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <Eigen/SparseCore>
#include "solvenullspace.h"

/*
 Code to compute position and steepness for a GRF, and function value at a point, from numerical values.
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

using namespace std;
using namespace Eigen;

typedef double T;
typedef Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;
typedef Eigen::SparseMatrix<T> SM;

typedef long double T2;
T2 GRF(vector<T2>&num, vector<T2>&den, double x){
	T2 num_sum=0;
	T2 den_sum=0;
	int nnum=num.size();
	int nden=den.size();
	int i;
    if (nnum==nden){
        double xp;
        for (i=0;i<nnum;i++){
            xp=pow(x,i);
            num_sum+=num[i]*xp;
            den_sum+=den[i]*xp;
        }

    }else{
        for (i=0;i<nnum;i++){
            num_sum+=num[i]*pow(x,i);
        }
	 
	    for (i=0;i<nden;i++){
		    den_sum+=den[i]*pow(x,i);
	    }
    }

	return num_sum/den_sum;

}


vector<double> compute_pos_stp_fromGRF(vector<T2>&num, vector<T2>&den, bool verbose=false, int npoints=1000){
    //this function uses the numerator and denominator of the ss, obtained from MTT
    int Nmax=70;
    vector<double> xvecmax_(Nmax);
    //vector<double> xvecmax(Nmax); //10^xvecmax_
    vector<T2> outmax(Nmax);
    double xval=-15;
    double xval10;
    int i;
    vector<double>result={-1.0,-1.0};
    T maxGRF=GRF(num,den,pow(10,20));
    T GRF01=maxGRF*0.02;
    T GRF99=maxGRF*0.98;
    T GRF95=maxGRF*0.95;
    T GRF05=maxGRF*0.5;
    double x0=-15;
    double x1=-15;
    //double x5val=-15;
    T2 GRFval;

    //First compute GRF at points from 10^-15 to 10^20 (70 points total) to get the range of the GRF
    for (i=0;i<Nmax;i++){
    	xval10=pow(10,xval);
    	xvecmax_[i]=xval;
    	//xvecmax[i]=xval10;
    	GRFval=GRF(num,den,xval10);
    	xval+=0.5;

        if ((GRFval>=GRF01)&(x0<-14)){
            x0=xvecmax_[i]-1;
        }
        if ((GRFval>=GRF99)&(x1<-14)){
            x1=xvecmax_[i]+1;
            break;
        }

    }

    if ((x0<-14) | (x1<-14)){
    	return result;
    }else{
    	double dx;
        int norders=(int)(x1-x0+0.5);
        int N=npoints*norders; //1000 points per order magnitude
        dx=(x1-x0)/N;

    	//vector<double>xvec_(N);
    	vector<double>xvec(N);
    	vector<double>out(N);
    	double x=x0;
    	for (i=0;i<N;i++){
    		//xvec_[i]=x;
    		xval10=pow(10,x);
    		xvec[i]=xval10;
    		out[i]=GRF(num,den,xval10);
    		x+=dx;
    	}

    	double halfX=-1;
    	T2 der=0;
    	T2 maxder=0;
    	double xmaxder=0;

    	for (i=0;i<N-1;i++){
    		der=(out[i+1]-out[i])/(xvec[i+1]-xvec[i]);
    		if (der>maxder){
    			maxder=der;
    		    xmaxder=xvec[i];
    		}else if(out[i]>GRF95){
                break;
            }
            if ((out[i]>=GRF05)&(halfX<0)){
                halfX=xvec[i];
            }
    	}
        //compute derivative at 0. Careful, this is likely to run into accuracy issues.
        T2 GRFat0=GRF(num,den,pow(10,-15));
        T2 GRFat0dx=GRF(num,den,pow(10,-13));
        T2 derat0=(GRFat0dx-GRFat0)/(0.99*pow(10,-13));

        if (derat0<maxder){

            result={(double)xmaxder/halfX,(double)maxder*halfX};
        }else{
            if (verbose){
            cout << "Max at 0!\n";
        }
        }
        return result;

    }

}

void insert_L_Lx_atval(SM& L, const std::vector<Eigen::Triplet<T>>& Lx, T xval){
    Eigen::Triplet<T> trd;
    for (int j=0;j<Lx.size();j++){
        trd=Lx[j];
        L.insert(trd.row(),trd.col())=trd.value()*xval;
    }
}


T GRFatX(const SM& L_, const std::vector<Eigen::Triplet<T>>& Lx, double xval, vector<int> &indices, vector<double> &coefficients, bool doublecheck=false){
    //Lx should contain a 1 at each position for which L has to be multiplied by xval
    
    int n=L_.rows(); //L is square so this is number of rows/columns
    int i, j;
    T ss=0;
    T cs;
    SM L;
    L=SM(L_);
    
    insert_L_Lx_atval(L,Lx,xval);
    //Eigen::SparseMatrix<T>
    

    //diagonals
    for (int k=0; k<L.outerSize(); ++k) {
         cs=0;
        for(typename Eigen::SparseMatrix<T>::InnerIterator it (L,k); it; ++it){
            cs+=it.value();
        }
        L.insert(k,k)=-cs;
    } 

    //cout << "After multiplying by x\n.";
    //cout << L;
    //cout << "\n";

    ss=ssfromnullspace(L,indices,coefficients,doublecheck);
    return ss;

}


//vector<double> compute_pos_stp_fromGRF(Ref<MatrixXd> L, vector<Eigen::Triplet<int>>& Lx, vector<int> &indices, vector <double> &coefficients, bool verbose=false, int npoints=1000, bool doublecheck=false){
vector<double> compute_pos_stp_fromGRF(const SM& L, const std::vector<Eigen::Triplet<T>>& Lx, vector<int> &indices, vector <double> &coefficients, bool verbose=false, int npoints=1000, bool doublecheck=false){

    //this function solves for the nullspace of the L to obtain the ss
    //L: Laplacian with only parameter values that do not depend on TF. This is fixed.
    //triplets with parameter values that depend on TF.
    //indices: indices of the nullspace vector to use for the ss computation
    //coefficients: coefficient that multiplies each of rho_i (with i in indices) to compute the ss. 
    int Nmax=70;
    vector<double> xvecmax_(Nmax);
    //vector<double> xvecmax(Nmax); //10^xvecmax_
    vector<T> outmax(Nmax);
    double xval=-15;
    double xval10;
    int i;
    vector<double>result={-1.0,-1.0};

    
    T maxGRF=GRFatX(L,Lx,pow(10,20),indices,coefficients);
    if (maxGRF<0){
        cout << "Failed when computing maxGRF\n";
        return result;
    }
    T GRF01=maxGRF*0.02;
    T GRF99=maxGRF*0.98;
    T GRF95=maxGRF*0.95;
    T GRF05=maxGRF*0.5;
    double x0=-15;
    double x1=-15;
    //double x5val=-15;
    T GRFval;

    //First compute GRF at points from 10^-15 to 10^20 (70 points total) to get the range of the GRF
    for (i=0;i<Nmax;i++){
        xval10=pow(10,xval);
        xvecmax_[i]=xval;
        //xvecmax[i]=xval10;
        GRFval=GRFatX(L,Lx,xval10,indices,coefficients,doublecheck);
        xval+=0.5;
        if (GRFval<0){
        cout << "Failed when computing GRFval, " << xval10 << "\n";
        return result;
        }

        if ((GRFval>=GRF01)&(x0<-14)){
            x0=xvecmax_[i]-1;
        }
        if ((GRFval>=GRF99)&(x1<-14)){
            x1=xvecmax_[i]+1;
            break;
        }

    }

    if ((x0<-14) | (x1<-14)){
        return result;
    }else{
        double dx;
        int norders=(int)(x1-x0+0.5);
        int N=npoints*norders; //1000 points per order magnitude
        dx=(x1-x0)/N;
        cout << "N is" << N;
        cout << "\n";

        //vector<double>xvec_(N);
        vector<double>xvec(N);
        vector<double>out(N);
        double x=x0;
        for (i=0;i<N;i++){
            //xvec_[i]=x;
            xval10=pow(10,x);
            xvec[i]=xval10;
            out[i]=GRFatX(L,Lx,xval10,indices,coefficients,doublecheck); // GRF(num,den,xval10);
            if (out[i]<0){
                cout << "Failed when computing out[i] at " << xval10 << "\n";
                return result;
            }
            x+=dx;
        }

        double halfX=-1;
        T der=0;
        T maxder=0;
        double xmaxder=0;

        for (i=0;i<N-1;i++){
            der=(out[i+1]-out[i])/(xvec[i+1]-xvec[i]);
            if (der>maxder){
                maxder=der;
                xmaxder=xvec[i];
            }else if(out[i]>GRF95){
                break;
            }
            if ((out[i]>=GRF05)&(halfX<0)){
                halfX=xvec[i];
            }
        }
        T GRFat0=GRFatX(L,Lx,pow(10,-15),indices,coefficients,doublecheck);
        T GRFat0dx=GRFatX(L,Lx,pow(10,-13),indices,coefficients,doublecheck);
        T derat0=(GRFat0dx-GRFat0)/(0.99*pow(10,-13));

        if (derat0<maxder){

            result={(double)xmaxder/halfX,(double)maxder*halfX};
        }else{
            if (verbose){
            cout << "Max at 0!\n";
        }
        }

        return result;

    }

}
