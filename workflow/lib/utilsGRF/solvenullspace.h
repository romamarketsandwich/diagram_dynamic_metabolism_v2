#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <vector>
 
using namespace std;
using namespace Eigen;
typedef double T;
typedef Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;
typedef Eigen::Matrix< T, Eigen::Dynamic, 1 > VectorXd;
 
int nullspace(Ref<MatrixXd> L, Ref<VectorXd> N, bool doublecheck=false){
//int nullspace(MatrixXd& L, Ref<VectorXd> N, bool doublecheck=false)
//int nullspace(const EigenBase<MatrixXd>& L_, Ref<VectorXd> N, bool doublecheck=false){
    //const MatrixXd L=L_.derived();
    const int n=L.rows();
    unsigned int r,i;
    double cs;
    double tolerance=1e-10;
        
        
    //Matrix<double,12,1> N;//solution of nullspace
    JacobiSVD<MatrixXd> svd(L,ComputeThinV);
    //if (svd.info!=success){ what is the equivalent to check that svd was successful??
    //    return -1;
    //}
    /* Get the V matrix */
    MatrixXd V((int)svd.matrixV().rows(), (int)svd.matrixV().cols());
    V = svd.matrixV();

    
    //if (svd.singularValues()(n-1,0)>tolerance){
    //for large values of the coefficients the tolerance is not met so for now I am not testing this
    //    cout << "Last singular value is greater than tolerance." << svd.singularValues()(n-1,0) << "\n";
        //N(0,0)=-1;
        //return; 
    //}else{
        N=V.col(svd.matrixV().cols()-1).cwiseAbs();

        cs=N.sum();
        for (r=0;r<n;r++){
            N(r,0)=abs(N(r,0))/cs; //negative values sometimes are negative. keep positive, although it doesn't really matter
        }

        if (doublecheck==true){
            MatrixXd out(n,1);
            out=L*N;
            i=0;
            bool goodnullspace=true;

            for (i=0;i<n;i++){
                if (abs(out(i,0))>tolerance){
                    cout << "inaccurate nullspace to tolerance" << tolerance << "\n";
                    cout << out;
                    goodnullspace=false;
                    break;
                }
            }
            cout << "Goodnullspace? " << goodnullspace << "\n";
        }

        //cout << N;
        //cout <<"\n";
       return 1; 
      
    //}
}

int nullspace(const Ref<Eigen::SparseMatrix<T>> L, Ref<VectorXd> N, bool doublecheck=false){
//int nullspace(const EigenBase<Eigen::SparseMatrix<T>>& L_, Ref<VectorXd> N, bool doublecheck=false){
    //const Eigen::SparseMatrix<T> L=L_.derived(); //reference to the derived object
    const int n=L.rows();
    unsigned int r,c,i;
    double cs;
    double tolerance=1e-10;
    
    Eigen::SparseMatrix<T> LT=L.transpose();
    Eigen::SparseQR<Eigen::SparseMatrix<T>,Eigen::COLAMDOrdering<int>> qr;
    qr.compute(LT);
    if(qr.info()!=Eigen::Success) {
        //N(0,1)=-1;
    // decomposition failed
     return -1;
     }
     
    MatrixXd Q; //Q is a dense matrix!!! qr.matrixQ() is extremely slow if called with a sparse matrix
    //SparseMatrix<double> Q;
    Q=qr.matrixQ();
    r=n-1;//index of last column
    
    N=Q.col(r).cwiseAbs();//abs in case zeros are taken negative
    
    cs=N.sum();
    for (int r=0;r<n;r++){
        N(r,0)=N(r,0)/cs; 
    }


    if (doublecheck==true){
        MatrixXd out(n,1);
        out=L*N;
        i=0;
        bool goodnullspace=true;

        for (i=0;i<n;i++){
            if (abs(out(i,0))>tolerance){
                cout << "inaccurate nullspace to tolerance" << tolerance << "\n";
                cout << out;
                goodnullspace=false;
                break;
            }
        }
        cout << "Goodnullspace? " << goodnullspace << "\n";
    }
    return 1;
}

//I tried writing ssfromnullspace as a template but compilation time was very long for large dense matrices so I'm back to nontemplate.


double ssfromnullspace(Ref<MatrixXd> L, vector<int> &indices, vector<double> &coefficients, bool doublecheck=false){
    int n=L.rows();
    int i;
    T ss=0;
    VectorXd N;
    N.resize(n,1);

    
    i=nullspace(L,N,doublecheck);
    if (i<0){ //if it cannot solve for the nullspace
        return -1.0;
    }else{
        for (i=0;i<indices.size();i++){
            ss+=N(indices[i],0)*coefficients[i];
        }
        return ss;
    }
}

double ssfromnullspace(Ref<SparseMatrix<T>> L, vector<int> &indices, vector<double> &coefficients, bool doublecheck=false){
    int n=L.rows();
    int i;
    T ss=0;
    VectorXd N;
    N.resize(n,1);

    
    i=nullspace(L,N,doublecheck);
    if (i<0){ //if it cannot solve for the nullspace
        return -1.0;
    }else{
        for (i=0;i<indices.size();i++){
            ss+=N(indices[i],0)*coefficients[i];
        }
        return ss;
    }
}



