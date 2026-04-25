"""writescripts.py Code to generate .cpp and mathematica files to compute gene regulatory function properties.
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


import sys, os, sympy, re, glob
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.cxx import cxxcode
import numpy as np
import subprocess
import itertools
import auxfuncCG

class PrepareFiles():
    """Parent class in the hierarchy, with common functionality to write to make calculations for graphs at equilibrium and away.
    It is intended to write files to compute the value of a GRF at a given input value,
    and position-steepness of the GRF, using pybind to call c++ code. In addition, it writes mathematica files to double check.
    Initialisation arguments:
    - varGRF:variable with respect to which the GRF is computed
    - concvars: all variables that can change concentration (including varGRF.) For example, rN the Pol model, this would be ['x','P']
    - parlist: list of parameter values. Leave empty for the equilibrium models.

    """ 


    def __init__(self, varGRF='x',concvars=[''],parlist=[]):
        self.varGRF=varGRF #variable with respect to which the GRF is computed
        self.concvars=concvars #all variables that can change concentraiton
        self.parlist=parlist #rates
        self.numsympy=None
        self.densympy=None
        self.coeffs_num=None
        self.coeffs_den=None
        self.coeffs_num_string=None
        self.coeffs_den_string=None
        self.numstring=None
        self.denstring=None
        self.CG=False
        self.multiconf=False #multiple conformation model explicitly
       
    def simpify_num_den(self,numstr,denstr):
        """Given the strings with the mathematical expressions for numerator and denominator of the GRF (numstr, denstr), computes their corresponding sympy expressions. 
        This can be useful to do further symbolic calculation, and is used by compute_coeffs function.
        It is likely to fail for large graphs. So it is probably best to use simpify_rhos and parse_GRF functions
        """
        
        self.numstring=numstr
        self.denstring=denstr

        for label in self.parlist:
            #print(label)
            exec("%s=sympy.symbols('%s')"%(label,label))
        for var in self.concvars:
            #print('symbol', label)
            exec("%s=sympy.symbols('%s')"%(var,var))

        print('Parsing into sympy expressions. This can take a while...')
        sympyexpr=parse_expr(numstr)
        self.numsympy=sympyexpr
        sympyexpr=parse_expr(denstr)
            #print('symbol', label)
        self.densympy=sympyexpr
   
        var=sympy.symbols(self.varGRF)
        self.coeffs_num=sympy.Poly(self.numsympy,var).all_coeffs()[::-1]
        self.coeffs_den=sympy.Poly(self.densympy,var).all_coeffs()[::-1]
        self.coeffs_num_string=','.join(map(cxxcode,self.coeffs_num))
        self.coeffs_den_string=','.join(map(cxxcode,self.coeffs_den))
        self.coeffs_num_fromrhos=None
        self.coeffs_den_fromrhos=None
    
    def simpify_rhos(self):
        """Given the expressions for the rhos in self.rhos, obtains the corresponding coefficients with respect to varGRF and puts them in coeffs_rhos."""
        for label in self.parlist:
            #print(label)
            exec("%s=sympy.symbols('%s')"%(label,label))
        for var in self.concvars:
            exec("%s=sympy.symbols('%s')"%(var,var))

        coeffs_rhos=[]
        #convert each rho into a polynomial and get corresponding coefficients
        for rho in self.all_rhos:
            expr=parse_expr(rho.split("=")[1])
            poly=sympy.Poly(expr,sympy.symbols(self.varGRF))
            coeffs=poly.all_coeffs()[::-1]
            coeffs_rhos.append(coeffs)
        self.coeffs_rhos=coeffs_rhos
        
    def parse_GRF(self,numexpr,denexpr):
        """The expressions should be of the form 1*(rho1)+0.5*(rho2)+0.75*(rho3)+ktan*(rho4). (Notice that even if multiplied by 1, this has to be specified.)
        Notice that the indexing of the rhos starts at 1, to be consistent with the MTT.py. 
        Is also used for the equilibrium graphs, in this case the rhos are the products along paths."""
        
        self.numstring=numexpr
        self.denstring=denexpr
        pat=re.compile("([\(\)0-9aA-zZ\./]*)\*(\([a-z0-9\+]+\))")
        patrho=re.compile("rho([0-9]+)")
        degrees_rhos=[len(x) for x in self.coeffs_rhos] #degree of the polynomials corresponding to each rho
        for enum, expr in enumerate([numexpr,denexpr]):
            all_coeffs_str=[[] for i in range(max(degrees_rhos))] #prepare a list for each of the degrees
            terms=pat.findall(expr)
            for term in terms:
                multiplier,expr=term
                idxs=patrho.findall(expr)
                for idx_ in idxs: #each of the rhos
                    idx=int(idx_)-1 #rho1 occupies position 0 so idx is 0,and so on
                    for i in range(degrees_rhos[idx]): #for each of the terms, decide if coefficient is 0 or not
                        #print('term',term,'expr',expr, 'idx_',idx_)
                        #print('i',i,self.coeffs_rhos[idx], self.coeffs_rhos[idx][i], )
                        if self.coeffs_rhos[idx][i]!=0:
                            all_coeffs_str[i].append(multiplier+"*coeffs_%s[%d]"%(idx_,i))
            print(enum)
            if enum==0:
                self.coeffs_num_fromrhos=[]
                for coeff in all_coeffs_str:
                    #print('coeff num',coeff)
                    if len(coeff)>0:
                        self.coeffs_num_fromrhos.append("+".join(coeff))
                    else:
                        self.coeffs_num_fromrhos.append("0")
            else:
                self.coeffs_den_fromrhos=[]
                for coeff in all_coeffs_str:
                    if len(coeff)>0:
                        self.coeffs_den_fromrhos.append("+".join(coeff))
                    else:
                        self.coeffs_den_fromrhos.append("0")

                #remove 1* so that there are less operations to be done
                pat_one=re.compile(r"^1\*c")
                self.coeffs_den_fromrhos=[re.sub(pat_one,"c",x).replace("+1*c","+c") for x in self.coeffs_den_fromrhos]


    def __write_header(self, fh,posstpfromGRF=False):
        """Writes includes of cpp file."""
        fh.write("#include \"commonincludes.hpp\"\n")
        if posstpfromGRF:
            fh.write("#include \"pos_stp_fromGRF.h\"\n")

        
    def __write_GRF_coeffs(self,fh,funcname_varGRF, typestring, additionallinespars=None):
        """write function that returns the coefficients of the numerator and denominator of the GRF (with respect to varGRF)."""
        
        fh.write("template <typename T>\n")
        fh.write("void %s(py::array_t<double> parsar, vector<T> &num, vector<T> &den, py::array_t<double>othervars){\n"%(funcname_varGRF))
        fh.write("""
    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;\n""")

        if self.CG: #need to get the expressions for the coarsegrained pars, which are the ones used in the computation of the rhos and in self.parlist
            parslistc=auxfuncCG.get_parslist_atconf(self.c,self.N,intrinsiccoop=self.intrinsiccoop,samesites=self.samesites)
            parslistc=parslistc.split(',')
            for i in range(len(parslistc)):
                fh.write("    T %s=pars[%d];\n"%(parslistc[i],i))
            if self.samesites:
                for cn in range(1,self.c+1): #for each conformation
                    for site in range(1,self.N+1): #for each site
                        fh.write('    T K_%d_%d=K_%d;\n'%(cn,site,cn)) #this is just so that I can use same function for effective parameters in all cases
            effK,lines=auxfuncCG.write_effective_Ks0(self.c,self.N,intrinsiccoop=self.intrinsiccoop,pybind=True)
            for line in lines:
                fh.write(line)
            fh.write('\n')
            effw,lines=auxfuncCG.write_effective_w(self.c,self.N,intrinsiccoop=self.intrinsiccoop,pybind=True)
            for line in lines:
                fh.write(line)
            fh.write('\n')
            allpars_CG=effK+effw
            if len(allpars_CG) != len(self.parlist):
                print("Wrong parameter list after coarsegraining.\n Coarse-grained parameters are: %s, parlist parameters are %s"%(",".join(allpars_CG),".".join(self.parlist)))
                raise RuntimeError


        else:

            for pnum,par in enumerate(self.parlist):

                fh.write("    T %s=pars[%d];\n"%(par,pnum))

        if True: #len(self.concvars)>1:
            fh.write("""
    auto varsarbuf=othervars.request();
    double *varsar=(double *) varsarbuf.ptr;\n""")
            i=0
            for var in self.concvars:
                if var != self.varGRF:
                    fh.write("    T %s=varsar[%d];\n"%(var,i))
                    i+=1
        if additionallinespars is not None:
            fh.write(additionallinespars)
        
        if self.coeffs_num_fromrhos is None:

            fh.write("    num={%s};\n"%self.coeffs_num_string)
            fh.write("    den={%s};\n"%self.coeffs_den_string)
        else:
            for i in range(len(self.coeffs_rhos)):
                string_=cxxcode(self.coeffs_rhos[i])
                string_=string_.replace('[','').replace(']','')
                fh.write("    vector<T> coeffs_%d={%s};\n"%(i+1,string_))
            
            for i in range(len(self.coeffs_num_fromrhos)):
                fh.write("    T numdeg%d=%s;\n"%(i,self.coeffs_num_fromrhos[i]))
            for i in range(len(self.coeffs_den_fromrhos)):
                fh.write("    T dendeg%d=%s;\n"%(i,self.coeffs_den_fromrhos[i]))
            
            
            fh.write("    num={")
            for i in range(len(self.coeffs_num_fromrhos)):
                if i<len(self.coeffs_num_fromrhos)-1:
                    end=','
                else:
                    end='};\n'
                fh.write("numdeg%d%s"%(i,end))
                
            fh.write("    den={")
            for i in range(len(self.coeffs_den_fromrhos)):
                if i<len(self.coeffs_den_fromrhos)-1:
                    end=','
                else:
                    end='};\n'
                fh.write("dendeg%d%s"%(i,end))
            
        fh.write("}\n\n")

    def __write_rhos_coeffs(self,fh,funcname_varGRF, additionallinespars=None):
        """write function that returns the coefficients of the rhos (with respect to varGRF)."""
        
        fh.write("template <typename T>\n")
        fh.write("void rhos_%s(py::array_t<double> parsar, vector<T> &rhos, py::array_t<double>othervars, double valGRF){\n"%(funcname_varGRF))
        fh.write("""
    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;\n""")

        if self.CG: #need to get the expressions for the coarsegrained pars, which are the ones used in the computation of the rhos and in self.parlist
            parslistc=auxfuncCG.get_parslist_atconf(self.c,self.N,intrinsiccoop=self.intrinsiccoop,samesites=self.samesites)
            parslistc=parslistc.split(',')
            for i in range(len(parslistc)):
                fh.write("    T %s=pars[%d];\n"%(parslistc[i],i))
            if self.samesites:
                for cn in range(1,self.c+1): #for each conformation
                    for site in range(1,self.N+1): #for each site
                        fh.write('    T K_%d_%d=K_%d;\n'%(cn,site,cn)) #this is just so that I can use same function for effective parameters in all cases
            effK,lines=auxfuncCG.write_effective_Ks0(self.c,self.N,intrinsiccoop=self.intrinsiccoop,pybind=True)
            for line in lines:
                fh.write(line)
            fh.write('\n')
            effw,lines=auxfuncCG.write_effective_w(self.c,self.N,intrinsiccoop=self.intrinsiccoop,pybind=True)
            for line in lines:
                fh.write(line)
            fh.write('\n')
            allpars_CG=effK+effw
            if len(allpars_CG) != len(self.parlist):
                print("Wrong parameter list after coarsegraining.\n Coarse-grained parameters are: %s, parlist parameters are %s"%(",".join(allpars_CG),".".join(self.parlist)))
                raise RuntimeError

        else:
        
            for pnum,par in enumerate(self.parlist):

                fh.write("    T %s=pars[%d];\n"%(par,pnum))

        if len(self.concvars)>1:
            fh.write("""
    auto varsarbuf=othervars.request();
    double *varsar=(double *) varsarbuf.ptr;\n""")
            i=0
            for var in self.concvars:
                if var != self.varGRF:
                    fh.write("    T %s=varsar[%d];\n"%(var,i))
                    i+=1
        if additionallinespars is not None:
            fh.write(additionallinespars)
        
        
        for i in range(len(self.coeffs_rhos)): #for each rho
            string_=""
            for j in range(len(self.coeffs_rhos[i])):
                if j<len(self.coeffs_rhos[i])-1:
                    end="+"
                else:
                    end=";"
                if self.coeffs_rhos[i][j]!=0:
                    string_+="("+cxxcode(self.coeffs_rhos[i][j])+")"
                    string_+="*pow(valGRF,%d)%s"%(j,end)

            fh.write("    T rho_%d=%s\n"%(i+1,string_))
        
            
            
        fh.write("    rhos={")
        for i in range(len(self.coeffs_rhos)):
            if i<len(self.coeffs_rhos)-1:
                end=','
            else:
                end='};\n'
            fh.write("rho_%d%s"%(i+1,end))
                
            
        fh.write("}\n\n")
        
    


    def __write_inheritance_specialisations(self,fh,funcname,funcname_varGRF):
        """Inherit class and define functions to fill numerator and denominator of GRF, as well as rhos."""

        fh.write("""template <typename T, typename Tpolyc, typename polytype, typename thresholdtype>
class %s_GRFCalculations: public GRFCalculations <T, Tpolyc, polytype, thresholdtype>{

    public:

    void fill_num_den(py::array_t<double> parsar, py::array_t<double>othervars){
            
        %s<T>(parsar, this->num,this->den,othervars);
    }

    void fill_rhos(py::array_t<double> parsar, py::array_t<double>othervars, double xval){

        rhos_%s<T>(parsar, this->rhos,othervars, xval);
    }
    
};\n"""%(funcname, funcname_varGRF, funcname_varGRF))
        
        
    def write_pybind_interface(self,fname=None, funcname=None, additionallinespars=None, posstpfromGRF=False, precision_types=[("long double","50","15"),("50","50","15"), ("100","100","15")], typestring="double"):
        """Writes .cpp file with GRF functions and computation of position and steepness. pybind is used to interface with them.
        precision_types: [precision of GRF numerator and denominator, precision of root solving calculations, precision of thershold to decide if imaginary part is zero or not]

        """

        var=sympy.symbols(self.varGRF)
        funcname_varGRF=funcname+'_'+self.varGRF
        
        
        #print(coeffs_num)
        #print(coeffs_den)
        f=open(fname,'w')
        self.__write_header(f,posstpfromGRF=posstpfromGRF)
        
        self.__write_GRF_coeffs(f,funcname_varGRF,additionallinespars) #coefficients of num and den of GRF with respect to input
        self.__write_rhos_coeffs(f,funcname_varGRF,additionallinespars) #rhos with respect to input
        self.__write_inheritance_specialisations(f,funcname,funcname_varGRF)
        if posstpfromGRF:
            raise ValueError("Not implemented for posstpfromGRF=True")
            self.__write_interface_posstp_fromGRF(f,funcname_varGRF,typestring)
        
        template_strings=[]
        template_names=[]
        written_precisions_mpc=[]
        written_precisions_mpfr=[]
        written_precisions_poly=[]
        for precision in precision_types:
            precGRF, precpoly,precthreshold=precision
            template_string="<"
            template_name=""
            if precGRF!= "long double":
                if not precGRF in written_precisions_mpfr:
                    f.write("typedef number<mpfr_float_backend<%s> > mpfr_%s;\n"%(precGRF,precGRF))
                    written_precisions_mpfr.append(precGRF)
                template_string+="mpfr_%s,"%precGRF
                template_name+="%s_"%precGRF
                
            else:
                template_string+="%s,"%precGRF
                template_name+="ld_"
            if not precpoly in written_precisions_mpc: #otherwise it is already written
                f.write("typedef number<mpc_complex_backend<%s> > mpc_%s;\n"%(precpoly,precpoly))
                written_precisions_mpc.append(precpoly)
            if not precpoly in written_precisions_poly:
                f.write("typedef Polynomial<%s> polytype_%s;\n"%(precpoly, precpoly))
                written_precisions_poly.append(precpoly)
            template_string+="mpc_%s,polytype_%s,"%(precpoly,precpoly)
            template_name+="%s_"%precpoly
            if precthreshold!="long double":
                if not precthreshold in written_precisions_mpfr:
                    f.write("typedef number<mpfr_float_backend<%s> > mpfr_%s;\n"%(precthreshold,precthreshold))
                    written_precisions_mpfr.append(precthreshold)

                template_string+="mpfr_%s>"%precthreshold
                template_name+="%s"%precthreshold
            else:
                template_string+="long double>"
                template_name+="ld"

            template_strings.append(template_string)
            template_names.append(template_name)


        f.write("\n\nPYBIND11_MODULE(%s, m) {\n"%fname.split('/')[-1].replace('.cpp',''))
        for p in range(len(template_strings)):
            ts=template_strings[p]
            tn=template_names[p]
            f.write("    py::class_<%s_GRFCalculations%s>(m, \"GRFCalculations_%s\", py::module_local())\n"%(funcname, ts,tn))
            f.write("    .def(py::init())\n")
            f.write("    .def(\"fill_num_den\", &%s_GRFCalculations%s::fill_num_den)\n"%(funcname,ts))
            f.write("    .def(\"fill_rhos\", &%s_GRFCalculations%s::fill_rhos)\n"%(funcname,ts))
            f.write("    .def(\"interfaceGRF\", &%s_GRFCalculations%s::interfaceGRF)\n"%(funcname,ts))
            f.write("    .def(\"getrhos\", &%s_GRFCalculations%s::getrhos)\n"%(funcname,ts))
            f.write("    .def(\"interfaceps\", &%s_GRFCalculations%s::interfaceps,py::arg(\"verbose\")=false, py::arg(\"thresholdimag\")=1e-15,py::arg(\"minx0\")=false,py::arg(\"maxx1\")=false,py::arg(\"writex05coeffs\")=false, py::arg(\"absder\")=false, py::arg(\"normalisefirst\")=true, py::arg(\"fnamecoeffs\")=\"coefficients.txt\")\n"%(funcname,ts))
            f.write("    .def(\"interface_return_num_den\", &%s_GRFCalculations%s::interface_return_num_den)\n"%(funcname,ts))
            f.write("    .def(\"interfacemonotonic\", &%s_GRFCalculations%s::interfacemonotonic, py::arg(\"thresholdimag\")=1e-15)\n"%(funcname,ts))
            f.write("    .def(\"print_num_den\", &%s_GRFCalculations%s::print_num_den);\n\n"%(funcname,ts))

        f.write("}\n")
        f.close()
        
    def write_checkfile_mathematica_singlevar(self,fname='./testmf.txt', additionallinespars=None, min0=True, max1=True):
        print('writing mathematica file. max1 set to', max1)
        if self.strategy=="pol" and max1:
            print("pol model may not asymptote to 1. Set max1 to False. Exiting...")
            raise ValueError

        if self.strategy=="pol_basal" and (max1 or min0):
            print("pol model with basal expression may not start at 0 and may not asymptote to 1. Set min0=False and max1 to False. Exiting...")
            raise ValueError
        
        if len(self.concvars)>1:
            if additionallinespars is None:
                print("There is more than 1 input variable. Pass a string to additionallinespars argument with the values corresponding to those")
              
        f=open(fname,'w')
        f.write("""
Print[\"First line\"];
infolder=$ScriptCommandLine[[2]];
absder=$ScriptCommandLine[[3]];
If [absder=="True", Print["absder True"], Print["absder False"]];
SetDirectory[infolder];
tolpos=0.005;
tolstp=0.005;
Print[\"starting at\"];
Print[infolder];
infiles=FileNames[\"mat*.in\"];\n
        """)
        if self.strategy=="pol_basal":
            f.write("P=ToExpression[$ScriptCommandLine[[4]]];")
            f.write("Print[\"P value is \", P];")
        parstring_underscore=''
        for pnum,par in enumerate(self.parlist):
            par=par.replace('_','U')
            parstring_underscore+=par+'_,'
        print(parstring_underscore)

        parsvars=parstring_underscore

        for var in self.concvars:
            if var != self.varGRF:
                parsvars+=var+'_,'
        parsvars=parsvars.strip(',') #remove last comma
        allunderscore=parsvars+','+self.varGRF+'_'
        allnounderscore=allunderscore.replace('_','')
        
        if self.coeffs_num_fromrhos is None:
            f.write('num[%s]:=%s;\n'%(allunderscore,self.numstring.replace('_','U')))
            f.write('den[%s]:=%s;\n'%(allunderscore,self.denstring.replace('_','U')))
        else:
            for i in range(len(self.coeffs_rhos)):
                rho_coeffs=self.coeffs_rhos[i] #rho_i expressed as the coefficients of the polynomial
                poly_string=[]
                for d in range(len(rho_coeffs)): #for each coeffiscient
                    poly_string.append('('+str(rho_coeffs[d]).replace('_','U')+')'+"*"+self.varGRF+"^%d"%d)
                
                f.write("rho%d[%s]:=%s;\n"%(i+1,allunderscore,"+".join(poly_string)))
            
            pat=re.compile("(rho[0-9]+)")
            numstring_pars=re.sub(pat,"\g<1>[%s]"%(allnounderscore),self.numstring)
            denstring_pars=re.sub(pat,"\g<1>[%s]"%(allnounderscore),self.denstring)
            f.write("num[%s]:=%s;\n"%(allunderscore,numstring_pars))
            f.write("den[%s]:=%s;\n"%(allunderscore,denstring_pars))
        
        
        f.write('GRF[%s]:=num[%s]/den[%s];\n'%(allunderscore,allnounderscore,allnounderscore))
        parstring_nounderscore=parstring_underscore.replace('_','')
        parstring_nounderscore=parstring_nounderscore.strip(',')
        npars=len(parstring_nounderscore.split(','))
        if not self.CG:
            f.write('{%s}=Table[0,{i,1,%d}];\n'%(parstring_nounderscore,npars))
        else:
            pars_conf=auxfuncCG.get_parslist_atconf(self.c,self.N,intrinsiccoop=self.intrinsiccoop,samesites=self.samesites)
            pars_conf_l=pars_conf.split(',')
            parstring_conf_l_u=""

            for par in pars_conf_l:
                parU=par.replace('_','U')
                parstring_conf_l_u+=par+","
                f.write('%s=0;\n'%parU)
            parstring_conf_l_u.strip(",")
        if additionallinespars is not None:
            f.write(additionallinespars)
        if not self.CG:
            parstring_nounderscore_commas=["\"%s\""%x for x in parstring_nounderscore.split(',')]
        else:
            parstring_nounderscore_commas=["\"%s\""%x for x in parstring_conf_l_u.split(',')]
        
        f.write("parsliststring={%s};"%(",".join(parstring_nounderscore_commas))) #parameters searched
        f.write("""
Print[\"Defined GRF\"];
logspace[a_, b_, n_] := 10.0^Range[a, b, (b - a)/(n - 1)];
ar = logspace[-60, 60, 6000];
For[j=1,j<=Length[infiles],j++, 
infname=infiles[[j]];
WriteString[\"stdout\",infname,\"\\n\"];
outfname=StringJoin[StringSplit[infname,\".\"][[1]],\"_checked.out\"];
outfname2=StringJoin[StringSplit[infname,\".\"][[1]],\"_discarded_maxD0.out\"];
WriteString[\"stdout\",outfname,\"\\n\"];
data=Import[infname,\"CSV\"];
outf=OpenWrite[outfname];
outf2=OpenWrite[outfname2];
WriteString[outf,StringRiffle[{\"pos\",\"rho\"},","],";", StringRiffle[parsliststring,","],"\\n"];
For[i =1,i<= Length[data], i++,
{pos0, stp0}=data[[i]][[1;;2]];
PLIST=data[[i]][[3;;]];
""")
        if not self.CG:
            f.write("{%s}=PLIST[[1;;Length[PLIST]]];\n"%parstring_nounderscore)
        else:
            parscgexpr,parscg=auxfuncCG.return_CGpars_expr(self.c,self.N,intrinsiccoop=self.intrinsiccoop)
            f.write("{%s}=PLIST[[1;;Length[PLIST]]];\n"%pars_conf.replace('_','U'))
            if self.samesites: #in this case need to go back to the extended names with the site at which binding is occurring
                for cn in range(1,self.c+1): #for each conformation
                    for site in range(1,self.N+1): #for each site
                        name="KU%dU%d"%(cn,site)
                        f.write('    %s=KU%d;\n'%(name,cn)) #this is just so that I can use same function for effective parameters in all cases
            for i in range(len(parscgexpr)):
                f.write(parscgexpr[i].strip().replace('_','U')+';\n')
        f.write("f[%s_]:=GRF[%s];\n"%(self.varGRF,allnounderscore))
        if min0 and max1:
            f.write("halfX = Solve[f[%s]==1/2&&%s>0,%s];\n"%(self.varGRF,self.varGRF,self.varGRF))
        elif min0 and not max1:
            f.write("maxX = f[x]/. {x -> $MaxMachineNumber};\n")
            f.write("halfX = Solve[f[x]==maxX*0.5&&x>0,x];\n")
        else: # max1 and not min0:
            f.write("min=Min[f[ar]];\n")
            f.write("max=Max[f[ar]];\n")
            f.write("midpoint=min+0.5*(max-min);\n")
            f.write("halfX = Solve[f[x]==midpoint&&x>0,x];\n")

        f.write("""

g[yv2]:=f[halfX[[1]][[1]][[2]]*yv2];
maxY = Solve[D[g[yv2],{yv2,2}]==0&&yv2>0,yv2];
maxY=Append[maxY,{yv2->0}];
If [absder=="True", maxD = Abs[D[g[yv2],yv2]/.maxY], maxD = D[g[yv2],yv2]/.maxY];
rho =Max[maxD]; (* TAKE MAX GLOBAL DERIVATIVE *)
If [absder=="True", checkrho=Abs[D[g[yv2], yv2] /. {yv2 -> 0}], checkrho=D[g[yv2], yv2] /. {yv2 -> 0}];
If [rho == checkrho, 
WriteString["stdout",i, "maxat0,"];
WriteString[outf2,StringRiffle[PLIST,","],"\\n"];,

pos = maxY[[Position[maxD,Max[maxD]][[1]][[1]]]][[1]][[2]];(* TAKE POS CORRESPONDING TO MAX GLOBAL DERIVATIVE *)
If[ pos-pos0<tolpos&&rho-stp0<tolstp,
WriteString["stdout",i,","];
WriteString[outf,StringRiffle[{pos,rho},","],";", StringRiffle[PLIST,","],"\\n"];,

WriteString["stdout","\\n not correct"];
WriteString["stdout",pos,",",rho,";",pos0,",", stp0,";"];
]
]
]
Close[outf];
Close[outf2];
]
        """)
        
    def write_checksingleparset_mathematica_singlevar(self,fname='./testmf.txt', additionallinespars=None, min0=True,max1=True):
        """Writes a mathematica file to check the calculation for a given parameter set. Adapted from code by Felix Wong."""
        print('writing mathematica file. max1 set to', max1)
        if self.strategy=="pol" and max1:
            print("pol model may not asymptote to 1. Set max1 to False. Exiting...")
            raise ValueError
        if self.strategy=="pol_basal" and (max1 or min0):
            print("pol model with basal expression may not start at 0 and may not asymptote to 1. Set min0=False and max1 to False. Exiting...")
            raise ValueError
        if len(self.concvars)>1:
            if additionallinespars is None:
                print("There is more than 1 input variable. Pass a string to additionallinespars argument with the values corresponding to those")
              
        f=open(fname,'w')
        f.write("""absder=\"False\";\n""")
        parstring_underscore=''
        for pnum,par in enumerate(self.parlist):
            par=par.replace('_','U')
            parstring_underscore+=par+'_,'
        print(parstring_underscore)

        parsvars=parstring_underscore

        for var in self.concvars:
            if var != self.varGRF:
                parsvars+=var+'_,'
        parsvars=parsvars.strip(',') #remove last comma
        allunderscore=parsvars+','+self.varGRF+'_'
        allnounderscore=allunderscore.replace('_','')
        
        if self.coeffs_num_fromrhos is None:
            f.write('num[%s]:=%s;\n'%(allunderscore,self.numstring.replace('_','U')))
            f.write('den[%s]:=%s;\n'%(allunderscore,self.denstring.replace('_','U')))
        else:
            for i in range(len(self.coeffs_rhos)):
                rho_coeffs=self.coeffs_rhos[i] #rho_i expressed as the coefficients of the polynomial
                poly_string=[]
                for d in range(len(rho_coeffs)): #for each coeffiscient
                    poly_string.append('('+str(rho_coeffs[d]).replace('_','U')+')'+"*"+self.varGRF+"^%d"%d)
                
                f.write("rho%d[%s]:=%s;\n"%(i+1,allunderscore,"+".join(poly_string)))
            
            pat=re.compile("(rho[0-9]+)")
            numstring_pars=re.sub(pat,"\g<1>[%s]"%(allnounderscore),self.numstring)
            denstring_pars=re.sub(pat,"\g<1>[%s]"%(allnounderscore),self.denstring)
            f.write("num[%s]:=%s;\n"%(allunderscore,numstring_pars))
            f.write("den[%s]:=%s;\n"%(allunderscore,denstring_pars))
        
        
        f.write('GRF[%s]:=num[%s]/den[%s];\n'%(allunderscore,allnounderscore,allnounderscore))
        parstring_nounderscore=parstring_underscore.replace('_','')
        parstring_nounderscore=parstring_nounderscore.strip(',')
        npars=len(parstring_nounderscore.split(','))
        if not self.CG:
            f.write('{%s}=Table[0,{i,1,%d}];\n'%(parstring_nounderscore,npars))
        else:
            pars_conf=auxfuncCG.get_parslist_atconf(self.c,self.N,intrinsiccoop=self.intrinsiccoop,samesites=self.samesites)
            pars_conf_l=pars_conf.split(',')
            for par in pars_conf_l:
                f.write('%s=0;\n'%par.replace('_','U'))
        if additionallinespars is not None:
            f.write(additionallinespars)
        f.write("""
Print[\"Defined GRF\"];
logspace[a_, b_, n_] := 10.0^Range[a, b, (b - a)/(n - 1)];
ar = logspace[-60, 60, 6000];
PLIST={};
""")
        if not self.CG:
            f.write("{%s}=PLIST[[1;;Length[PLIST]]];\n"%parstring_nounderscore)
        else:
            parscgexpr,parscg=auxfuncCG.return_CGpars_expr(self.c,self.N,intrinsiccoop=self.intrinsiccoop)
            f.write("{%s}=PLIST[[1;;Length[PLIST]]];\n"%pars_conf.replace('_','U'))
            if self.samesites: #in this case need to go back to the extended names with the site at which binding is occurring
                for cn in range(1,self.c+1): #for each conformation
                    for site in range(1,self.N+1): #for each site
                        name="KU%dU%d"%(cn,site)
                        f.write('    %s=KU%d;\n'%(name,cn)) #this is just so that I can use same function for effective parameters in all cases
            for i in range(len(parscgexpr)):
                f.write(parscgexpr[i].strip().replace('_','U')+';\n')
        f.write("f[%s_]:=GRF[%s];\n"%(self.varGRF,allnounderscore))
        if min0 and max1:
            f.write("halfX = Solve[f[%s]==1/2&&%s>0,%s];\n"%(self.varGRF,self.varGRF,self.varGRF))
        elif min0 and not max1:
            f.write("maxX = f[x]/. {x -> $MaxMachineNumber};\n")
            f.write("halfX = Solve[f[x]==maxX*0.5&&x>0,x];\n")
        else: # max1 and not min0:
            f.write("min=Min[f[ar]];\n")
            f.write("max=Max[f[ar]];\n")
            f.write("midpoint=min+0.5*(max-min);\n")
            f.write("halfX = Solve[f[x]==midpoint&&x>0,x];\n")
        f.write("""
g[yv2]:=f[halfX[[1]][[1]][[2]]*yv2];
maxY = Solve[D[g[yv2],{yv2,2}]==0&&yv2>0,yv2];
maxY=Append[maxY,{yv2->0}];
If [absder=="True", maxD = Abs[D[g[yv2],yv2]/.maxY], maxD = D[g[yv2],yv2]/.maxY];
rho =Max[maxD]; (* TAKE MAX GLOBAL DERIVATIVE *)
If [absder=="True", checkrho=Abs[D[g[yv2], yv2] /. {yv2 -> 0}], checkrho=D[g[yv2], yv2] /. {yv2 -> 0}];
If [rho == checkrho, 
WriteString["stdout","discarding GRF with max D at 0\\n"];
WriteString["stdout",StringRiffle[PLIST,","],"\\n"];,

pos = maxY[[Position[maxD,Max[maxD]][[1]][[1]]]][[1]][[2]];(* TAKE POS CORRESPONDING TO MAX GLOBAL DERIVATIVE *)
{pos,rho}
]
Plot[g[yv2] /. {yv2 -> y0}, {y0, 0, 2}]

        """)
    

class PrepareFilesNoneq(PrepareFiles):
    """Child class to be used for graphs away from equilibrium. Initialisation arguments:
    - edges: should be a list of edges in the form [inputnode,'label',targetnode].
    - MTT folder: folder where MTT.py script is found.
    - graphbasename: name for the grpah files. Notice that it erases existing files with this basename."""

    def __init__(self,varGRF='x',concvars=[''],parlist=[],edgelist=[],MTTfolder='',graphbasename='graph',strategy="noneq"):
        super().__init__(varGRF=varGRF,concvars=concvars,parlist=parlist)
        self.edges=edgelist
        self.MTTfolder=MTTfolder
        self.graphbasename=graphbasename
        self.graphname=os.path.join(self.MTTfolder,self.graphbasename+'.txt')
        self.strategy=strategy
        files=glob.glob(os.path.join(self.MTTfolder,self.graphbasename)+"*")
        for f in files:
            print('removing ',f)
            os.remove(f)
            
    def write_execute_parse(self):
        #write input file for MTT.py
        
        outf=open(self.graphname,'w')
        for x in self.edges:
            x_str=list(map(str,x))
            outf.write('('+','.join(x_str)+') ')
        outf.close()
        
        #execute_MTT(self):
        
        args=["python",os.path.join(self.MTTfolder,'MTT.py'),self.graphname]
        print("executing MTT")
        print(subprocess.check_call(args))
        
    
        #parse_rhos_from_graphfiles(self):
        #this was previously called rhos_from_edges
        edges=np.array(self.edges)
        #find files that correspond to the spanning trees
        files=glob.glob(os.path.join(self.MTTfolder,self.graphbasename+'*'))
        pat=re.compile('[0-9]+.txt')
        n=0
        for f in files:
            if pat.findall(f):
                n+=1
        all_rhos=[] 
        for i in range(1,n+1):
            fname=os.path.join(self.MTTfolder,self.graphbasename+'-%d.txt'%i)

            fi=open(fname,'r')
            rho_expr='rho_%d='%i

            for lnum,l in enumerate(fi.readlines()):
                idxs=np.array(list(map(int,l.strip().split())),dtype=bool)
                subset=edges[idxs]
                expr=[x[1] for x in subset] #labels that multiply together in a term of the rho
                if lnum<1:
                    rho_expr=rho_expr+'*'.join(expr)
                else:
                    rho_expr=rho_expr+'+'+'*'.join(expr)

            all_rhos.append(rho_expr)
        
        self.all_rhos=[]
        for rho in all_rhos:
            for var in self.concvars:
                var_1='-'+var
                var_2='*'+var
                if var_1 in rho:
                    #print('replacing',var_1,var_2)
                    rho=rho.replace(var_1,var_2)
                    #print(rho)
            self.all_rhos.append(rho)

class PrepareFilesEqbindingmodels(PrepareFiles):
    """Child class to be used for hypercube graphs with only TF or TF and Pol. 
    N is the number of TF binding sites (excluding polymerase in case of the Pol model).
    Initialise only with varGRF and concvars."""
    def __init__(self,varGRF='x',concvars=['x','P'], N=3, c=1, strategy='pol', sitedict=None, intrinsiccoop=False,samesites=False):
        
        super().__init__(varGRF=varGRF,concvars=concvars)
        
        self.multiconf=False
        self.multiTF=False
        if intrinsiccoop is True and samesites is True:
            print("Intrinsiccoop and samesites are incompatible. Exiting...")
            raise ValueError
        
        strategies=["an", "av", "oom", "pol", "pol_basal", "anyfi"]
        if not strategy in strategies:
            print("%s is not a known strategy"%strategy)
            print("Acceptable strategies are %s"%",".join(strategies))
        else:
            self.strategy=strategy #an, av, oom, pol, pol_basal, anyfi, 

        if ('P' in concvars and not strategy in ["pol", "pol_basal"]) or (strategy in ["pol", "pol_basal"] and not 'P' in concvars):
            print("Only for Pol model strategy should be pol or pol_basal.  Alternatively, use multiTF or multiTF&multiconf strategies with a variable name that is not P. Exiting...")
            raise ValueError
        if sitedict is not None:
            
            sitedict2=dict() #make sure keys are strings
            for key in sitedict.keys():
                sitedict2[str(key)]=str(sitedict[key])
            sitedict=sitedict2
            self.sitedict=sitedict
            self.multiTF=True
    
        if c>1:
            self.multiconf=True
        self.N=N
        self.c=c
        self.intrinsiccoop=intrinsiccoop
        self.samesites=samesites
    
        
    def __get_rho(self, sites, sitep=None,c=1):
        #product of labels from reference to node, such that labels are the bare constants and cooperativities with wij, i<j
        #sites is a string with the bound sites
        #sitep is the Pol site (should be the site with largest index)

        sites_=sites.split(',')
        if len(sites_)==1:
            TFs=""
            site=sites_[0]
            if self.multiTF:
                TFs+="%s**1"%self.sitedict[site]
            else:
                TFs+="%s**1"%self.varGRF

            if sitep is None:
                if self.multiconf:
                    prefix=""
                    K='K_%d_'%c
                    if c>1:
                        prefix="l%d * "%c

                    rho=prefix+'%s%s * %s'%(K,sites_[0],TFs)
                else:
                    rho='K%s * %s'%(sites_[0],TFs)
            else:
                if sites_[0]==sitep:
                    rho='P'
                else:
                    rho='K%s * %s'%(sites_[0],TFs) #the power to the 1 is used below for parsing
        else:
            rho=''

            if self.multiconf:
                prefix=""
                if c>1:
                    prefix="l%d * "%c
                    rho=prefix
            if (sitep is None) or (sitep not in sites_):
                niter=len(sites_)-1
                for i in range(niter):
                    i1=sites_[i]
                    #if len(sites_[i+1:])>1:
                    i2=''.join(sites_[i+1:])
                    if self.multiconf:
                        prefix=""
                        K='K_%d_'%c
                        w='w_%d_'%c
                        rho+='%s%s * %s%s%s * '%(K,i1,w, i1,i2)
                    else:
                        rho+='K%s * w%s%s * '%(i1, i1,i2)
            else:
                niter=len(sites_)-2
                for i in range(niter):
                    i1=sites_[i]
                    #if len(sites_[i+1:])>1:
                    i2=''.join(sites_[i+1:-1])
                    rho+='K%s * w%s%s * '%(i1, i1,i2)
                rho+='K%s * '%(sites_[niter])

            if sites_[-1]==sitep:
                ns=len(sites_)-1

                if self.multiTF:
                    TFlist=[self.sitedict[site] for site in sites_[:-1]]
                    TFs=",".join(TFlist)
                else:
                    TFs="%s**%d"%(self.varGRF,ns)

                
                rho+='P * wp%s * %s'%(''.join(sites_[:-1]),TFs)
            else:
                ns=len(sites_)
                if self.multiTF:
                    TFlist=[self.sitedict[site] for site in sites_]
                    TFs=""
                    for TF in np.unique(TFlist):
                        ncounts=TFlist.count(TF)
                        TFs+="%s**%d * "%(TF,ncounts)
                    TFs=TFs.strip(" * ")
                    
                else:
                    TFs="%s**%d"%(self.varGRF,ns)
                if self.multiconf:
                    rho+='K_%d_%s * %s'%(c,sites_[-1],TFs)
                else:
                    rho+='K%s * %s'%(sites_[-1],TFs)

        return rho

    
    
    def get_rhos_and_GRFs(self, verbose=True):
        """Generate expressions for rhos and GRFs."""
        
        if 'P' in self.concvars:
            sitep=str(self.N+1)
            Niter=self.N+2
        else:
            sitep=None
            Niter=self.N+1
        
        
        symbols_str=set() #to get parlist
        
        nrho=2
        if self.multiconf:
            Niter_conf=self.c
        else:
            Niter_conf=1
        for c in range(1,Niter_conf+1):
            if c==1:
                self.all_rhos=['rho1=1'] #reference node
            else:
                self.all_rhos.append('rho%d=l%d'%(nrho,c))
                nrho+=1
            for i in range(Niter):
                combis=itertools.combinations(np.arange(1,Niter),i)
                for x in combis:
                    if len(x)>0:
                        combistr=','.join(map(str,x))
                        rho=self.__get_rho(combistr,sitep=sitep,c=c)
            
                        self.all_rhos.append('rho%d=%s'%(nrho,rho))
                        #print(nrho,rho)
                        nrho+=1
                        for word in rho.split(' '):
                            symbols_str.add(word.replace('(','').replace(')','').replace('*','').replace('+',''))
    
        #get parlist, sorted such that Ks are first, omegas are second and omegasP last
        #as I am introducing the multiconf, I will put the "l" in terms of the omegasP
        
        nrhos=len(self.all_rhos)

        symbols_str_sorted=[[],[],[]]
        for i in range(3):
            for snum,s in enumerate(symbols_str):
                if len(s)>0 and not 'P' in s and not s in self.concvars:
                    if i==0:
                        if 'K' in s:
                            symbols_str_sorted[i].append(s)
                    if i==1:
                        if 'w' in s and not 'wp' in s:
                            symbols_str_sorted[i].append(s)
                    elif i==2:
                        if 'wp' in s or 'l' in s:
                            symbols_str_sorted[i].append(s)
        symbols_str_sorted_=[]
        for x in symbols_str_sorted:
            #from a stackoverflow question: python sort is stable, so you can sort in multiple passes (in reverse order):
            x.sort() 
            x.sort(key=lambda x:len(x))
            symbols_str_sorted_.extend(x)
        print(symbols_str_sorted_)
        if self.strategy=="anyfi":
            list_fi=["f%d"%(i+1) for i in range(nrhos)]
            self.parlist=list_fi+symbols_str_sorted_ #I am putting them at the beginning because the li need to go at the end so that they are appropriately treated when exploring
        else:
            self.parlist=symbols_str_sorted_
        print('parlist is', self.parlist)
        
        #get GRF
        
        if self.strategy=='an':
            if self.c>1:
                raise Exception("strategy an is not implemented for more than 1 conformation.")
            numstring='1*(rho%d)'%(len(self.all_rhos))
        if self.strategy=='av':
            xppat=re.compile('%s\*\*([0-9]*)'%self.varGRF)
            numstring=''
            for i in range(len(self.all_rhos)):
                rho=self.all_rhos[i]
                power_=xppat.findall(rho)
                if len(power_)>0:
                    for power in power_:
                        numstring+='(%s.0/%d)*(rho%d)+'%(power,self.N,i+1)
            numstring=numstring.strip('+')
        if self.strategy=='oom':
            print("strategy oom has not been implemented. Exiting...")
            sys.exit()
        if self.strategy=='pol':
            numstring='1*('
            for i in range(len(self.all_rhos)):
                rho=self.all_rhos[i]
                if 'P' in rho and self.varGRF in rho:
                    numstring+='rho%d+'%(i+1)
            numstring=numstring.strip('+')
            numstring+=')'

        if self.strategy=='pol_basal':
            numstring='1*('
            for i in range(len(self.all_rhos)):
                rho=self.all_rhos[i]
                if 'P' in rho: #only P is enough
                    numstring+='rho%d+'%(i+1)
            numstring=numstring.strip('+')
            numstring+=')'
        if self.strategy=="anyfi":
            numstring=''
            for i in range(nrhos):
                #rho=self.all_rhos[i]
                numstring+="(f%d)*(rho%d)+"%(i+1,i+1)
            numstring=numstring.strip("+")


        denstring='1*('
        for i in range(nrhos):
            denstring+='rho%d+'%(i+1)
        denstring=denstring.strip('+')
        denstring+=')'
        
        self.simpify_rhos()
        if verbose:
            print('GRF numerator:%s'%numstring)
            print('GRF denominator:%s'%denstring)
        self.parse_GRF(numstring,denstring)

class PrepareFilesEqbindingmodels_CG(PrepareFilesEqbindingmodels):

    """It coarsegrains conformations (c) and proceeds with the coarse-grained parameters as if it were a regular hypercube graph. Only implemented for simple ligand binding x, not P.
    intrinsiccoop=True if intrinsiccoop is allowed at each conformation.
    samesites=True if all sites in a conformation behave equally. 
    If intrinsiccoop=False and samesites=False, then each site at each conformation has their own affinity, which is independent of the other occupied sites at that conformation."""

    def __init__(self,varGRF='x',concvars=['x'], N=3, strategy='av', c=2, intrinsiccoop=False,samesites=False):
        super().__init__(varGRF=varGRF,concvars=concvars,strategy=strategy,N=N)
        if c<2:
            print("CG should only be used if there are at least 2 conformations. Exiting...")
            raise ValueError
        if intrinsiccoop is True and samesites is True:
            print("Intrinsiccoop and samesites are incompatible. Exiting...")
            raise ValueError
        self.CG=True
        self.c=c
        self.intrinsiccoop=intrinsiccoop
        self.samesites=samesites

class PrepareFilesEqbindingmodels_multiconf(PrepareFilesEqbindingmodels):

    """It writes code for multiple conformations, detailing each node. Only implemented for simple ligand binding x, not P.
    intrinsiccoop=True if intrinsiccoop is allowed at each conformation.
    samesites=True if all sites in a conformation behave equally. 
    If intrinsiccoop=False and samesites=False, then each site at each conformation has their own affinity, which is independent of the other occupied sites at that conformation."""

    def __init__(self,varGRF='x',concvars=['x'], N=3, strategy='av', c=2, intrinsiccoop=False,samesites=False,sitedict=None):
        raise Exception("This class is deprecated. Use PrepareFilesEqbindingmodels with the desired number of conformations passed to the c argument, and appropriate intrinsiccoop and samesites flags.")

    
