
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int N,i,n_iter,N_iter,isC,m,n;
    float *x2,*xo2,*b2,*Ap2,*r2,*p2,*op,alpha,beta,s0,s,s2,tmp,ErrTol;
    char *AtA;
    mxArray *AtAx,*r,*p,*x,*b,*AtA_lhs[1],*AtA_rhs[2],*xo,*op_ErrTol,*op_n_iter;

    if(nrhs != 6 || nlhs != 3)
        mexErrMsgIdAndTxt("CG033114:Arguments", "Expected six inputs and three outputs.");
    if(!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]) ||
       mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]) ||
       mxGetNumberOfElements(prhs[0]) != mxGetNumberOfElements(prhs[1]) ||
       mxIsComplex(prhs[0]) != mxIsComplex(prhs[1]))
        mexErrMsgIdAndTxt("CG033114:Type", "x and b must be matching full single arrays.");
    if(mxGetScalar(prhs[4]) <= 0 || mxGetScalar(prhs[4]) >= 1 || mxGetScalar(prhs[5]) < 1)
        mexErrMsgIdAndTxt("CG033114:Tolerance", "Invalid tolerance or iteration count.");
    if(mxIsComplex(prhs[0]))
    {isC=1;}
    else
    {isC=0;}
    x=(mxArray *)prhs[0];
    b=(mxArray *)prhs[1];
    AtA=mxArrayToString(prhs[2]);
    AtA_rhs[1]=(mxArray *)prhs[3];
    ErrTol=(float)mxGetScalar(prhs[4]);
    N_iter=(int)mxGetScalar(prhs[5]);

    N=(int)mxGetNumberOfElements(x);
    m=(int)mxGetM(x);
    n=(int)mxGetN(x);

    op_ErrTol=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,0);
    op_n_iter=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,0);

    if(isC==1)
    {   AtA_lhs[0]=NULL;
        r=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,1);
        p=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,1);
        xo=mxCreateNumericMatrix(m,n,mxSINGLE_CLASS,1);
    }
    else
    {   AtA_lhs[0]=NULL;
        r=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,0);
        p=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,0);
        xo=mxCreateNumericMatrix(m,n,mxSINGLE_CLASS,0);
    }
    //Initialization
    {   x2=mxGetData(x);
        xo2=mxGetData(xo);
        for(i=0;i<N;i++)
        {xo2[i]=x2[i];}
    }
    if(isC==1)
    {   x2=mxGetImagData(x);
        xo2=mxGetImagData(xo);
        for(i=0;i<N;i++)
        {xo2[i]=x2[i];}
    }

    AtA_rhs[0]=xo;
    mexCallMATLAB(1,AtA_lhs,2,AtA_rhs,AtA);
    s=0;
    {   Ap2=mxGetData(AtA_lhs[0]);
        b2=mxGetData(b);
        r2=mxGetData(r);
        p2=mxGetData(p);
        for(i=0;i<N;i++)
        {   r2[i]=b2[i]-Ap2[i];
            p2[i]=r2[i];
            s+=r2[i]*r2[i];
        }
    }
    if(isC==1)
    {   Ap2=mxGetImagData(AtA_lhs[0]);
        b2=mxGetImagData(b);
        r2=mxGetImagData(r);
        p2=mxGetImagData(p);
        for(i=0;i<N;i++)
        {   r2[i]=b2[i]-Ap2[i];
            p2[i]=r2[i];
            s+=r2[i]*r2[i];
        }
    }
    mxDestroyArray(AtA_lhs[0]);
    AtA_lhs[0]=NULL;
    s0=s;
//    printf("%f\n",ErrTol);
    ErrTol=s*ErrTol;
//    printf("%f\n",ErrTol);
    //CG iteration
    for(n_iter=0;n_iter<N_iter&&s>ErrTol;n_iter++)
    {   //Step 1
        AtA_rhs[0]=p;
        mexCallMATLAB(1,AtA_lhs,2,AtA_rhs,AtA);
        //Step 2
        tmp=0;
        {   Ap2=mxGetData(AtA_lhs[0]);
            p2=mxGetData(p);
            for(i=0;i<N;i++)
            {tmp+=p2[i]*Ap2[i];}
        }
        if(isC==1)
        {   Ap2=mxGetImagData(AtA_lhs[0]);
            p2=mxGetImagData(p);
            for(i=0;i<N;i++)
            {tmp+=p2[i]*Ap2[i];}
        }
        //Step 3
        if(!mxIsFinite(tmp) || tmp <= 0)
            mexErrMsgIdAndTxt("CG033114:Curvature", "Normal operator is not positive definite.");
        alpha=s/tmp;
//        printf("s=%le,tmp=%le,alpha=%le\n",s,tmp,alpha);
        s2=0;
        {   Ap2=mxGetData(AtA_lhs[0]);
            r2=mxGetData(r);
            xo2=mxGetData(xo);
            p2=mxGetData(p);
            for(i=0;i<N;i++)
            {   xo2[i]+=alpha*p2[i];
                r2[i]+=-alpha*Ap2[i];
                s2+=r2[i]*r2[i];
            }
        }
        if(isC==1)
        {   Ap2=mxGetImagData(AtA_lhs[0]);
            r2=mxGetImagData(r);
            xo2=mxGetImagData(xo);
            p2=mxGetImagData(p);
            for(i=0;i<N;i++)
            {   xo2[i]+=alpha*p2[i];
                r2[i]+=-alpha*Ap2[i];
                s2+=r2[i]*r2[i];
            }
        }
        mxDestroyArray(AtA_lhs[0]);
        AtA_lhs[0]=NULL;
        //Step 4
        beta=s2/s;
        s=s2;
        {   r2=mxGetData(r);
            p2=mxGetData(p);
            for(i=0;i<N;i++)
            {p2[i]=r2[i]+beta*p2[i];}
        }
        if(isC==1)
        {   r2=mxGetImagData(r);
            p2=mxGetImagData(p);
            for(i=0;i<N;i++)
            {p2[i]=r2[i]+beta*p2[i];}
        }
//        printf("s=%le,tol=%le\n",s,ErrTol);
    }

    op=mxGetData(op_ErrTol);op[0]=s0>0 ? s/s0 : 0;
    op=mxGetData(op_n_iter);op[0]=n_iter;

    plhs[0]=xo;
    plhs[1]=op_ErrTol;
    plhs[2]=op_n_iter;

    mxFree(AtA);
    mxDestroyArray(r);
    mxDestroyArray(p);
}
