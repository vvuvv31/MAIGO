
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int id_x12,id_x,id_xr,id_xs,N,i,j,isC;
    float *res,*x0,*x,mu_x1,mu_x2,mu_x;
    mxArray *var,*X,*AX,*AmX,*AtmX,*lhs[1],*rhs[4];

    if(nrhs != 2 || nlhs != 1 || !mxIsSingle(prhs[0]) ||
       mxIsSparse(prhs[0]) || !mxIsStruct(prhs[1]))
        mexErrMsgIdAndTxt("JtJ:Arguments", "Expected a full single vector and operator struct.");
    if(mxIsComplex(prhs[0]))
    {isC=1;}
    else
    {isC=0;}
    id_x12=(int)mxGetScalar(mxGetField(prhs[1],0,"id_x12"));
    if(id_x12==1)
    {   N=(int)mxGetNumberOfElements(prhs[0])/2;
        mu_x1=(float)mxGetScalar(mxGetField(prhs[1],0,"mu_x1"));
        mu_x2=(float)mxGetScalar(mxGetField(prhs[1],0,"mu_x2"));
        if(isC==1)
        {   plhs[0]=mxCreateNumericMatrix(2*N,1,mxSINGLE_CLASS,1);
            X=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,1);
            AX=NULL;
        }
        else
        {   plhs[0]=mxCreateNumericMatrix(2*N,1,mxSINGLE_CLASS,0);
            X=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,0);
            AX=NULL;
        }
    }
    else
    {   N=(int)mxGetNumberOfElements(prhs[0]);
        if(isC==1)
        {   plhs[0]=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,1);
            AX=NULL;
        }
        else
        {   plhs[0]=mxCreateNumericMatrix(N,1,mxSINGLE_CLASS,0);
            AX=NULL;
        }
    }

    id_xs=(int)mxGetScalar(mxGetField(prhs[1],0,"id_xs"));
    id_xr=(int)mxGetScalar(mxGetField(prhs[1],0,"id_xr"));
    if(id_xs==1||id_xr==1)
    {id_x=1;}
    else
    {id_x=0;}
    mu_x=0;
    if(id_xr==1)
    {mu_x+=(float)mxGetScalar(mxGetField(prhs[1],0,"mu_xr"));}
    if(id_xs==1)
    {mu_x+=(float)mxGetScalar(mxGetField(prhs[1],0,"mu_xs"));}

    AmX=(mxArray *)mxGetField(prhs[1],0,"AmX");
    AtmX=(mxArray *)mxGetField(prhs[1],0,"AtmX");
    var=(mxArray *)mxGetField(prhs[1],0,"var_AtA");

    if(id_x12==1)
    {   x0=mxGetData(prhs[0]);
        x=mxGetData(X);
        for(i=0,j=N;i<N;i++,j++)
        {x[i]=x0[i]+x0[j];}
        if(isC==1)
        {   x0=mxGetImagData(prhs[0]);
            x=mxGetImagData(X);
            for(i=0,j=N;i<N;i++,j++)
            {x[i]=x0[i]+x0[j];}
        }
        rhs[0]=X;lhs[0]=AX;
    }
    else
    {rhs[0]=(mxArray *)prhs[0];lhs[0]=AX;}

    rhs[1]=AmX;rhs[2]=AtmX;rhs[3]=var;
    mexCallMATLAB(1,lhs,4,rhs,"AtAmX");

    res=mxGetData(plhs[0]);
    x=mxGetData(lhs[0]);
    for(i=0;i<N;i++)
    {res[i]=x[i];}
    if(isC==1)
    {   res=mxGetImagData(plhs[0]);
        x=mxGetImagData(lhs[0]);
        for(i=0;i<N;i++)
        {res[i]=x[i];}
    }

    if(id_x12==1)
    {   res=mxGetData(plhs[0]);
        x0=mxGetData(prhs[0]);
        x=mxGetData(X);
        if(id_x==1)
        {   for(i=0;i<N;i++)
            {res[i]+=mu_x*x[i];}
        }
        for(i=0,j=N;i<N;i++,j++)
        {res[j]=res[i];}
        for(i=0;i<N;i++)
        {res[i]+=mu_x1*x0[i];}
        for(j=N;j<2*N;j++)
        {res[j]+=mu_x2*x0[j];}
        if(isC==1)
        {   res=mxGetImagData(plhs[0]);
            x0=mxGetImagData(prhs[0]);
            x=mxGetImagData(X);
            if(id_x==1)
            {   for(i=0;i<N;i++)
                {res[i]+=mu_x*x[i];}
            }
            for(i=0,j=N;i<N;i++,j++)
            {res[j]=res[i];}
            for(i=0;i<N;i++)
            {res[i]+=mu_x1*x0[i];}
            for(j=N;j<2*N;j++)
            {res[j]+=mu_x2*x0[j];}
        }
        mxDestroyArray(X);
    }
    else
    {   res=mxGetData(plhs[0]);
        x0=mxGetData(prhs[0]);
        for(i=0;i<N;i++)
        {res[i]+=mu_x*x0[i];}
        if(isC==1)
        {   res=mxGetImagData(plhs[0]);
            x0=mxGetImagData(prhs[0]);
            for(i=0;i<N;i++)
            {res[i]+=mu_x*x0[i];}
        }
    }
    mxDestroyArray(lhs[0]);
}
