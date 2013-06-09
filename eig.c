#include <stdio.h>
#include <math.h>
#include "eig.h"
#include "matrix.h"
#include "tracker.h"

/*----------------------------------------------------------------------------*/

void eig (Matrix *M, Matrix *Vec, Matrix *Val)
{
  Matrix sizeM = sizeOfMatrix(*M);
  int dim = elem(sizeM, 0, 0);
  freeMatrix(sizeM); 
  float d[dim+1];
  float e[dim+1];
 
  //*Vec = addShiftMatrix(*M);
  Matrix Vec_tmp = addShiftMatrix(*M);
 
  //tred2(*Vec, dim, d, e);
  tred2(Vec_tmp, dim, d, e);
  
  //tqli(d, e, dim, *Vec);
  tqli(d, e, dim, Vec_tmp);
  
  //*Vec = subShiftMatrix(*Vec);
  Matrix Vec_tmp2 = subShiftMatrix(Vec_tmp);
  
  //*Val = getValMatrix(d, dim);
  appMatrix( *Vec, 0, dim-1, 0, dim-1, Vec_tmp2, 0, dim-1, 0, dim-1 );
  
  Matrix Val_tmp = getValMatrix(d, dim);

  appMatrix( *Val, 0, dim-1, 0, dim-1, Val_tmp, 0, dim-1, 0, dim-1 );
  
  freeMatrix(Vec_tmp);
  freeMatrix(Vec_tmp2);
  freeMatrix(Val_tmp);
}

/*----------------------------------------------------------------------------*/

void tred2(Matrix M, int n, float d[], float e[])
{
  Lines a = M->lines;
  
	int l,k,j,i;
	float scale,hh,h,g,f;
    int iter = 0;

	for (i=n;i>=2;i--) {
		l=i-1;
		h=scale=0.0;
		if (l > 1) {
			for (k=1;k<=l;k++)
				scale += fabs(a[i][k]);
			if (scale == 0.0)
				e[i]=a[i][l];
			else {
				for (k=1;k<=l;k++) {
					a[i][k] /= scale;
					h += a[i][k]*a[i][k];
				}
				f=a[i][l];
				g=(f >= 0.0 ? -sqrt(h) : sqrt(h));
				e[i]=scale*g;
				h -= f*g;
				a[i][l]=f-g;
				f=0.0;
				for (j=1;j<=l;j++) {
					a[j][i]=a[i][j]/h;
					g=0.0;
					for (k=1;k<=j;k++)
						g += a[j][k]*a[i][k];
					for (k=j+1;k<=l;k++)
						g += a[k][j]*a[i][k];
					e[j]=g/h;
					f += e[j]*a[i][j];
				}
				hh=f/(h+h);
				for (j=1;j<=l;j++) {
					f=a[i][j];
					e[j]=g=e[j]-hh*f;
				  for (k=1;k<=j;k++)
						a[j][k] -= (f*e[k]+g*a[i][k]);
				}
			}
		} else
			e[i]=a[i][l];
		d[i]=h;
	}
	d[1]=0.0;
	e[1]=0.0;
	/* Contents of this loop can be omitted if eigenvectors not
			wanted except for statement d[i]=a[i][i]; */
	for (i=1;i<=n;i++) {
		l=i-1;
		if (d[i]) {
			for (j=1;j<=l;j++) {
				g=0.0;
				for (k=1;k<=l;k++)
					g += a[i][k]*a[k][j];
        for (k=1;k<=l;k++)
					a[k][j] -= g*a[k][i];
			}
		}
		d[i]=a[i][i];
		a[i][i]=1.0;
		for (j=1;j<=l;j++) a[j][i]=a[i][j]=0.0;
	}
}

/*----------------------------------------------------------------------------*/

void tqli(float d[], float e[], int n, Matrix z0)
{
  Lines z = z0->lines; 
	float pythag(float a, float b);
	int m,l,iter,i,k;
	float s,r,p,g,f,dd,c,b;

	for (i=2;i<=n;i++) e[i-1]=e[i];
	e[n]=0.0;
	for (l=1;l<=n;l++) {
		iter=0;
		do {
		for (m=l;m<=n-1;m++) {
				dd=fabs(d[m])+fabs(d[m+1]);
				if ((float)(fabs(e[m])+dd) == dd)
        {
            break;
           }
			}
			if (m != l) {
				if (iter++ == 30) printf("Too many iterations in tqli\n");
				g=(d[l+1]-d[l])/(2.0*e[l]);
				r=pythag(g,1.0);
				g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
				s=c=1.0;
				p=0.0;
				for (i=m-1;i>=l;i--) {
					f=s*e[i];
					b=c*e[i];
					e[i+1]=(r=pythag(f,g));
					if (r == 0.0) {
						d[i+1] -= p;
						e[m]=0.0;
						break;
					}
					s=f/r;
					c=g/r;
					g=d[i+1]-p;
					r=(d[i]-g)*s+2.0*c*b;
					d[i+1]=g+(p=s*r);
					g=c*r-b;
					for (k=1;k<=n;k++) {
						f=z[k][i+1];
						z[k][i+1]=s*z[k][i]+c*f;
						z[k][i]=c*z[k][i]-s*f;
					}
				}
				if (r == 0.0 && i >= l) continue;
				d[l] -= p;
				e[l]=g;
				e[m]=0.0;
			}
		} while (m != l);
	}
}

/*----------------------------------------------------------------------------*/

float pythag(float a, float b)
{
	float absa,absb;
	absa=fabs(a);
	absb=fabs(b);
	if (absa > absb) return absa*sqrt(1.0+powf(absb/absa,2));
	else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+powf(absa/absb,2)));
}

/*----------------------------------------------------------------------------*/

Matrix
addShiftMatrix (Matrix m)
{
  int i, j;
  int w = m->width;
  int h = m->height;
  Matrix res = zeroMatrix (w+1, h+1);
  Lines rr = res->lines;
  Lines mr = m->lines;
  
  for (i = 0; i < w; ++i)
  {
    for (j = 0; j < h; ++j)
    {
      rr[i+1][j+1] = mr[i][j];
    }
  }
  return res;
}

/*----------------------------------------------------------------------------*/

Matrix
subShiftMatrix (Matrix m)
{
  int i, j;
  int w = m->width;
  int h = m->height;
  Matrix res = zeroMatrix (w-1, h-1);
  Lines rr = res->lines;
  Lines mr = m->lines;
  
  for (i = 0; i < w-1; ++i)
  {
    for (j = 0; j < h-1; ++j)
    {
      rr[i][j] = mr[i+1][j+1];
    }
  }
  return res;
}

/*----------------------------------------------------------------------------*/

Matrix
getValMatrix (float d[], int dim)
{
  Matrix res = zeroMatrix (dim, dim);
  Lines r = res->lines;
  int i;
  for (i = 0; i < dim; i++)
  {
      r[i][i] = d[i+1];
  }
  return res;            
}




