import numpy as np
cimport cython
DTYPE = np.intc	

@cython.boundscheck(False)
@cython.wraparound(False)   
cpdef mF_fmed(int[:] sol, int[:,:] orders):
    """
    Crea la matriz de tiempos de finalización Fij
    para una solución sol y ordenes de trabajo orders.
    """
    cdef int i,j,n,m,c,index,indexprev
    cdef float result,sum
    n=len(sol)
    m=orders.shape[1]
    mat0 = np.zeros((n,m), dtype=DTYPE)
    c=0
    index=sol[0]-1
    for i in range(m):
        c+=orders[index][i]
        mat0[index][i] = c
    sum=mat0[index,-1]
    cdef int[:, :] mat = mat0
    for i in range(1,n): #ordenes a partir de la primera
        index=sol[i]-1
        indexprev=sol[i-1]-1
        mat[index,0]=mat[indexprev,0]+orders[index,0] #primera máquina
        for j in range(1,m): #máquinas a partir de la primera
            mat[index,j]=max(mat[indexprev,j],mat[index,j-1])+orders[index,j]        
        sum+=mat[index,m-1]
    result=sum/n
    return result

@cython.boundscheck(False)
@cython.wraparound(False)   
cdef rz(int[:] perm, int[:,:] orders):
    cdef int n,i,j,k
    cdef float yvalue,best_value,bestyvalue
    n = len(perm)
    b0=np.copy(perm)
    z0=np.empty(n-1,dtype=DTYPE)
    y0=np.empty(n,dtype=DTYPE)
    besty0=np.empty(n,dtype=DTYPE)
    cdef int[:] y = y0
    cdef int[:] z = z0
    cdef int[:] b = b0
    cdef int[:] besty = besty0
    best_value=np.inf  
    
    for val in perm:
        bestyvalue=np.inf
        #z=np.delete(b, np.where(b == val))
        i=0
        while b[i]!=val:
            z[i]=b[i]
            i+=1
        for k in range(i,n-1):
            z[k]=b[k+1]
        ##
        for j in range(0,n):                    
            #y=np.insert(z,j,val)
            for i in range(n):
                if i<j:
                    y[i]=z[i]
                elif i>j:
                    y[i]=z[i-1]
                else:
                    y[i]=val
           ##
            yvalue=mF_fmed(y,orders) 
            if yvalue<bestyvalue:
                bestyvalue=yvalue
                besty[...]=y
        if bestyvalue<best_value:          
            best_value=bestyvalue
            b[...]=besty
    return best_value,b0

@cython.boundscheck(False)
@cython.wraparound(False)  
def irz(int[:] perm, int[:,:] orders):
    cdef float v,last_v    
    v,s = rz(perm,orders)
    last_v=np.inf
    while last_v>v:
        last_v=v
        v,s = rz(s,orders)
    return v,s