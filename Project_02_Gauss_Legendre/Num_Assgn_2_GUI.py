'''
How to run GUI in Linux:
    1) Open Terminal
    2) Install all libraries:
            pip install streamlit
            pip install numpy
            pip install pandas
            pip install sympy
            pip install scipy.linalg
            
    3) Save and run this file
            streamlit run Num_Assgn_GUI.py
            
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import streamlit as st

# Show app title and description.
st.set_page_config(page_title="Calculating roots and weights of Gauss Legendre polynomial", page_icon="🎫")
st.title("Calculating roots and weights of Gauss Legendre polynomial")
'''
This assignment was done by 
Sanvi Nowal     (CH23BTECH11039)
Firdose Anjum   (CH23BTECH11017)
Mahathi Priya   (CH23BTECH11024)
M S Soummya     (CH23BTECH11028)
Ceelam Rachana  (CH23BTECH11012)
Lavudiya Hasini (CH23BTECH11027)
'''

with st.form("Select value of n:"):
    n = st.slider("Select a value for n:", min_value=0, max_value=64, value=42)
    method = st.selectbox("Select Method: ", ["Using Jacobian Matrix", "Generating polynomial and companion matrix"])
    submitted = st.form_submit_button("Generate roots and weights")

def lud(a):
    n = a.shape[0]
    l = np.zeros((n, n))
    u = np.zeros((n, n))
    np.fill_diagonal(l, 1)
    u[0] = a[0]

    for i in range(1, n):
        for j in range(n):
            if i <= j:
                u[i][j] = a[i][j] - sum(u[k][j] * l[i][k] for k in range(i))
            if i > j:
                l[i][j] = (a[i][j] - sum(u[k][j] * l[i][k] for k in range(j))) / u[j][j]
                
    return l, u
    
def shift(A):
    possible_shift_vals = []
    
    for i in range(np.shape(A)[0]):
        up_lim = A[i][i]
        low_lim = A[i][i] 
        
        for j in range(np.shape(A)[0]):
            if i != j :
                up_lim=up_lim+abs(A[i][j])
                low_lim=low_lim-abs(A[i][j])
                
        possible_shift_vals.append(up_lim )
        possible_shift_vals.append(low_lim)    

    shift=np.max(np.abs(possible_shift_vals))
    return shift

def UL_eigen (A, iters= 50000, tol = 1e-15):
    m,n = A.shape 
    I = np.identity (np.shape(A)[0])
    shift_A = shift(A) + 1
    A = A + I * (shift_A)
    
    D1 = A ; D2 = np.ones(np.shape(A))
    iter = 0
  
    while (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==False) :
        L,U = lud(D1)
        D2 = np.matmul (U,L)
        
        if (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==True):
            return np.diagonal(D2) -(shift_A)
            
        D1 = D2
        D2 = np.zeros((m,n))
        iter = iter + 1

        if (iter > iters):
            raise ValueError ("System fails to converge after 50000 iterations. Try another matrix")
            return "NA"
        
def eigenvector_for_eigenvalue(A, eigenvalue, tol=1e-6):
    n = A.shape[0]
    I = np.eye(n)
    matrix = A - eigenvalue * I
    U, S, Vt = np.linalg.svd(matrix)
    null_space_index = np.argmin(S)
    
    if S[null_space_index] > tol:
        raise ValueError("Provided eigenvalue does not correspond to an eigenvector in this matrix.")

    eigenvector = Vt.T[:, null_space_index]
    eigenvector /= np.linalg.norm(eigenvector)
    
    return eigenvector
        
def eigenv (matrix):
    table=[]
    e = UL_eigen(matrix)
    
    for i in range(len(e)):
        vect = eigenvector_for_eigenvalue(matrix, e[i])
        table.append(vect)
        
    eigv = pd.DataFrame(table)
    eigv = eigv.iloc[:,0].values
    return eigv

if submitted:
    if method == "Using Jacobian Matrix":
        st.write(
            """
            Basic Outline: Part 1

            1)  Forming the Jacobi matrix. 
            2)  Calculating the eigenvalues and eigenvectors of this matrix. 
            3)  Eigenvalues of J correspond to the roots of the P_n polynomial.
            4)  Weights = (eigenvector)^2 * [integral(w(x)) limits -1 to 1].
            5)  For legendre polynomial w(x) = 1; Weights = (eigenvector)^2 * 2

            Jacobi Matrix - Tridiagonal matrix

            * For Legendre polynomial of the form: (n+1)P_n+1 = (2n+1)xP_n - (n)P_n-1
            * The diagonal entries of the Jacobi matrix are equal to 0
            * The non-diagonal entries are symmetric and equal to n/(root(2n-1)(2n+1))
            """
        )

        def jacobi_matrix(n):
            matrix = np.zeros([n,n])
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i][j] = 0 
                    elif np.abs(i-j) == 1 and i>j:
                        matrix[i][j] = i/np.sqrt((2*i-1)*(2*i+1))
                        matrix[j][i] = matrix[i][j]
            return matrix

        st.subheader (f"Generating Jacobi matix for Gauss Legendre P_{n}")
        matrix = jacobi_matrix(n)
        st.dataframe (matrix)
        '''
        Note that this is a tridiagonal matrix.
        '''
        def roots_weights(matrix):
            if (n > 8):
                roots, eigenvectors = np.linalg.eigh(matrix)
                weights = np.zeros([len(eigenvectors)])
                weights = (eigenvectors[0] ** 2) * 2
            else:
                roots = UL_eigen(matrix)
                eigenvectors = eigenv (matrix)
                weights = (eigenvectors ** 2) * 2
            
            return roots, weights
        
        st.subheader (f"Generating roots and weights from Jacobi")
        roots, weights = roots_weights(matrix)

        st.subheader (f"Roots and corresponding Weights")
        st.table ([roots, weights])

        st.write (f"Sum of the roots is {np.round(np.sum(roots),10)}, sum of the weights is {np.round(np.sum(weights),10)}")

        fig,ax = plt.subplots(3, 1, figsize=(12, 15))
        fig.patch.set_facecolor("black")
        for i in range (3):
            ax[i].tick_params(axis="x", colors="white")
            ax[i].tick_params(axis="y", colors="white")
            ax[i].spines['bottom'].set_color('white')
            ax[i].spines['left'].set_color('white') 
            ax[i].set_facecolor("black")
            ax[i].set_xlabel (f"Roots of P_{n}", color="white")

        ax[0].set_ylabel (f"Weights of P_{n}", color="white")
        ax[1].set_ylabel (f"Weights of P_{n}", color="white")
        ax[2].set_ylabel (f"Frequency of roots", color="white")
        ax[0].plot (roots, weights, 'cyan') 
        ax[1].scatter (roots, weights, c='darkred') 

        if (n > 30):
            bin = int(n/5)
        else:
            bin = int(n/2)

        ax[2].hist(np.array(roots, dtype=float), bins=bin, color='skyblue', edgecolor='black')
        st.pyplot (fig)

    if method == "Generating polynomial and companion matrix":
        def Gauss_polynomial(n):
            def P(n):
                    x = sp.symbols('x')
                    if n == 0:
                        return 1
                    elif n == 1:
                        return x
                        
                    result = ((2 * n - 1) * x * P(n-1) - (n - 1) * P(n-2)) / n
                    return sp.expand(result)

            def P_above_23(n, P_23, P_24):
                    x = sp.symbols('x')
                    if n == 23:
                        return P_23
                    elif n == 24:
                        return P_24
                        
                    result = ((2 * n - 1) * x * P_above_23 (n-1,P_23, P_24) - (n - 1) * P_above_23 (n-2,P_23, P_24)) / n
                    return sp.expand(result)

            def P_above_38(n, P_38, P_39):
                    x = sp.symbols('x')
                    if n == 38:
                        return P_38
                    elif n == 39:
                        return P_39
                        
                    result = ((2 * n - 1) * x * P_above_38 (n-1,P_38, P_39) - (n - 1) * P_above_38 (n-2,P_38, P_39)) / n
                    return sp.expand(result)
                
            def P_above_52(n, P_52, P_53):
                    x = sp.symbols('x')
                    if n == 52:
                        return P_52
                    elif n == 53:
                        return P_53
                        
                    result = ((2 * n - 1) * x * P_above_52 (n-1,P_52, P_53) - (n - 1) * P_above_52 (n-2,P_52, P_53)) / n
                    return sp.expand(result)

            if (n < 24):
                return P(n)
            
            elif (n > 23 and n < 39):
                P_23 = P(23); P_24 = P(24)
                return P_above_23(n, P_23, P_24)

            elif (n > 38 and n < 52):
                P_23 = P(23); P_24 = P(24)
                P_38 = P_above_23 (38, P_23, P_24); P_39 = P_above_23 (39, P_23, P_24)
                return P_above_38(n, P_38, P_39)
                
            elif (n > 51):
                P_23 = P(23); P_24 = P(24)
                P_38 = P_above_23 (38, P_23, P_24); P_39 = P_above_23 (39, P_23, P_24)
                P_52 = P_above_38(52, P_38, P_39); P_53 = P_above_38(53, P_38, P_39)
                return P_above_52(n, P_52, P_53)

        st.subheader (f"Generating polynomial for n = {n}")   
        legendre_n = Gauss_polynomial(n)
        st.latex (legendre_n)

        def companion (n):
            poly = sp.Poly(Gauss_polynomial (n))
            coeffs = poly.all_coeffs()
            degree = len(coeffs) - 1
            coeffs = [c / coeffs[0] for c in coeffs]
            
            matrix = sp.zeros(degree, degree)
            for i in range(1, degree):
                matrix[i, i - 1] = 1
            arr = np.array([-c for c in coeffs[1:]]).reshape(-1,1)
            matrix[:, -1] = arr [::-1]
            return matrix

        st.subheader (f"Generating Companion Matrix from Polynomial")
        matrix_n = companion (n)
        num_py_mat_n = np.array(matrix_n.tolist())
        st.dataframe (num_py_mat_n)

        st.subheader (f"Finding eigenvalues of companion matrix i.e roots of P_{n}")

        roots = []
        if (n>8):
            eigs = matrix_n.eigenvals()
            for r in eigs.keys():
                roots.append(r.evalf(50))
        else:
            matrix_n1 = np.array(matrix_n, dtype=float)
            eigs = UL_eigen(matrix_n1)
            for r in eigs:
                roots.append(r)

        roots_x = np.sort(roots)
        st.dataframe (np.round(np.array(roots_x, dtype = float).reshape(n,-1),4))

        def weight_i (root, roots):
            prod = 1; x = sp.symbols('x')
            for r in roots:
                if (r == root):
                    continue
                else:
                    prod *= (x - r) / (root - r)
            lagrangian = sp.simplify(prod)
            definite_integral = sp.integrate(lagrangian, (x, -1, 1))
            return sp.expand(definite_integral)

        st.subheader (f"Finding Weights by integrating Lagrangian of the roots")
        weights_n = []
        for j in range (n):
            weights_n.append(weight_i(roots_x[j], roots_x))
        st.dataframe (np.round(np.array(weights_n, dtype = float).reshape(n,-1),4))

        st.write (f"Sum of the roots is {np.round(np.sum(roots_x.astype(float)),10)}, Sum of the weights is {np.round(np.sum(np.array(weights_n).astype(float)),10)}.")
        fig,ax = plt.subplots(3, 1, figsize=(12, 15))
        fig.patch.set_facecolor("black")
        for i in range (3):
            ax[i].tick_params(axis="x", colors="white")
            ax[i].tick_params(axis="y", colors="white")
            ax[i].spines['bottom'].set_color('white')
            ax[i].spines['left'].set_color('white') 
            ax[i].set_facecolor("black")
            ax[i].set_xlabel (f"Roots of P_{n}", color="white")

        ax[0].set_ylabel (f"Weights of P_{n}", color="white")
        ax[1].set_ylabel (f"Weights of P_{n}", color="white")
        ax[2].set_ylabel (f"Frequency of roots", color="white")
        ax[0].plot (roots_x, weights_n, 'cyan') 
        ax[1].scatter (roots_x, weights_n, c='darkred') 

        if (n > 30):
            bin = int(n/5)
        else:
            bin = int(n/2)

        ax[2].hist(np.array(roots_x, dtype=float), bins=bin, color='skyblue', edgecolor='black')
        st.pyplot (fig)