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

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import Function, dsolve, Eq, Derivative, symbols, solve, latex 
import sympy as sp

# Show app title and description.
st.set_page_config(page_title="Assignment 3")
st.title("Solving IVP and BVP using Numerical Methods")
st.write(
    '''
    This assignment was done by 
    Sanvi Nowal     (CH23BTECH11039)
    Firdose Anjum   (CH23BTECH11017)
    Mahathi Priya   (CH23BTECH11024)
    M S Soummya     (CH23BTECH11028)
    Ceelam Rachana  (CH23BTECH11012)
    Lavudiya Hasini (CH23BTECH11027)
    '''
)

st.subheader("Choose N and P")
with st.form("form"):
    P_in = st.number_input('Enter a value of P', -100.000, 100.000, -4.000, 0.001)
    points = st.slider("Choose number of datapoints (N)", 100, 1000, 500, 10)
    submitted = st.form_submit_button("Submit")

st.write(
        """
        Basic Outline:

        1) Solving the ODEs using BVP- finite differences.
        2) Convert to IVP
        3) Solve the IVP using Implicit Euler technique.
        4) Solve the IVP using Explicit Euler technique but with a smaller step.
        5) Finding the Jacobian and it's eigenvalues.
        6) Finding the particular solution for different values of P analytically.
        """
    )

if submitted:
    st.subheader ("Solving BVP")
    y = np.linspace(0,1,points)
    delta_y = y[1] - y[0]
    P_values = [-2,0,2,5,10]

    N = points
    A = np.zeros((N,N))
    np.fill_diagonal(A, -2)
    for i in range(N):
        for j in range(N):
            if np.abs(i-j) == 1:
                A[i][j] = 1
                A[j][i] = 1 
    A[0][0] = 1 ; A[0][1] = 0
    A[-1][-1] = 1 ; A[-1][-2] = 0

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("black")

    def plot_P (P_values, i):
        for P in P_values:
            b = np.ones(N)
            u = np.zeros(N)
            b = (-P * (delta_y)**2)*b
            b[0] = 0 ; b[-1] = 1
            A_inverse = np.linalg.inv(A)
            u = np.matmul(A_inverse, b)
            ax[i].plot(u, y, label = f'P = {P}')
    
    plot_P ([P_in], 0); plot_P (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="white")
        ax[i].tick_params(axis="y", colors="white")
        ax[i].spines['bottom'].set_color('white')
        ax[i].spines['left'].set_color('white') 
        ax[i].set_facecolor("black")
        ax[i].set_xlabel ('u(y)', color="white")
        ax[i].set_ylabel ('y', color="white")
        ax[i].set_title ('BVP using finite difference', color="white")
        ax[i].legend ()

    st.pyplot (fig)

    st.subheader ("Finding 2 IVPs and solving using shooting method, Explicit Euler")
    Y = np.linspace(0, 1, points)
    h = 1 / points
    P_values = [-2, 0, 2, 5, 10]

    def explicit_euler(s, P):
        u = np.zeros(points)
        u1 = np.zeros(points)
        u[0] = 0 
        u1[0] = s  
        
        for i in range(points-1): 
            u[i+1] = u[i] + h * u1[i]
            u1[i+1] = u1[i] - h * P
        
        return u, u1

    def s_explicit(s, P):
        u, _ = explicit_euler(s, P)
        return u[-1] - 1

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("black")

    def plot_P_expl (P_values, i):
        for P in P_values:
            slope = fsolve(s_explicit, 0, args=(P))[0]
            print("slope: ", slope)
            u_final, _ = explicit_euler(slope, P)
            ax[i].plot(u_final, Y, label=f'P = {P}')
    
    plot_P_expl ([P_in], 0); plot_P_expl (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="white")
        ax[i].tick_params(axis="y", colors="white")
        ax[i].spines['bottom'].set_color('white')
        ax[i].spines['left'].set_color('white') 
        ax[i].set_facecolor("black")
        ax[i].set_xlabel ('u(y)', color="white")
        ax[i].set_ylabel ('y', color="white")
        ax[i].set_title ('IVP using Explicit Euler', color="white")
        ax[i].legend ()

    st.pyplot (fig)

    st.subheader ("Finding 2 IVPs and solving using shooting method, Implicit Euler")

    def implicit_euler(s, P):
        u = np.zeros(points)
        u1 = np.zeros(points)
        u[0] = 0
        u1[0] = s
        for i in range(points-1):
            u1_new = u1[i] - h * P
            u_new = u[i] + h * u1_new
            u[i+1] = u_new
            u1[i+1] = u1_new
        return u, u1

    def s_implicit(s, P):
        u, _ = implicit_euler(s, P)
        return u[-1] - 1  

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("black")

    def plot_P_impl (P_values, i):
        for P in P_values:
            slope = fsolve(s_implicit, 0.5, args=(P))[0]
            print(slope)
            u_final, _ = implicit_euler(slope, P)
            ax[i].plot(u_final, Y, label=f'P = {P}')

    plot_P_impl ([P_in], 0); plot_P_impl (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="white")
        ax[i].tick_params(axis="y", colors="white")
        ax[i].spines['bottom'].set_color('white')
        ax[i].spines['left'].set_color('white') 
        ax[i].set_facecolor("black")
        ax[i].set_xlabel ('u(y)', color="white")
        ax[i].set_ylabel ('y', color="white")
        ax[i].set_title ('IVP using Implicit Euler', color="white")
        ax[i].legend ()

    st.pyplot(fig)

    st.subheader ("Calculating Jacobi Matrix")

    v = sp.symbols('v')
    u = sp.symbols('u')
    dudt = v
    dvdt = -P_in

    def Jacobian(dudt,dvdt):
        dudt_u = sp.diff(dudt, u)
        dudt_v = sp.diff(dudt, v)
        dvdt_u = sp.diff(dvdt, u)
        dvdt_v = sp.diff(dvdt, v)
        J = np.array ([[dudt_u, dudt_v], [dvdt_u, dvdt_v]], dtype = float)

        return J
    
    J = Jacobian (dudt,dvdt)
    st.table (J)

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

    st.write (f"The eigenvalues this matrix calculated by UL Method are: {UL_eigen(J)}")
            
    P_values = [-2,0,2,5,10]

    plt.figure(figsize=(10, 5))
    st.subheader( "SOLUTION BY ANALYTICAL METHODS")
    
    def analytical (P_input, n):
        for P in P_input:
            x = symbols('x')
            f = Function('f')
            ode = Eq(Derivative(f(x), x, x) + P, 0)

            general_solution = dsolve(ode)
            boundary_conditions = {f(0): 0, f(1): 1}
            
            constants = solve([general_solution.rhs.subs(x, 0) - boundary_conditions[f(0)],
                            general_solution.rhs.subs(x, 1) - boundary_conditions[f(1)]])
                            
            particular_solution = general_solution.subs(constants)
            st.write(f"P = {P}, Particular Solution : ")
            st.latex(f"{sp.latex(particular_solution)}")

            x_ = np.linspace(0,1,points)
            y_ = np.zeros(len(x_))
            for i in range(len(x_)):
                y_[i] = particular_solution.rhs.subs(x,x_[i])
            
            ax[n].plot(y_, x_, label = f'P = {P}')

    fig, ax = plt.subplots (2,1,figsize = (12,15))
    fig.patch.set_facecolor("black")
    analytical ([int(P_in)], 0); analytical (P_values, 1)

    for i in range (2):
        ax[i].tick_params(axis="x", colors="white")
        ax[i].tick_params(axis="y", colors="white")
        ax[i].spines['bottom'].set_color('white')
        ax[i].spines['left'].set_color('white') 
        ax[i].set_facecolor("black")
        ax[i].set_xlabel ('u(y)', color="white")
        ax[i].set_ylabel ('y', color="white")
        ax[i].set_title ('Analytical solution', color="white")
        ax[i].legend ()

    st.pyplot (fig)