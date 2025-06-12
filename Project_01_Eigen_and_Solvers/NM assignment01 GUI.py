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
import pandas as pd
import numpy as np
import math
import sympy as sp
import scipy.linalg as la
import itertools

st.set_page_config(
    page_title='Numerical Methods: Eigenvalues and solutions',
    page_icon=':books:',
)

'''
# :books: Numerical Methods: Eigenvalues and solutions

In this assignment, we will calculate eigenvalues using power method, 
UL Method and further use these eigenvalues to calculate determinant, 
uniqueness, condition number and characteristic polynomial. Also, we 
will find solution of matrix form _Ax = b_.

This assignment is by:
ch23btech11039, ch23btech11028, ch23btech11024, ch23btech11017, 
ch23btech11012, ch23btech11027. 
'''

#A = np.array ([[3,1,2],[2,4,1],[1,2,7]])
option = st.selectbox("How would you like to provide the matrix?", 
                      ("Manual Input", "Upload File"))

in_or_not = 0; type = 0
if (option == "Manual Input"):
    n = st.number_input("Enter the size of the matrix (n x n)", min_value=1, value=3, step=1)
    matrix = np.zeros((n, n))

    st.write(f"Input a {n}x{n} matrix:")
    for i in range(n):
        cols = st.columns(n)  # Create n columns for the matrix row
        for j in range(n):
            matrix[i, j] = cols[j].number_input(f"({i+1},{j+1})", value=0.0, key=f"input_{i}_{j}")
            A = matrix
    in_or_not = 1; type = 0

else:
    type = 1
    uploaded_file = st.file_uploader("Upload your matrix file (CSV or Excel)", type=["csv", "xlsx"])
    n = st.number_input("Enter the size of the matrix (n x n)", min_value=1, value=3, step=1)

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):                             
            matrix = pd.read_csv(uploaded_file).values  
        else:
            matrix = pd.read_excel(uploaded_file).values

        row_num = st.number_input("Enter row of file to enter", min_value=0)
        required_values = n * n

        try:
            if (matrix.shape[1] < required_values):
                raise ValueError(f"Expected at least {required_values} values, "
                                f"but found {matrix.shape[1]}.")
            
            else:
                st.write ("All rows meet the required number of values.")
                in_or_not = 1

        except ValueError as e:
            st.error (e)

        for value in (matrix[row_num]):
            try:
                if pd.isna(value):
                    raise ValueError(f"Value is NaN: {value}")
                
                else:
                    value = float(value)
            
            except ValueError:
                st.error(f"Cannot convert {value} to float. Non-numeric value encountered.")
                in_or_not = 0

if (in_or_not > 0):
    if type != 0:
        A = matrix [row_num][:(n*n)].reshape (n,n)
        st.dataframe(pd.DataFrame(A))

    eigenvalues = la.eig (A)[0]
    if np.any(np.iscomplex(eigenvalues)):
            st.error ("Error: The matrix has complex eigenvalues.")
    else:
        st.write ("All eigenvalues are real.")

    ''''''
    operations = st.multiselect("Select which operation to perform:", 
                                ['UL Method', 'Power Method', 'Solution Ax=b'], 
                                default=['UL Method', 'Power Method', 'Solution Ax=b'])
    
    if 'UL Method' in operations:
        st.title("Calculating eigenvalues using UL Method")
        st.subheader("LUD Decomposition of input matrix")

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

        L,U = lud(A)
        st.write ("L: ") ; st.dataframe(L)
        st.write ("U: ") ; st.dataframe(U)

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

        def UL_eigen (A, tol = 1e-15):
            m,n = A.shape; iter = 0
            I = np.identity (np.shape(A)[0])
            shift_A = shift(A) + 1
            A = A + I * (shift_A)
            
            D1 = A ; D2 = np.ones(np.shape(A))
            while (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==False):
                L,U = lud(D1)
                D2 = np.matmul (U,L)
                
                if (np.allclose(np.diagonal (D1), np.diagonal (D2), tol)==True):
                    return np.diagonal(D2) -(shift_A)
                    
                D1 = D2
                D2 = np.zeros((m,n))
                iter = iter + 1

                if (iter > 5000):
                    st.error ("System fails to converge after 5000 iterations. Try another matrix")
                    return "NA"

        '''
        ## Eigenvalues after performing UL Method
        '''
        eigen = UL_eigen (A)
        if (np.all(eigen) != 'NA'):
            eigen_str = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;".join([f"{val:0.4f}" for val in eigen])
            st.markdown(f"""
                <div style="border:2px solid #4CAF50;padding:10px;border-radius:5px; text-align:center;">
                    {eigen_str}
                </div>
                """, unsafe_allow_html=True)

        '''
        '''
        functions = ['Find Condition Number', 'Check Uniqueness', 'Find Determinant',
                    'Find characteristic polynomial' ]
        selected_value = st.selectbox("Choose an function:", functions)

        if (selected_value == 'Find characteristic polynomial'):
            def eqn(matrix):
                eigen = np.linalg.eigvals(matrix)
                product_of_roots = [1]
                dimension = np.shape(matrix)[0]

                for i in range(dimension):
                    combination = itertools.combinations(eigen, i + 1)
                    products = [math.prod(roots) for roots in combination]
                    product_of_roots.append(((-1) ** (i + 1)) * np.sum(products))

                x = sp.symbols("x")
                st.write("Characteristic polynomial: ")
                
                equation = ""
                n = len(product_of_roots)
                
                for i in range(1, n + 1):
                    coefficient = product_of_roots[i - 1]
                    power = n - i
                    if power > 0:
                        equation += f"({coefficient:0.2f})x^{power} + "
                    else:
                        equation += f"({coefficient:0.2f})"
                st.latex(equation)

            eqn (A)

        if (selected_value == 'Check Uniqueness'):
            det=np.prod(eigen)
            det=np.round(det,4)

            st.write(f"The determinant of a matrix |A| is {det:0.4f}")

            if np.allclose(det,0) :
                st.write("The system has either no solutions or infinite solutions and is inconsistent.")
            else:
                st.write("The system has unique solutions and is consistent.")
        
        if (selected_value == 'Find Condition Number'):
            def condition_num(eigenvalues):
                n = len (eigenvalues)
                max = np.max(eigenvalues)
                min = np.min(eigenvalues)
                condition_number = max/min
                
                hilbert = np.zeros([n,n])
                for i in range(0,n):
                    for j in range(0,n):
                        hilbert[i][j] = 1/(i+j+1)

                st.write ("Hilbert's Matrix: ")
                st.dataframe (hilbert)

                eigen_hilbert = UL_eigen(hilbert)
                condition_hilbert = np.max(eigen_hilbert)/np.min(eigen_hilbert)
                
                if condition_hilbert > condition_number:
                    st.write(f'It is well-conditioned matrix with Condition number = {condition_number:0.4f}, Hilbert matrix = {condition_hilbert:0.4f}')
                else:
                    st.write(f'It is ill - conditioned matrix with Condition number = {condition_number:0.4f}, Hilbert matrix = {condition_hilbert:0.4f}')
                return hilbert

            hilbert = condition_num (eigen)

            ''' __Note:__ '''
            st.write('This formula of Condition Number = (max eigenvalue) / (min eigenvalue) is only applicable if A is a normal matrix. Otherwise, we use function of (maximum singular value) / (minimum singular value), where singular values can be found by performing SVD of the matrix.')
            '''
            '''
            _,s,_ = la.svd (A); _,s1,_ = la.svd (hilbert)
            st.write (f"If calculated using SVD, we get: Matrix condition number = {s[0]/s[-1]:0.4f} and Hilbert condition number = {s1[0]/s1[-1]:0.4f}.")

        if (selected_value == 'Find Determinant'):
            st.write ("Determinant of a matrix is the product of its eigenvalues.")
            st.write (f"Determinant of the chosen matrix is {np.prod (UL_eigen(A)):0.4f}")

    if 'Power Method' in operations:
        st.title ("Calculating eigenvalues using Power Method")
        st.subheader("Power method for input matrix A")

        def power_method (A, x_input, iters = 5000, tol = 1e-15):
            b = np.zeros ((x_input.shape[0], iters))
            c = np.zeros ((iters))
            
            b_in = np.matmul (A, x_input) 
            c[0] = np.max (b_in)
            x_in = b_in / c[0]

            for i in range (iters):
                b = np.matmul(A, x_in)
                c[i+1] = np.max (b)
                x = b / c[i+1]
               
                if (i == iters - 2):
                    st.error ("System fails to converge after 5000 iterations. Try another matrix")
                    return "Returned with no value"

                if (np.allclose(c[i+1], c[i], tol)):
                    return c[i+1]
                else:
                    c[i+2] = c [i+1]
                    x_in = x

        st.write ("Press Ctrl+Enter to enter the values")
        user_input = st.text_area("Enter initial vector for power methods (separate elements using commas):")
        if user_input:
            try:
                array = np.array([float(x) for x in user_input.split(',')])
                if array.shape[0] != n:
                    st.error(f"Dimension mismatch: Expected {n} elements, but got {array.shape[0]}.")
                
                else:   
                    max_eig = power_method (A, array)
                    st.write (f"Maximum eigenvalue detected: {np.array(max_eig)}")

            except ValueError:
                st.write("Please enter valid numbers separated by commas.")

        zero_check = 0
        def gauss_jordan(sample_matrix):
                global zero_check
                n = len(sample_matrix)
                augmented_matrix = np.hstack((sample_matrix, np.identity(n)))  # Augment with identity matrix

                for i in range(n):
                    # Partial pivoting: Find the maximum element in the current column
                    max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
                    if augmented_matrix[max_row, i] == 0:
                        st.error("Matrix is singular and cannot be inverted.")
                        zero_check = 1

                    # Swap rows to place the largest element on the diagonal
                    if i != max_row:
                        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
                    
                    # Make the diagonal element 1 by dividing the row by the diagonal element
                    diag_element = augmented_matrix[i][i]
                    augmented_matrix[i] = augmented_matrix[i] / diag_element

                    # Make all elements in the current column, except the diagonal element, 0
                    for j in range(n):
                        if i != j:
                            multiplication_factor = augmented_matrix[j][i]
                            augmented_matrix[j] = augmented_matrix[j] - multiplication_factor * augmented_matrix[i]

                # Extract the inverse matrix from the augmented matrix
                inverse_matrix = augmented_matrix[:, n:]
                return inverse_matrix
        
        det = np.round(np.prod(UL_eigen(A)),4)
        if np.allclose(det,0):
            st.error ("Determinant = 0, Matrix is singular, non-invertible.")
            zero_check = 1
        else:
            A_inv = gauss_jordan(A)

        if (zero_check == 0):
            st.write ("Inverse of A calculated using Gauss-Jordan: ")
            st.dataframe (A_inv)

            max_eig_inv = power_method (A_inv, array)
            if (isinstance(max_eig_inv, float)):
                st.write (f"Maximum eigenvalue of A inverse detected is {max_eig_inv:0.4f}. Notice that 1/(max eigen of A inverse) = {1/max_eig_inv:0.4f}.")
                st.write (f"These values correspond to the maximum and minimum eigenvalues of A obtained by UL Decomposition.")

    if 'Solution Ax=b' in operations:
        '''
        # Solution Ax=b
        '''
        def solve_LU(sample_matrix, b):
            L,U = lud(sample_matrix)
            y = np.zeros(len(b))
            x = np.zeros(len(b))
            
            # Ly = b
            y[0] = b[0]
            for i in range(1, len(b)):
                sum_Ly = 0
                for j in range(i):
                    sum_Ly += L[i][j] * y[j]
                y[i] = b[i] - sum_Ly

            #Ux = y
            x[-1] = y[-1] / U[-1][-1]
            for i in range(len(b) - 2, -1, -1):
                sum_Ux = 0
                for j in range(i + 1, len(b)):
                    sum_Ux += U[i][j] * x[j]
                x[i] = (y[i] - sum_Ux) / U[i][i]

            if (np.isfinite(x)).any():
                st.write("Solution vector x:", x)
            else:
                st.write("Solution not possible.")
            return x

        st.write ("Press Ctrl+Enter to enter the values")
        user_input = st.text_area("Enter solution vector b (separate elements using commas):")
        if user_input:
            #try:
            b = np.array([float(t.strip()) for t in user_input.split(',')])
            if b.shape[0] != n:
                st.error(f"Dimension mismatch: Expected {n} elements, but got {b.shape[0]}.")
            
            else:   
                x = solve_LU(A, b)

            #except ValueError:
                #st.write("Please enter valid numbers separated by commas.")
        