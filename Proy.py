import numpy as np
print('Sea A y B paginas de internet. En A se encuetra un enlace de B pero en B no se encuentra enlace alguno de A. Esto se representa como:')
mat=[[0,0],[1,0]]
for i in range(2):
	for j in range (2):
		print(' |',mat[i][j], end='')
	print()
print('Podemos notar que la diagonal es 0, pues entre una pagina y la misma (A y A) no se cuentan enlaces que redireccionen a esta.\n--------------------------------------------\n*NOTA*\nLos valores que se ingresen deben ser 0 o 1\n--------------------------------------------')
print('Tomando el ejemplo y las observaciones anteriores. Ingrese la cantidad de paginas que desea analizar:')
D=int(input())
print('Ingrese la cantidad de enlaces que existe entre las respectivas paginas:')
Mat=np.zeros((D,D))
for i in range(D):
	for j in range(D):
		if i!=j:
			print('Matriz['f"{i+1}"']['f"{j+1}"']:')
			Mat[i][j]=float(input())
			A=Mat[i][j]
			while A!=0 and A!=1:
				print('Recuerde que los valores ingresados deben ser 0 o 1')
				Mat[i][j]=float(input())
				A=Mat[i][j]
		else:
			Mat[i][i]=0
			print('Matriz['f"{i+1}"']['f"{j+1}"']:\n0')
print('Sus datos ingresados son:')
for i in range(D):
	for j in range(D):
		print(' |',Mat[i][j], end='')
	print()
k=0
arr=np.zeros(D)
for i in range(D):
	for j in range(D):
		arr[k]+= Mat[j][i]
	k+=1
k=0
for i in range(D):
    for j in range(D):
        Mat[j][i]/=arr[k]
    k+=1
vect=np.ones((D,1))
def EIG_Potencia(A, x_0, epsilon, n_iter, norm, verbose=False):
	x_k = x_0
	lambda_k = 0
	lambda_k_old = -1e100

	for i in range(0, n_iter):
		z_k = np.matmul(A, x_k)
		x_k = z_k/np.linalg.norm(z_k, norm)
		lambda_k = np.matmul(x_k.T, z_k)
		if np.abs(lambda_k - lambda_k_old) < epsilon:
			break;
		lambda_k_old = lambda_k
	return lambda_k
lambd = EIG_Potencia(Mat, vect, 1, 100000,"fro", True)
print("Su nueva matriz sera: ")
for i in range(D):
	for j in range(D):
		if i==j:
			Mat[i][j]=-lambd
		print(' |',"%.1f"%Mat[i][j], end='')
	print()
b=np.zeros(D)
def Gauss(A, b, pivot="none"):
    A_save = A
    A_b = np.matrix(np.c_[A, b]) 
    n_row, n_col = A_b.shape[0], A_b.shape[1]
    process_string = "A_b"
    P = np.eye(n_row)
    L = np.eye(n_row)
    for i in range(0, n_row-1):
        P_i = np.eye(n_row)
        L_i = np.eye(n_row)
        if (A_b[i,i] == 0 and pivot == "partial") or pivot=="total":
            P_i = Pivot_mat(A_b, i)
        A_b = np.matmul(P_i,A_b)
        P = np.matmul(P_i, P)
        L = np.matmul(P_i, L)
        alpha_i = np.zeros(n_row)
        alpha_i[i+1:] = np.transpose(A_b[i+1:n_row,i])/A_b[i,i]
        e_i = np.zeros(n_row)
        e_i[i] = 1
        L_i = np.eye(n_row) - np.outer(alpha_i, e_i)
        L = np.matmul(L_i, L)
        A_b = np.matmul(L_i,A_b)
    U = A_b[:,:n_col-1]
    L = np.linalg.inv(np.matmul(L, np.linalg.inv(P)))
    return P, L, U
def for_solve_triangular(A,b):
    x = np.zeros(A.shape[1])
    rows = A.shape[0]
    for i in range(0, rows):
        if i == 0:
            x[0] = b[0]/A[0,0]
        else:
            pre_sum = A[i,0:i].dot(x[0:i])
            x[i] = (b[i] - pre_sum)/A[i,i]
    return x
def back_solve_triangular(A, b):
    x = np.empty(A.shape[1])
    rows = A.shape[0]
    for i in range(rows-1, -1, -1):
        pre_sum = A[i, i+1:].dot(x[i+1:])
        x[i] = (b[i] - pre_sum)/A[i,i]
    return x
def solve_Gauss(A, b, pivot="none"):
	P, L, U = Gauss(A, b,pivot)
	z = for_solve_triangular(L, np.matmul(P, b))
	x = back_solve_triangular(U, z)
	return x
x=np.zeros(D)
b[0]=2
x=solve_Gauss(Mat, b, "none")
x_ord=solve_Gauss(Mat, b, "none")
for i in range(D):
	x[i]=abs(x[i])
for i in range(D):
	x_ord[i]=abs(x_ord[i])
print('Nuestro autovector es:')
print(x)
for i in range(D):
        for j in range(D-1):
            if(x_ord[j+1] > x_ord[j]):
                aux=x_ord[j]
                x_ord[j]=x_ord[j+1]
                x_ord[j+1]=aux
print('Nuestro autovector ordenado es:')
print(x_ord)
print('Las paginas ordenadas por relevancia son:')
for i in range (D):
    for j in range (D):
        if x_ord[i]==x[j]:
            print(chr(j+65))